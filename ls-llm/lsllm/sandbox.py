"""Run untrusted Python (model output) in a fresh interpreter inside the macOS sandbox.

run_python(code) puts the code into a private temporary directory and runs it with the base CPython behind
the virtualenv (`-I -S`: isolated mode, no site-packages, so only the standard library can be imported) under
`sandbox-exec` with a deny-by-default profile: no network, no child processes, no file writes at all, no file
reads under the home directory except the interpreter installation. Stdout and stderr are two files of the
parent (not pipes, which the parent would have to drain); their first 64 KB are returned.

Limits. The parent is the watchdog: every POLL_S seconds it looks at the clock and at the child's memory
footprint, and it kills the child's whole process group when the child runs too long or uses too much memory.
(The memory rlimits of macOS do not do this: see _footprint.) Two kernel limits, CPU seconds and file size,
are set by a one-line shell prelude; they hold even if the parent is killed, so a child that lost its
watchdog cannot spin forever and cannot write more than FILE_MB to stdout or to stderr.
"""
import ctypes
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

PYTHON = os.path.realpath(sys.executable)        # the base interpreter, not the virtualenv symlink
PYTHON_HOME = os.path.realpath(sys.base_prefix)  # its installation (binary and standard library)
HOME = os.path.realpath(os.path.expanduser("~"))
POLL_S = 0.01            # watchdog period
OUTPUT_BYTES = 65536     # captured stdout and stderr are cut to this length
FILE_MB = 2              # kernel limit on the size of a file, here of stdout and of stderr
MB = 2 ** 20

# Later rules win over earlier ones. Reading "/" itself (the directory entry, not its content) is needed by
# the dynamic loader. Everything not allowed here is denied: sockets, fork, exec of other programs, IPC,
# creating or changing any file. TMP is the private directory that holds the program.
PROFILE = """(version 1)
(deny default)
(allow process-exec (literal (param "PYTHON")))
(allow sysctl-read)
(allow file-read* (literal "/") (subpath "/usr") (subpath "/System") (literal "/dev/null") (literal "/dev/urandom"))
(deny file-read* (subpath (param "HOME")))
(allow file-read* (subpath (param "PYTHON_HOME")) (subpath (param "TMP")))
(allow file-write* (literal "/dev/null"))
"""
# The shell sets the two kernel limits and then replaces itself with the command, so the limits (and the
# process id) carry over to sandbox-exec and from there to Python. bash counts `ulimit -f` in KiB.
PRELUDE = 'ulimit -t "$1"; ulimit -f "$2"; shift 2; exec "$@"'

_libc = ctypes.CDLL(None)


@dataclass
class Result:
    status: str      # "ok" (exit code 0) | "error" (exception or crash) | "timeout" | "memory"
    stdout: str
    stderr: str
    seconds: float   # wall-clock time of the run
    peak_mb: float   # highest memory footprint the watchdog saw


def _footprint(pid):
    """Memory footprint of a process in bytes (0 if it is gone).

    This is the "Memory" column of Activity Monitor. Unlike the resident size it also counts pages that
    macOS has compressed, so a process cannot get under the limit by being compressed. We poll it because
    the rlimits do not limit memory here: a 512 MB RLIMIT_AS is refused (a fresh process already has 466 GB
    of address space), and a limit of "current size + 192 MB" stops one big block but not growth in small
    pieces (measured: 600 MB taken in 1 MB pieces without an error).
    """
    info = (ctypes.c_uint64 * 12)()    # struct rusage_info_v0 of <sys/resource.h>, as 64-bit words
    if _libc.proc_pid_rusage(pid, 0, info) != 0:
        return 0
    return info[9]                     # ri_phys_footprint


def _watch(proc, deadline, memory_bytes):
    """Wait for the child. Return (why it must be killed, or None if it ended by itself; peak footprint)."""
    peak = 0
    while True:
        try:
            proc.wait(timeout=POLL_S)
            return None, peak
        except subprocess.TimeoutExpired:
            pass
        peak = max(peak, _footprint(proc.pid))
        if time.monotonic() > deadline:
            return "timeout", peak
        if peak > memory_bytes:
            return "memory", peak


def _kill(proc):
    """Kill the child's whole process group if it is still there, and collect the child."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass       # it ended by itself in this instant; macOS answers EPERM for a group that holds only a zombie
    proc.wait()


def _head(f):
    """The first OUTPUT_BYTES that the child wrote to the file `f`."""
    f.seek(0)
    return f.read(OUTPUT_BYTES).decode("utf-8", "replace")


def run_python(code, timeout_s=5.0, memory_mb=512):
    """Run `code` as a script in the sandbox and return a Result. Nothing is shared with this process."""
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="lsllm_"))
    command = ["/bin/bash", "-c", PRELUDE, "bash", str(int(timeout_s) + 2), str(FILE_MB * 1024),
               "/usr/bin/sandbox-exec", "-D", "PYTHON=" + PYTHON, "-D", "PYTHON_HOME=" + PYTHON_HOME,
               "-D", "HOME=" + HOME, "-D", "TMP=" + tmp, "-p", PROFILE, PYTHON, "-I", "-S", "main.py"]
    try:
        with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:     # files without a name
            start = time.monotonic()
            # start_new_session: the child leads its own process group, so killpg reaches all of it
            proc = subprocess.Popen(command, cwd=tmp, env={}, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    start_new_session=True)
            try:
                killed, peak = _watch(proc, start + timeout_s, memory_mb * MB)
            finally:   # also reached on Ctrl-C: never leave the child running
                _kill(proc)
            seconds = time.monotonic() - start
            stdout, stderr = _head(out), _head(err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if killed:
        status = killed                           # "timeout" or "memory"
    elif proc.returncode == -signal.SIGXCPU:      # the kernel CPU limit fired before the watchdog
        status = "timeout"
    else:
        status = "ok" if proc.returncode == 0 else "error"
    return Result(status, stdout, stderr, seconds, peak / MB)


def run_many(codes, workers=4, timeout_s=5.0, memory_mb=512):
    """Run many programs, `workers` at a time. The results are in the order of `codes`."""
    pool = ThreadPoolExecutor(workers)
    try:
        return list(pool.map(lambda code: run_python(code, timeout_s, memory_mb), codes))
    finally:
        pool.shutdown(cancel_futures=True)       # on Ctrl-C: do not start the runs still waiting
