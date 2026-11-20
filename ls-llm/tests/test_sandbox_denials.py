"""The sandbox must run ordinary code and refuse everything else. Each test proves one denial."""
import os
import socket
import tempfile
import time
from pathlib import Path

from lsllm import sandbox
from lsllm.sandbox import FILE_MB, OUTPUT_BYTES, run_many, run_python


def test_plain_code_runs_with_the_standard_library_only():
    r = run_python("import json, math, re, sys\nprint(json.dumps([math.gcd(12, 18), sys.flags.isolated, sys.flags.no_site]))")
    assert (r.status, r.stdout, r.stderr) == ("ok", "[6, 1, 1]\n", "")
    r = run_python("import numpy")      # installed in the virtualenv, must not be visible inside
    assert r.status == "error" and "ModuleNotFoundError" in r.stderr


def test_exception_is_an_error():
    r = run_python("print('before')\nraise ValueError('boom')")
    assert r.status == "error" and r.stdout == "before\n" and "ValueError: boom" in r.stderr


def test_no_network():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(0.5)
    port = server.getsockname()[1]
    socket.create_connection(("127.0.0.1", port)).close()    # control: outside the sandbox the server is reachable
    server.accept()[0].close()
    r = run_python(f"import socket\nsocket.create_connection(('127.0.0.1', {port}), timeout=2)\nprint('CONNECTED')")
    assert r.status == "error" and "PermissionError" in r.stderr and "CONNECTED" not in r.stdout
    try:
        server.accept()
        raise AssertionError("the sandboxed program reached the server")
    except socket.timeout:
        pass
    server.close()
    r = run_python("import socket\ns = socket.socket()\ns.bind(('127.0.0.1', 0))\nprint('BOUND')")
    assert r.status == "error" and "PermissionError" in r.stderr and "BOUND" not in r.stdout
    r = run_python("import urllib.request\nurllib.request.urlopen('http://example.com', timeout=2)\nprint('FETCHED')")
    assert r.status == "error" and "FETCHED" not in r.stdout


def test_no_file_writes_anywhere():
    targets = [Path.home() / ".lsllm_sandbox_must_not_write", Path(__file__).with_name("sandbox_must_not_write.txt"),
               Path(tempfile.gettempdir()) / "lsllm_sandbox_must_not_write"]
    try:
        for target in targets:
            r = run_python(f"open({str(target)!r}, 'w').write('x')\nprint('WROTE')")
            assert r.status == "error" and "PermissionError" in r.stderr and "WROTE" not in r.stdout
            assert not target.exists()
        r = run_python(f"import os\nos.remove({__file__!r})\nprint('REMOVED')")
        assert r.status == "error" and "REMOVED" not in r.stdout and Path(__file__).exists()
        # not in its own directory either: a program cannot create, change or delete any file
        for code in ("open('note.txt', 'w')", "import os\nos.mkdir('d')", "import os\nos.remove('main.py')",
                     "open(__file__, 'a')", "import tempfile\ntempfile.TemporaryFile()"):
            r = run_python(code + "\nprint('WROTE')")
            assert r.status == "error" and "WROTE" not in r.stdout, code
    finally:
        for target in targets:
            target.unlink(missing_ok=True)


def test_private_directory_holds_only_the_program_and_is_removed_afterwards():
    r = run_python("import os\nprint(os.listdir('.'), open(__file__).read().count('listdir'))\nprint(os.getcwd())")
    lines = r.stdout.split("\n")
    assert r.status == "ok" and lines[0] == "['main.py'] 2"       # it can read its own text: "listdir" occurs twice
    assert lines[1].startswith(os.path.realpath(tempfile.gettempdir())) and not os.path.exists(lines[1])


def test_no_reads_under_the_home_directory():
    for code in (f"print(open({__file__!r}).read())", f"import os\nprint(os.listdir({str(Path.home())!r}))",
                 f"import os\nprint(os.stat({__file__!r}))", "print(open('/etc/passwd').read())"):
        r = run_python(code)
        assert r.status == "error" and "PermissionError" in r.stderr and r.stdout == ""
    r = run_python("import os\nprint(len(open(os.__file__).read()) > 1000)")   # the interpreter's own files are readable
    assert (r.status, r.stdout) == ("ok", "True\n")


def test_no_child_processes():
    for code in ("import os\nos.fork()\nprint('STARTED')",
                 "import subprocess\nsubprocess.run(['/bin/echo', 'x'])\nprint('STARTED')",
                 "import os\nos.posix_spawn('/bin/echo', ['echo', 'x'], {})\nprint('STARTED')",
                 "import os\nos.execv('/bin/echo', ['echo', 'STARTED'])"):
        r = run_python(code)
        assert r.status == "error" and "PermissionError" in r.stderr and "STARTED" not in r.stdout
    r = run_python("import os\nprint(os.system('echo STARTED') != 0)")
    assert r.stdout == "True\n"


def test_cannot_signal_other_processes():
    r = run_python("import os\nos.kill(os.getppid(), 0)\nprint('SIGNALLED')")
    assert r.status == "error" and "PermissionError" in r.stderr and "SIGNALLED" not in r.stdout


def test_environment_is_not_passed_on(monkeypatch):
    monkeypatch.setenv("LSLLM_SECRET", "hunter2")
    r = run_python("import os\nprint(sorted(os.environ))")
    assert r.status == "ok" and "LSLLM_SECRET" not in r.stdout and "PATH" not in r.stdout and "HOME" not in r.stdout


def test_infinite_loop_is_killed():
    r = run_python("import os\nprint(os.getpid(), flush=True)\nwhile True:\n    pass", timeout_s=1.0)
    assert r.status == "timeout" and 1.0 <= r.seconds < 5.0
    try:
        os.kill(int(r.stdout), 0)
        raise AssertionError("the child is still alive")
    except ProcessLookupError:
        pass


def test_sleeping_program_is_killed():
    r = run_python("import time\ntime.sleep(60)", timeout_s=1.0)     # wall-clock time, not CPU time
    assert r.status == "timeout" and r.seconds < 5.0


def test_memory_blow_up_is_killed_early():
    r = run_python("x = bytearray(b'a') * (2 * 1024 ** 3)\nprint('ALLOCATED')", memory_mb=100)   # about 2 GB
    assert r.status == "memory" and "ALLOCATED" not in r.stdout
    assert r.seconds < 5.0 and 100 < r.peak_mb < 1024      # stopped long before 2 GB were in use
    # growth in small pieces; the program stops by itself at 400 MB in case nobody is left to kill it
    r = run_python("x = []\nwhile len(x) < 4000:\n    x.append('y' * 100000 + str(len(x)))\nprint('ALLOCATED')", memory_mb=100)
    assert r.status == "memory" and "ALLOCATED" not in r.stdout and r.seconds < 5.0 and r.peak_mb < 350


def test_output_is_capped():
    r = run_python("print('x' * 200000)\nimport sys\nprint('e' * 200000, file=sys.stderr)")
    assert r.status == "ok" and len(r.stdout) == OUTPUT_BYTES and len(r.stderr) == OUTPUT_BYTES
    start = time.monotonic()
    r = run_python("while True:\n    print('x' * 1000)")           # the kernel stops the file at FILE_MB
    assert r.status == "error" and len(r.stdout) == OUTPUT_BYTES and "File too large" in r.stderr
    assert time.monotonic() - start < 5.0


def test_kernel_limits_are_set():
    r = run_python("import resource\nprint(resource.getrlimit(resource.RLIMIT_FSIZE), resource.getrlimit(resource.RLIMIT_CPU))")
    assert r.stdout == "(%d, %d) (7, 7)\n" % (FILE_MB * 2 ** 20, FILE_MB * 2 ** 20)     # timeout 5 s + 2
    # stdout is the only place left to write to: the kernel stops it at exactly FILE_MB
    r = run_python("import os, sys\nn = 0\ntry:\n    while True:\n        n += os.write(1, b'x' * 65536)\n"
                   "except OSError as problem:\n    print(n, problem, file=sys.stderr)")
    assert r.stderr == "%d [Errno 27] File too large\n" % (FILE_MB * 2 ** 20) and len(r.stdout) == OUTPUT_BYTES


def test_child_that_ends_while_it_is_being_killed(monkeypatch):
    """The child can end by itself between the decision to kill it and the kill. Then only a zombie is left,
    and macOS answers the kill with EPERM. Reproduced here by letting the child die just before killpg."""
    killpg = os.killpg

    def killpg_after_the_child_died(pid, sig):
        os.kill(pid, sig)
        time.sleep(0.3)
        killpg(pid, sig)

    monkeypatch.setattr(sandbox.os, "killpg", killpg_after_the_child_died)
    r = run_python("while True:\n    pass", timeout_s=0.5)
    assert r.status == "timeout"


def test_run_many_keeps_order_and_runs_in_parallel():
    code = "import time\nstart = time.time()\ntime.sleep(0.3)\nprint(%d, start, time.time())"
    results = run_many([code % g for g in range(8)], workers=4)
    rows = [r.stdout.split() for r in results]
    assert [row[0] for row in rows] == [str(g) for g in range(8)]             # results in the order of the programs
    spans = sorted((float(start), float(end)) for _, start, end in rows)
    assert any(later[0] < earlier[1] for earlier, later in zip(spans, spans[1:]))      # two of them ran at the same time
