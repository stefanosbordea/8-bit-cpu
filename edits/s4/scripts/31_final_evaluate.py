"""Final run: evaluate the kept model (the state of run 23, restored with `restore.py 17`) once more, to prove
that the working copy in edits/s4 is that model."""
import sys
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

report(Session("edits/s4"), "final: kept model (state of run 23: 19 W2_1 columns and 8 Wo_1 columns zeroed, EXP[1] kappa 3)")
