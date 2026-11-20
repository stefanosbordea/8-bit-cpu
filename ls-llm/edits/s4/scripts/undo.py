"""No model run. One Session.undo (the inline calls of the session, saved as a script). Never call it twice in a
row: the second call takes back the first (see restore.py)."""
from lsllm.edit.session import Session
s = Session("edits/s4")
print("undo:", s.undo()["changed"], "numbers restored")
