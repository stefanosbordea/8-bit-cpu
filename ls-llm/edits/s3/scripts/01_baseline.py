"""Run 1: the baseline score of the untouched model."""
from common import open_session, show

s, tok = open_session()
show(s.evaluate("baseline"))
