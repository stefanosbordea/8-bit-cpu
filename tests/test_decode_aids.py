from lsllm.decode_aids import name_like, repeated


def test_repeated_blocks_the_token_that_would_repeat_a_run_too_often():
    answer = [5, 6, 7, 5, 6, 7, 5, 6]
    assert repeated(answer, 3, 2) == {7}            # "5 6 7" exists twice: a third time is not allowed
    assert repeated(answer, 3, 3) == set()
    assert repeated([5, 6, 7, 5, 6], 3, 2) == set() and repeated([5, 6, 7, 5, 6], 3, 1) == {7}
    assert repeated([4, 4, 4], 1, 3) == {4} and repeated([], 3, 2) == set() and repeated([1], 3, 1) == set()


def test_name_like_tokens():
    assert name_like("count") and name_like(" n") and name_like("10") and name_like("_x1")
    assert not any(name_like(piece) for piece in ("<NL>", "<IND>", "return", " if", "(", ", ", '"""', "", " "))
