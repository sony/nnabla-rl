from contextlib import contextmanager

_evaluating = False


@contextmanager
def eval_scope():
    global _evaluating
    try:
        _evaluating = True
        yield
    finally:
        _evaluating = False


def is_eval_scope():
    return _evaluating
