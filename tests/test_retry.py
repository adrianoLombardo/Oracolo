import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.utils import retry_with_backoff, ERROR_COUNTS


class MyError(Exception):
    pass


def test_retry_with_backoff_counts_errors():
    ERROR_COUNTS.clear()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise MyError("boom")
        return 123

    res = retry_with_backoff(flaky, retries=5, base_delay=0.01)
    assert res == 123
    assert ERROR_COUNTS["MyError"] == 2
