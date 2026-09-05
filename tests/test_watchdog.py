"""The stall watchdog's startup behaviour.

The watchdog used to exit on its FIRST empty poll. run_workers.sh starts it
immediately after launching supervisors, each of which runs `conda run` --
which takes seconds to exec the real python child. So the first poll saw
only the (correctly excluded) wrapper processes, found no workers, and
exited in the same second it started. Every campaign then ran with NO hang
protection, and nothing said so: the log line read like a normal shutdown.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

WATCHDOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "watchdog.py"
)


def _run(campaign_dir, *args):
    return subprocess.Popen(
        [sys.executable, WATCHDOG, str(campaign_dir), *args],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _log(campaign_dir):
    path = os.path.join(campaign_dir, "logs", "watchdog.log")
    return open(path).read() if os.path.exists(path) else ""


@pytest.fixture
def campaign(tmp_path):
    # a tag that cannot match any real process on the machine
    d = tmp_path / "campaign_xyzzy_no_such_tag"
    (d / "logs").mkdir(parents=True)
    return d


def test_does_not_exit_on_the_first_empty_poll(campaign):
    """The regression: with no workers YET, it must keep waiting."""
    proc = _run(campaign, "--startup-grace", "30", "--poll", "1")
    try:
        time.sleep(4)
        assert proc.poll() is None, (
            "watchdog exited during the startup grace period -- the campaign "
            "would run unprotected"
        )
        assert "exiting" not in _log(campaign)
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_exits_after_the_startup_grace_expires(campaign):
    """It must still give up eventually, rather than linger forever."""
    proc = _run(campaign, "--startup-grace", "2", "--poll", "1")
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("watchdog never exited after its startup grace expired")
    assert "no workers appeared" in _log(campaign)


def test_startup_grace_default_tolerates_a_slow_conda_start(campaign):
    """15 min, not seconds: `conda run` plus env solving can be slow, and the
    cost of being wrong is an entire unprotected campaign."""
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location("watchdog", WATCHDOG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ap = argparse.ArgumentParser()
    # reproduce the parser defaults without invoking main()
    src = open(WATCHDOG).read()
    assert '"--startup-grace", type=float, default=900.0' in src
    assert '"--exit-after-empty", type=int, default=3' in src
