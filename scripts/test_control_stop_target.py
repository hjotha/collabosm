"""Targeted stop: an explicitly confirmed foreign session can be stopped.

Automatic paths (no target) must still refuse to touch VMs this frontend did
not start -- see scripts/test_restore_safety.py for the adopt/stop guards.
"""
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "frontend"))
import control


FOREIGN = {"name": "?", "endpoint": "gpu-a100-s-ass1c0-3ufkfj6l2sjz9",
           "hardware": "A100", "ours": False}


def make_control():
    root = tempfile.mkdtemp(prefix="tcb-stop-target-")
    ctl = control.Control(root, session="collabosm", fake=True, persist_ledger=False)
    ctl.state["billing"] = {"count": 1, "sessions": [dict(FOREIGN)],
                            "raw": "test", "checked_at": time.time(), "ok": True}
    return ctl


def wait_for(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class StopTarget(unittest.TestCase):
    def test_targeted_foreign_stop_runs_down_and_reports_stopped(self):
        ctl = make_control()
        run_down = MagicMock(return_value=True)
        other = {**FOREIGN, "endpoint": "gpu-other"}

        def probe():
            self.assertEqual(ctl.state["stage"], "stopping")
            ctl.state["billing"].update(count=1, sessions=[other])
            return [other]

        with patch.object(ctl, "_run_down_session", run_down), patch.object(ctl, "_probe_sessions", probe):
            res = ctl.stop("manual", target_session="?", target_endpoint=FOREIGN["endpoint"])
            self.assertTrue(wait_for(lambda: ctl.state["stage"] == "stopped"),
                            "stage is %r" % ctl.state.get("stage"))
        self.assertTrue(res.get("ok"))
        self.assertEqual(res.get("code"), "stopping")
        self.assertEqual(res.get("target"), FOREIGN["endpoint"])
        run_down.assert_called_once()
        args = run_down.call_args[0]
        self.assertTrue(args[0].startswith("tcb-stop-"))
        self.assertNotEqual(args[0], "collabosm")
        self.assertEqual(args[1], FOREIGN["endpoint"])
        self.assertEqual(ctl.state["billing"]["sessions"], [other])

    def test_unknown_target_is_rejected_without_side_effects(self):
        ctl = make_control()
        run_down = MagicMock(return_value=True)
        with patch.object(ctl, "_run_down_session", run_down):
            res = ctl.stop("manual", target_session="?", target_endpoint="gpu-nope-0")
        self.assertFalse(res.get("ok"))
        self.assertEqual(res.get("code"), "unknown_target")
        time.sleep(0.3)
        run_down.assert_not_called()
        self.assertEqual(ctl.state["stage"], "idle")

    def test_automatic_stop_still_refuses_foreign_vm(self):
        ctl = make_control()
        run_down = MagicMock(return_value=True)
        with patch.object(ctl, "_run_down_session", run_down):
            res = ctl.stop("manual")
        self.assertTrue(res.get("ok"))
        self.assertEqual(res.get("code"), "nothing_to_stop")
        time.sleep(0.3)
        run_down.assert_not_called()


if __name__ == "__main__":
    unittest.main()
