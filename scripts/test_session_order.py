"""Server sessions are stored ordered by machine capacity (ascending VRAM),
with the family VRAM shown next to the hardware name in parens.

The /colab panel renders billing sessions in stored order, so the weakest
box comes first: T4 < L4 < V6E1 (TPU) < A100 < G4. Unknown hardware sorts
last, ties break by session name.
"""
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "frontend"))
import control


def make_control():
    root = tempfile.mkdtemp(prefix="tcb-session-order-")
    ctl = control.Control(root, session="collabosm", fake=True, persist_ledger=False)
    ctl.fake = False
    return ctl


OUT = """[zeta] gpu-g4-0 | Hardware: G4 | Variant: GPU
[alpha] gpu-t4-0 | Hardware: T4 | Variant: GPU
[omega] gpu-a100-0 | Hardware: A100 | Variant: GPU
[mid] gpu-l4-0 | Hardware: L4 | Variant: GPU
[unk] gpu-xyz-0 | Hardware: XYZ | Variant: GPU
[tpu] gpu-tpu-0 | Hardware: TPU v6e-1 | Variant: TPU
"""


class SessionOrder(unittest.TestCase):
    def test_display_labels_carry_vram(self):
        self.assertEqual(control._display_hardware("T4"), "T4 (16 GB)")
        self.assertEqual(control._display_hardware("L4"), "L4 (24 GB)")
        self.assertEqual(control._display_hardware("TPU v6e-1"), "V6E1 (32 GB)")
        self.assertEqual(control._display_hardware("A100"), "A100 (40/80 GB)")
        self.assertEqual(control._display_hardware("G4"), "G4 (96 GB)")
        self.assertEqual(control._display_hardware("XYZ"), "XYZ")
        self.assertEqual(control._display_hardware("A100 (40/80 GB)"), "A100 (40/80 GB)")

    def test_key_orders_by_vram_then_name(self):
        ordered = sorted([
            {"name": "g", "hardware": "G4 (96 GB)"},
            {"name": "t", "hardware": "T4 (16 GB)"},
            {"name": "a", "hardware": "A100 (40/80 GB)"},
            {"name": "l", "hardware": "L4 (24 GB)"},
            {"name": "v", "hardware": "V6E1 (32 GB)"},
            {"name": "x", "hardware": "XYZ"},
        ], key=control._session_order_key)
        self.assertEqual([s["hardware"] for s in ordered],
                         ["T4 (16 GB)", "L4 (24 GB)", "V6E1 (32 GB)",
                          "A100 (40/80 GB)", "G4 (96 GB)", "XYZ"])

    def test_probe_stores_sessions_sorted_with_vram(self):
        ctl = make_control()
        with patch.object(ctl, "_wsl_run", return_value=OUT):
            sessions = ctl._probe_sessions()
        self.assertIsNotNone(sessions)
        stored = ctl.state["billing"]["sessions"]
        self.assertEqual([s["hardware"] for s in stored],
                         ["T4 (16 GB)", "L4 (24 GB)", "V6E1 (32 GB)",
                          "A100 (40/80 GB)", "G4 (96 GB)", "XYZ"])
        self.assertEqual(ctl.state["billing"]["count"], 6)


if __name__ == "__main__":
    unittest.main()
