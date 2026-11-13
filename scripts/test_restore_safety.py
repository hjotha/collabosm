"""Offline regression tests: an unrelated Colab VM must not be adopted or stopped."""
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import restore


class RestoreSafety(unittest.TestCase):
    def assignment(self, endpoint):
        return types.SimpleNamespace(
            endpoint=endpoint, accelerator=types.SimpleNamespace(value="A100"),
            machine_shape=types.SimpleNamespace(name="STANDARD"),
            runtime_proxy_info=types.SimpleNamespace(token="test", url="test"))

    def run_restore(self, mine, live, args, vram=20):
        state = types.SimpleNamespace(
            client=types.SimpleNamespace(list_assignments=lambda: live),
            store=types.SimpleNamespace(get=lambda name: mine))
        fake_session = types.ModuleType("colab_cli.commands.session")
        fake_session.Accelerator = types.SimpleNamespace(A100="A100")
        modules = {"colab_cli": types.ModuleType("colab_cli"),
                   "colab_cli.commands": types.ModuleType("colab_cli.commands"),
                   "colab_cli.commands.session": fake_session}
        with patch.dict(sys.modules, modules), \
             patch.object(sys, "argv", ["restore.py"] + args), \
             patch.object(restore, "_import_colab_cli", return_value=(state, object)), \
             patch.object(restore, "colab_argv", return_value=["colab"]), \
             patch.object(restore, "claimed_endpoints", return_value=set()), \
             patch.object(restore, "keep_alive"), \
             patch.object(restore, "register") as register, \
             patch.object(restore, "create_high_ram", return_value=("new-vm", "STANDARD")) as create, \
             patch.object(restore, "probe", return_value={"vram_GiB": vram}), \
             patch.object(restore, "stop_session") as stop, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = restore.main()
        return rc, register, create, stop

    def test_external_vm_is_neither_adopted_nor_stopped(self):
        rc, register, create, stop = self.run_restore(None, [self.assignment("external")],
                                                       ["--no-verify"])
        self.assertEqual(rc, 12)
        register.assert_not_called()
        create.assert_not_called()
        stop.assert_not_called()

    def test_force_new_does_not_adopt_external_vm(self):
        rc, register, create, stop = self.run_restore(None, [self.assignment("external")],
                                                       ["--no-verify", "--force-new"])
        self.assertEqual(rc, 0)
        register.assert_not_called()
        create.assert_called_once()
        stop.assert_not_called()

    def test_recorded_vm_is_reused_and_not_stopped_when_too_small(self):
        vm = self.assignment("ours")
        rc, register, create, stop = self.run_restore(vm, [vm], [])
        self.assertEqual(rc, 6)
        register.assert_called_once()
        create.assert_not_called()
        stop.assert_not_called()

    def test_new_vm_is_stopped_when_too_small(self):
        rc, _, create, stop = self.run_restore(None, [], [])
        self.assertEqual(rc, 6)
        create.assert_called_once()
        stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
