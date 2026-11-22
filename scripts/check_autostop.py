#!/usr/bin/env python3
"""Test that automatic cleanup only acts on an explicitly created endpoint."""
import contextlib
import io
import os
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'frontend'), os.path.join(ROOT, 'scripts')]
import control
import provision


class Tests(unittest.TestCase):
    def run_failure(self, lines, rc):
        c = control.Control.__new__(control.Control)
        c.stop_wanted = False
        c.state = {'stage': 'requesting'}
        c.colab = types.SimpleNamespace(check=Mock())
        c._spawn = Mock(return_value=types.SimpleNamespace(stdout=lines, wait=lambda: rc))
        c._learn_endpoint = Mock()
        c._absorb = Mock()
        c._fail = Mock()
        c._run({})
        return c

    def test_same_exit_code_before_and_after_create(self):
        before = self.run_failure([], 2)
        self.assertFalse(before._fail.call_args.kwargs['vm_possible'])
        after = self.run_failure(['COLLABOSM_CREATED_ENDPOINT new-vm-123\n'], 2)
        self.assertTrue(after._fail.call_args.kwargs['vm_possible'])
        after._learn_endpoint.assert_called_once_with('new-vm-123')
        self.assertEqual(after._fail.call_args.kwargs['stop_endpoint'], 'new-vm-123')

    def test_failure_without_marker_never_stops_preexisting_vm(self):
        c = self.run_failure(["[restore] re-issued token for existing registration\n"], 1)
        self.assertFalse(c._fail.call_args.kwargs['vm_possible'])
        c._learn_endpoint.assert_not_called()

    def test_manual_stop_failure_does_not_close_ledger_or_claim_stopped(self):
        c = control.Control.__new__(control.Control)
        c.lock = threading.RLock()
        c.state = {'stage': 'stopping'}
        c.known_endpoint = None
        c._run_down = Mock(return_value=False)
        c._close_session = Mock()
        c._let_go = Mock()
        c._log = Mock()
        c._probe_sessions = Mock(return_value=[])
        c._balance_soon = Mock()
        c._do_stop('manual')
        self.assertEqual(c.state['stage'], 'failed')
        self.assertEqual(c.state['note']['k'], 'fail.stop')
        c._close_session.assert_not_called()
        c._let_go.assert_not_called()

    def test_manual_stop_success_closes_ledger(self):
        c = control.Control.__new__(control.Control)
        c.lock = threading.RLock()
        c.state = {'stage': 'stopping'}
        c.known_endpoint = None
        c._run_down = Mock(return_value=True)
        c._close_session = Mock(return_value=0.1)
        c._let_go = Mock()
        c._refresh_money = Mock()

        def probe():
            self.assertEqual(c.state['stage'], 'stopping')
            return []

        c._probe_sessions = Mock(side_effect=probe)
        c._balance_soon = Mock()
        c._do_stop('manual')
        self.assertEqual(c.state['stage'], 'stopped')
        c._close_session.assert_called_once_with('manual')

    def test_manual_stop_requires_a_valid_listing_without_our_endpoint(self):
        for sessions in (None, [{'endpoint': 't4-vm', 'ours': False}]):
            with self.subTest(sessions=sessions):
                c = control.Control.__new__(control.Control)
                c.lock = threading.RLock()
                c.state = {'stage': 'stopping'}
                c.known_endpoint = 't4-vm'
                c._run_down = Mock(return_value=True)
                c._close_session = Mock(return_value=0.0)
                c._let_go = Mock()
                c._log = Mock()
                c._probe_sessions = Mock(return_value=sessions)
                c._balance_soon = Mock()
                c._do_stop('manual')
                self.assertEqual(c.state['stage'], 'failed')
                c._close_session.assert_not_called()

    def test_cancelling_provisioning_without_an_endpoint_cannot_confirm_a_stop(self):
        c = control.Control.__new__(control.Control)
        c.lock = threading.RLock()
        c.state = {'stage': 'stopping'}
        c.known_endpoint = None
        c._probe_sessions = Mock(return_value=[])
        c._run_down = Mock()
        c._close_session = Mock()
        c._log = Mock()
        c._balance_soon = Mock()
        c._do_stop('manual', starting=True)
        self.assertEqual(c.state['note']['k'], 'fail.stop')
        c._run_down.assert_not_called()
        c._close_session.assert_not_called()

    def test_down_refuses_wrong_endpoint(self):
        args = types.SimpleNamespace(session='collabosm', endpoint='new-vm')
        with patch.object(provision, 'cli_python', return_value='python'), \
             patch.object(provision, 'run', return_value=(0, 'KEEPALIVE ok other-vm\n')), \
             patch.object(provision, 'colab') as colab, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(provision.cmd_down(args), 1)
            colab.assert_not_called()

    def test_down_reports_failure_when_endpoint_remains_listed(self):
        args = types.SimpleNamespace(session='collabosm', endpoint='new-vm')
        with patch.object(provision, 'cli_python', return_value='python'), \
             patch.object(provision, 'run', return_value=(0, 'KEEPALIVE ok new-vm\n')), \
             patch.object(provision, 'colab', side_effect=[(0, 'stopped'),
                 (0, '[collabosm] new-vm | Hardware: T4 | Variant: GPU')]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(provision.cmd_down(args), 1)

    def test_down_targets_matching_endpoint(self):
        args = types.SimpleNamespace(session='collabosm', endpoint='new-vm')
        with patch.object(provision, 'cli_python', return_value='python'), \
             patch.object(provision, 'run', return_value=(0, 'KEEPALIVE ok new-vm reattached\n')), \
             patch.object(provision, 'colab', side_effect=[(0, 'stopped'), (0, 'No active sessions found')]) as colab, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(provision.cmd_down(args), 0)
            self.assertEqual(colab.call_args_list[0].args[0], ['stop', '-s', 'collabosm'])
            self.assertIn('COLLABOSM_STOP_OK', output.getvalue())

    def test_down_cannot_confirm_stop_of_an_unregistered_vm(self):
        args = types.SimpleNamespace(session='collabosm', endpoint=None)
        listed = '[?] t4-vm | Hardware: T4 | Variant: GPU'

        def colab_command(argv, timeout):
            if argv == ['sessions']:
                return 0, listed
            self.assertEqual(argv, ['stop', '-s', 'collabosm'])
            return 0, "[colab] Session 'collabosm' not found."

        with patch.object(provision, 'colab', side_effect=colab_command), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(provision.cmd_down(args), 1)
            self.assertNotIn('COLLABOSM_STOP_OK', output.getvalue())

    def test_name_only_stop_detects_target_even_if_the_name_disappears(self):
        args = types.SimpleNamespace(session='collabosm', endpoint=None)
        with patch.object(provision, 'run', return_value=(0, 'KEEPALIVE ok new-vm')), \
             patch.object(provision, 'colab', side_effect=[
                 (0, '[collabosm] new-vm | Hardware: T4'),
                 (0, '[colab] Session terminated.'),
                 (0, '  [?] new-vm | Hardware: T4')]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(provision.cmd_down(args), 1)


if __name__ == '__main__':
    unittest.main()
