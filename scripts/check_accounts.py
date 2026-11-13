#!/usr/bin/env python3
"""Offline account isolation and switching checks; never talks to Colab."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'frontend'))
from accounts import Accounts
from control import Control, RECIPES, account_operation


class Setup:
    where = 'native'
    account_home = None
    def __init__(self):
        self._stage = 'connected'
        self.settled_event = threading.Event()
    def stage(self):
        return self._stage
    def settled(self, timeout):
        return {"stage": self._stage}
    def _set(self, **values):
        self._stage = values['stage']


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.accounts = Accounts(self.temp.name, os.path.join(self.temp.name, 'old'))

    def control(self):
        c = Control.__new__(Control)
        c.lock = threading.RLock()
        c.account_operations = 0
        c.accounts = self.accounts
        c.colab = Setup()
        c.state = {'stage': 'idle', 'balance': {'cu': 5}, 'confirm': {},
                   'billing': {'sessions': []}}
        c.ledger = {'open': None}
        c.coupling = c.downing = c.keepalive_busy = False
        c.known_endpoint = 'old'
        c.last_session_count = 1
        c.absent_since = None
        c.epoch = 0
        c.stale = False
        return c

    def test_persist_and_isolate(self):
        profile = self.accounts.create('Segunda')
        self.assertNotEqual(self.accounts.home('default'), self.accounts.home(profile))
        self.assertFalse(self.accounts.switch(''))
        self.assertTrue(self.accounts.switch(profile))
        restored = Accounts(self.temp.name, '/ignored')
        self.assertEqual(restored.snapshot()['active_account'], profile)
        self.assertEqual(os.stat(restored.path).st_mode & 0o777, 0o600)
        script = os.path.join(self.temp.name, 'inspect.py')
        with open(script, 'w', encoding='utf-8') as f:
            f.write('import json,os; print(json.dumps([os.path.expanduser("~"),os.environ["HOME"]]))')
        out = subprocess.check_output([sys.executable, os.path.join(ROOT, 'scripts', 'account_exec.py'),
                                       restored.home(), script], text=True)
        self.assertEqual(json.loads(out), [restored.home(), restored.home()])

    def test_switch_refuses_active_operation_and_resets_cache(self):
        c = self.control()
        profile = self.accounts.create('Segunda')
        entered, release = threading.Event(), threading.Event()
        @account_operation
        def running(self):
            entered.set()
            release.wait(5)
        worker = threading.Thread(target=running, args=(c,))
        worker.start()
        self.assertTrue(entered.wait(5))
        self.assertEqual(c.account_switch(profile)['code'], 'account_busy')
        release.set()
        worker.join(5)
        with patch('control.threading.Thread'):
            self.assertEqual(c.account_switch(profile)['code'], 'account_switched')
        self.assertEqual(c.colab.account_home, self.accounts.home(profile))
        self.assertIsNone(c.state['balance'])
        self.assertEqual(c.state['billing']['sessions'], [])
        self.assertIsNone(c.known_endpoint)
        self.assertEqual(c.colab.stage(), 'checking')
        self.assertEqual(c.account_switch('default')['code'], 'account_busy')

    def test_old_confirmation_cannot_start_after_switch(self):
        c = self.control()
        recipe = next(r for r in RECIPES if r['runnable'])
        c.state['confirm'] = {'recipe': recipe['id']}
        profile = self.accounts.create('Segunda')
        with patch('control.threading.Thread'):
            self.assertEqual(c.account_switch(profile)['code'], 'account_switched')
        c.colab._stage = 'connected'
        self.assertEqual(c.select(recipe['id'], confirm=True)['code'], 'confirmation_expired')

    def test_free_t4_requires_verified_zero_balance_and_profile(self):
        c = self.control()
        c.budget_cu, c.idle_stop_min, c.max_session_h = 200, 20, 6
        rid = 't4-free/swift-qwen38-iq3s'
        c.state['balance'] = {'cu': 0}
        self.assertEqual(c.select(rid)['code'], 'free_account_required')
        with self.accounts.lock:
            self.accounts.data['accounts'].append({'id': 'free', 'label': 'Gratuita', 'home': self.temp.name})
        self.accounts.switch('free')
        c.state['balance'] = None
        c.state['cu_left'] = 0
        self.assertEqual(c.select(rid)['code'], 'free_account_required')
        c.state['balance'] = {'cu': 1}
        self.assertEqual(c.select(rid)['code'], 'free_account_required')
        c.state['balance'] = {'cu': 0}
        result = c.select(rid)
        self.assertEqual(result['code'], 'confirm_required')
        self.assertTrue(result['warning']['free_tier'])
        self.assertEqual(c.state['stage'], 'idle')

    def test_refuse_owned_session_and_auth(self):
        c = self.control()
        profile = self.accounts.create('Segunda')
        c.ledger['open'] = {'endpoint': 'owned'}
        self.assertEqual(c.account_switch(profile)['code'], 'account_busy')
        c.ledger['open'] = None
        c.colab._stage = 'authorizing'
        self.assertEqual(c.account_switch(profile)['code'], 'account_busy')
        c.colab._stage = 'connected'
        c.state['billing']['sessions'] = [{'ours': True}]
        self.assertEqual(c.account_switch(profile)['code'], 'account_busy')


if __name__ == '__main__':
    unittest.main()
