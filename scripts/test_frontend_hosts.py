"""Tests for the frontend Host/Origin allow-list used behind a reverse proxy."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "frontend"))
import collabosm  # noqa: E402


class FrontendHostsTests(unittest.TestCase):
    def setUp(self):
        collabosm.ARGS = types.SimpleNamespace(port=3020, host="10.21.0.1")
        collabosm.ALLOWED_HOSTS = {"umbrel.lan:3022"}

    def test_bind_address_and_loopback(self):
        self.assertTrue(collabosm._names_this_server("10.21.0.1:3020"))
        self.assertTrue(collabosm._names_this_server("127.0.0.1:3020"))
        self.assertTrue(collabosm._names_this_server("localhost:3020"))
        self.assertFalse(collabosm._names_this_server("10.21.0.1:3023"))

    def test_proxy_allow_list(self):
        self.assertTrue(collabosm._names_this_server("umbrel.lan:3022"))
        self.assertTrue(collabosm._names_this_server("UMBREL.LAN:3022"))
        self.assertFalse(collabosm._names_this_server("evil.example:3022"))
        self.assertFalse(collabosm._names_this_server("umbrel.lan:3023"))


if __name__ == "__main__":
    unittest.main()
