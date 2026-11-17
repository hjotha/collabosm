"""Tests for CLIProxyAPI config synchronization."""
import os
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "frontend"))
import cliproxy_sync  # noqa: E402


SAMPLE_CONFIG = """\
port: 8317
openai-compatibility:
- name: qwen57
  disabled: false
  prefix: qwen57
  base-url: http://192.168.1.57:8090/v1
  models:
  - name: qwen-3.8-27b
    alias: qwen-3.8-27b
other-section:
  keep: true
"""


class CliproxySyncTests(unittest.TestCase):
    def test_inserts_and_updates_colab_provider_preserving_inode_and_other_sections(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "config.yaml")
            with open(cfg, "w", encoding="utf-8") as fh:
                fh.write(SAMPLE_CONFIG)
            inode_before = os.stat(cfg).st_ino

            changed = cliproxy_sync.sync_cliproxy("qwen3.8-flash-next-exl3", disabled=False, config_path=cfg)
            self.assertTrue(changed)
            self.assertEqual(os.stat(cfg).st_ino, inode_before)

            with open(cfg, encoding="utf-8") as fh:
                parsed = yaml.safe_load(fh)
            self.assertTrue(parsed["other-section"]["keep"])
            providers = {p["name"]: p for p in parsed["openai-compatibility"]}
            self.assertIn("qwen57", providers)
            colab = providers["colab"]
            self.assertFalse(colab["disabled"])
            self.assertEqual(colab["prefix"], "colab")
            self.assertEqual(colab["base-url"], cliproxy_sync.DEFAULT_BASE_URL)
            aliases = {m["alias"]: m["name"] for m in colab["models"]}
            self.assertEqual(aliases["qwen3.8-flash-next-exl3"], "qwen3.8-flash-next-exl3")
            self.assertEqual(aliases["qwen-3.8-flash-next"], "qwen3.8-flash-next-exl3")

            # Subsequent update to another model and then disabled=True
            changed2 = cliproxy_sync.sync_cliproxy("swift-qwen38-27b-iq3s", disabled=False, config_path=cfg)
            self.assertTrue(changed2)
            with open(cfg, encoding="utf-8") as fh:
                parsed2 = yaml.safe_load(fh)
            colab2 = next(p for p in parsed2["openai-compatibility"] if p["name"] == "colab")
            aliases2 = {m["alias"]: m["name"] for m in colab2["models"]}
            self.assertEqual(aliases2["swift-qwen-38-27b"], "swift-qwen38-27b-iq3s")

            changed3 = cliproxy_sync.sync_cliproxy(None, disabled=True, config_path=cfg)
            self.assertTrue(changed3)
            with open(cfg, encoding="utf-8") as fh:
                parsed3 = yaml.safe_load(fh)
            colab3 = next(p for p in parsed3["openai-compatibility"] if p["name"] == "colab")
            self.assertTrue(colab3["disabled"])

    def test_keeps_lowercase_alias_for_mixed_case_model_id(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "config.yaml")
            with open(cfg, "w", encoding="utf-8") as fh:
                fh.write(SAMPLE_CONFIG)

            self.assertTrue(cliproxy_sync.sync_cliproxy("Qwen3.8-27B", config_path=cfg))
            with open(cfg, encoding="utf-8") as fh:
                parsed = yaml.safe_load(fh)
            colab = next(p for p in parsed["openai-compatibility"] if p["name"] == "colab")
            aliases = {m["alias"]: m["name"] for m in colab["models"]}
            self.assertEqual(aliases["Qwen3.8-27B"], "Qwen3.8-27B")
            self.assertEqual(aliases["qwen-3.8-27B"], "Qwen3.8-27B")
            self.assertEqual(aliases["qwen-3.8-27b"], "Qwen3.8-27B")


if __name__ == "__main__":
    unittest.main()
