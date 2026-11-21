"""Verify the recipe preview against the generated INI and llama-server arguments offline."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import provision
import recipe

BASH = shutil.which("bash")


@unittest.skipUnless(BASH and Path(BASH).parent.name.lower() != "windowsapps", "needs Bash")
class LlamaRecipeTests(unittest.TestCase):
    def test_preview_matches_ini_and_launch_flags(self):
        reg = recipe.load()
        t4 = next(r for r in recipe.flat(reg) if r["id"] == "t4-free/swift-qwen38-iq3s")
        self.assertTrue(t4["high_mem"])
        self.assertEqual(t4["high_mem_ram_gb"], 50)
        # Distinct values catch a preset or CLI argument accidentally left hardcoded.
        recipe.find(reg, "g4/swift-qwen38-iq3s")["env"].update(
            LLAMA_BATCH=1024, LLAMA_UBATCH=128, LLAMA_MTP_DRAFT=3, LLAMA_MTP_DRAFT_L=1,
            LLAMA_MTP_P_MIN=0.75, LLAMA_KV_TYPE="q8_0", LLAMA_SPEC_KV_TYPE="f16",
        )
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for name in provision.KIT:
                if name.startswith("llama"):
                    src = (HERE / name).read_text(encoding="utf-8")
                    (root / name).write_text(src.replace("/content/collabosm-llama", "vm"),
                                             encoding="utf-8", newline="\n")
            (root / "vm/llama").mkdir(parents=True)
            (root / "vm/llama/llama-server").touch()
            for row in recipe.flat(reg):
                env = row["env"]
                if env["ENGINE"] != "llama":
                    self.assertNotIn("LLAMA_BATCH", env)
                    continue
                run_env = dict(os.environ, **env)
                run_env.update(PORT="8090", KEY_FILE="key", LOG="log", LLAMA_STREAM_VRAM="1")
                with self.subTest(recipe=row["id"]):
                    subprocess.run([BASH, "-euc", '''
source ./llama.sh
nohup() { printf '%s\\n' "$@" > flags; }
llama_start
wait "$!"
'''], cwd=root, env=run_env,
                                   check=True, capture_output=True, text=True, timeout=5)
                    ini = (root / "vm/t4-production.ini").read_text().splitlines()
                    flags = (root / "flags").read_text().splitlines()
                    for tier, ctx, draft in (("s", "TIER_S_CTX", "LLAMA_MTP_DRAFT"),
                                              ("m", "TIER_M_CTX", "LLAMA_MTP_DRAFT"),
                                              ("l", "LLAMA_CONTEXT", "LLAMA_MTP_DRAFT_L")):
                        expected = {"ctx-size": ctx, "batch-size": "LLAMA_BATCH", "ubatch-size": "LLAMA_UBATCH",
                                    "spec-draft-n-max": draft, "cache-type-k": "LLAMA_KV_TYPE",
                                    "cache-type-v": "LLAMA_KV_TYPE"}
                        if tier != "l":
                            expected["spec-draft-p-min"] = "LLAMA_MTP_P_MIN"
                        for option, key in expected.items():
                            name = f"{option}-{tier}"
                            self.assertIn(f"{name} = {env[key]}", ini)
                            self.assertEqual(flags[flags.index("--" + name) + 1], env[key])
                    if row["id"] == t4["id"]:
                        self.assertIn("--hf-stream-vram", flags)
                        for option, value in (
                            ("--hf-download-threads", "16"),
                            ("--hf-stream-chunk-mb", "8"),
                            ("--hf-stream-disk-mode", "background"),
                        ):
                            self.assertEqual(flags[flags.index(option) + 1], value)
                    spec_kv = env.get("LLAMA_SPEC_KV_TYPE") or env["LLAMA_KV_TYPE"]
                    for suffix in ("k", "v"):
                        name = f"spec-draft-type-{suffix}"
                        self.assertIn(f"{name} = {spec_kv}", ini)
                        self.assertEqual(flags[flags.index("--" + name) + 1], spec_kv)
                        for tier in ("s", "m", "l"):
                            name = f"spec-draft-type-{suffix}-{tier}"
                            self.assertIn(f"{name} = {spec_kv}", ini)
                            self.assertEqual(flags[flags.index("--" + name) + 1], spec_kv)
                    for top_key in ("batch-size", "ubatch-size"):
                        val = str(env["LLAMA_BATCH" if top_key == "batch-size" else "LLAMA_UBATCH"])
                        self.assertIn(f"{top_key} = {val}", ini)
                        self.assertEqual(flags[flags.index("--" + top_key) + 1], val)
                    if str(env.get("LLAMA_EXL3_DIR", "0")) == "1":
                        self.assertIn(f"model = {env['MODEL_DIR']}", ini)
                        self.assertEqual(flags[flags.index("--model") + 1], env["MODEL_DIR"])
                        if str(env.get("LLAMA_STREAM_VRAM", "0")) == "1":
                            self.assertIn("--hf-stream-vram", flags)
                        else:
                            self.assertNotIn("--hf-stream-vram", flags)
                    else:
                        expected_model = f"{env['MODEL_DIR']}/{env['LLAMA_FILE']}"
                        self.assertIn(f"model = {expected_model}", ini)
                        self.assertEqual(flags[flags.index("--model") + 1], expected_model)

    def test_exl3_dir_non_adaptive_flags(self):
        defaults = dict(
            line.rstrip().split("=", 1)
            for line in (HERE / "llama-defaults.env").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for name in provision.KIT:
                if name.startswith("llama"):
                    src = (HERE / name).read_text(encoding="utf-8")
                    (root / name).write_text(src.replace("/content/collabosm-llama", "vm"),
                                             encoding="utf-8", newline="\n")
            (root / "vm/llama").mkdir(parents=True)
            (root / "vm/llama/llama-server").touch()
            (root / "vm/exl3").mkdir(parents=True)
            (root / "vm/exl3/config.json").write_text("{}", encoding="utf-8")
            run_env = dict(os.environ, **defaults)
            run_env.update(
                MODEL_REPO="turboderp/Qwen3.8-27B-exl3",
                MODEL_REVISION="516bf129059031c6da9416768ea6b7a1be00a8fc",
                MODEL_DIR="vm/exl3",
                MODEL_ID="qwen3.8-27b-exl3",
                LLAMA_EXL3_DIR="1",
                LLAMA_STREAM_VRAM="0",
                LLAMA_FILE="",
                LLAMA_CONTEXT="262144",
                TIER_S_CTX="0",
                TIER_M_CTX="0",
                LLAMA_BATCH="4096",
                LLAMA_UBATCH="2048",
                LLAMA_KV_TYPE="q4_0",
                LLAMA_SPEC_KV_TYPE="q8_0",
                LLAMA_MTP_DRAFT="4",
                LLAMA_MTP_DRAFT_L="4",
                LLAMA_MTP_P_MIN="0.65",
                PORT="8090",
                KEY_FILE="key",
                LOG="log",
            )
            subprocess.run([BASH, "-euc", """
source ./llama.sh
nohup() { printf '%s\n' "$@" > flags; }
llama_start
wait "$!"
"""], cwd=root, env=run_env, check=True, capture_output=True, text=True, timeout=5)
            ini = (root / "vm/t4-production.ini").read_text().splitlines()
            flags = (root / "flags").read_text().splitlines()
            self.assertIn("model = vm/exl3", ini)
            self.assertEqual(flags[flags.index("--model") + 1], "vm/exl3")
            self.assertNotIn("--hf-stream-vram", flags)
            self.assertEqual(flags[flags.index("--batch-size") + 1], "4096")
            self.assertEqual(flags[flags.index("--ubatch-size") + 1], "2048")
            self.assertEqual(flags[flags.index("--ctx-size-l") + 1], "262144")
            self.assertEqual(flags[flags.index("--spec-draft-n-max-m") + 1], "4")


if __name__ == "__main__":
    unittest.main()
