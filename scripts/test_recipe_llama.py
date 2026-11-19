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
        # Distinct values catch a preset or CLI argument accidentally left hardcoded.
        recipe.find(reg, "g4/swift-qwen38-iq3s")["env"].update(
            LLAMA_BATCH=1024, LLAMA_UBATCH=128, LLAMA_MTP_DRAFT=3, LLAMA_MTP_DRAFT_L=1,
            LLAMA_MTP_P_MIN=0.75, LLAMA_KV_TYPE="q8_0",
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


if __name__ == "__main__":
    unittest.main()
