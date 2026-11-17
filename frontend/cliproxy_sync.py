"""Sync the active collabosm model into CLIProxyAPI's openai-compatibility config.

Writes in-place so single-file Docker bind mounts keep the same inode and
CLIProxyAPI hot-reloads the updated provider automatically.
"""
from __future__ import annotations

import os
import re
import threading

try:
    import yaml
except Exception:  # pragma: no cover - optional in minimal test envs
    yaml = None

DEFAULT_CONFIG_PATH = "/mnt/D/data/app-data/cliproxyapi/config/config.yaml"
DEFAULT_BASE_URL = "http://10.21.0.1:3023/v1"
_LOCK = threading.Lock()


def friendly_alias(model_id: str) -> str:
    """Normalize served model IDs like 'qwen3.8-flash-next-exl3' -> 'qwen-3.8-flash-next'."""
    s = (model_id or "").strip()
    if not s:
        return s
    s = re.sub(r"-(exl3|iq3_s|iq3s|gguf)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bqwen(\d)", r"qwen-\1", s, flags=re.IGNORECASE)
    return s


def _render_colab_block(model_id: str | None, disabled: bool, base_url: str) -> str:
    mid = (model_id or "").strip() or "qwen3.8-27b-exl3"
    alias = friendly_alias(mid)
    models = [(mid, mid)]
    if alias and alias != mid:
        models.append((mid, alias))
    # Keep a lowercase spelling too: clients configured with the older
    # qwen-3.8-27b alias must survive a switch to a Qwen3.8-27B VM.
    if alias and alias.lower() != alias:
        models.append((mid, alias.lower()))
    lines = [
        "- name: colab",
        f"  disabled: {'true' if disabled else 'false'}",
        "  prefix: colab",
        f"  base-url: {base_url}",
        "  headers:",
        "    User-Agent: CLIProxyAPI CollabOSM dynamic-model",
        "  api-key-entries:",
        "  - api-key: sk-local",
        "  models:",
    ]
    for name, al in models:
        lines.append(f"  - name: {name}")
        lines.append(f"    alias: {al}")
    return "\n".join(lines) + "\n"


def _upsert_colab_provider(text: str, block: str) -> str:
    section_re = re.compile(r"(?m)^openai-compatibility:\s*\n")
    m = section_re.search(text)
    if not m:
        sep = "" if text.endswith("\n") or not text else "\n"
        return f"{text}{sep}openai-compatibility:\n{block}"

    sec_start = m.end()
    next_top = re.search(r"(?m)^(?=[A-Za-z0-9_-]+:)", text[sec_start:])
    sec_end = sec_start + next_top.start() if next_top else len(text)
    section = text[sec_start:sec_end]

    item_re = re.compile(r"(?ms)^- name:\s*['\"]?colab['\"]?\s*\n.*?(?=^- name:|\Z)")
    if item_re.search(section):
        new_section = item_re.sub(block, section, count=1)
    else:
        if section and not section.endswith("\n"):
            section += "\n"
        new_section = block + section

    return text[:sec_start] + new_section + text[sec_end:]


def sync_cliproxy(
    model_id: str | None,
    disabled: bool = False,
    config_path: str | None = None,
    base_url: str | None = None,
) -> bool:
    """Update the 'colab' provider in CLIProxyAPI config.yaml. Returns True if written."""
    path = os.environ.get("CLIPROXYAPI_CONFIG", DEFAULT_CONFIG_PATH) if config_path is None else config_path
    if not path or not os.path.exists(path):
        return False
    url = os.environ.get("CLIPROXYAPI_COLAB_BASE_URL", DEFAULT_BASE_URL) if base_url is None else base_url
    block = _render_colab_block(model_id, disabled=disabled, base_url=url)

    with _LOCK:
        with open(path, "r", encoding="utf-8") as fh:
            old_text = fh.read()
        new_text = _upsert_colab_provider(old_text, block)
        if new_text == old_text:
            return False
        if yaml is not None:
            yaml.safe_load(new_text)
        # Write in-place to preserve inode for single-file Docker bind mounts.
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new_text)
    return True
