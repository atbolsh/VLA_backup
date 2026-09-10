from __future__ import annotations

import os
import secrets
from pathlib import Path

from roboenv.paths import repo_root

AUTH_MESSAGE = (
    "Use the username and token printed by launch.sh "
    "(same token as Jupyter / Open if those env vars exist)."
)


def token_file() -> Path:
    return repo_root() / ".gradio_token"


def resolve_credentials() -> tuple[str, str, str]:
    """Return (username, password, source). launch.sh already prefers this order."""
    user = os.environ.get("ROBOENV_USER") or os.environ.get("ROBOENV_AUTH_USER") or "vastai"
    for key, source in (
        ("ROBOENV_PASSWORD", "ROBOENV_PASSWORD"),
        ("ROBOENV_AUTH_PASSWORD", "ROBOENV_AUTH_PASSWORD"),
        ("OPEN_BUTTON_TOKEN", "OPEN_BUTTON_TOKEN"),
        ("JUPYTER_TOKEN", "JUPYTER_TOKEN"),
    ):
        val = (os.environ.get(key) or "").strip()
        if val:
            return user, val, source
    path = token_file()
    if path.is_file():
        stored = path.read_text(encoding="utf-8").strip()
        if stored:
            return user, stored, str(path)
    token = secrets.token_urlsafe(24)
    path.write_text(token + "\n", encoding="utf-8")
    path.chmod(0o600)
    return user, token, f"generated:{path}"


def launch_auth() -> tuple[tuple[str, str], str]:
    user, password, _source = resolve_credentials()
    return (user, password), AUTH_MESSAGE
