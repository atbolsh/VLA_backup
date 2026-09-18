#!/usr/bin/env python3
"""Point LIBERO_CONFIG_PATH and ~/.libero/config.yaml at the cloned LIBERO package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from roboenv.env.libero_home import prepare_libero  # noqa: E402


def main() -> int:
    dest = prepare_libero()
    print(f"wrote {dest}")
    print(f"LIBERO_CONFIG_PATH={dest.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
