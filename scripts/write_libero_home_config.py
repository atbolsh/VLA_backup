#!/usr/bin/env python3
"""Point ~/.libero/config.yaml at the cloned LIBERO package."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "vendor" / "LIBERO" / "libero" / "libero"


def main() -> int:
    if not PKG.is_dir():
        raise SystemExit(f"LIBERO package not found at {PKG}")
    cfg = {
        "benchmark_root": str(PKG),
        "bddl_files": str(PKG / "bddl_files"),
        "init_states": str(PKG / "init_files"),
        "datasets": str(ROOT / "vendor" / "LIBERO" / "datasets"),
        "assets": str(PKG / "assets"),
    }
    dest = Path(os.path.expanduser("~")) / ".libero" / "config.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"wrote {dest}")
    for key, path in cfg.items():
        print(f"  {key}: {path}  exists={Path(path).exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
