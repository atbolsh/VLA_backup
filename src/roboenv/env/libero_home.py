"""Make LIBERO importable without its interactive ~/.libero prompt."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from roboenv.paths import repo_root


def vendor_libero_root() -> Path:
    return repo_root() / "vendor" / "LIBERO"


def libero_pkg_dir() -> Path:
    return vendor_libero_root() / "libero" / "libero"


def repo_libero_config_dir() -> Path:
    return repo_root() / ".libero"


def libero_config_payload() -> dict[str, str]:
    pkg = libero_pkg_dir()
    datasets = vendor_libero_root() / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    return {
        "benchmark_root": str(pkg),
        "bddl_files": str(pkg / "bddl_files"),
        "init_states": str(pkg / "init_files"),
        "datasets": str(datasets),
        "assets": str(pkg / "assets"),
    }


def _dump_yaml(cfg: dict[str, str]) -> str:
    return "".join(f"{key}: {value}\n" for key, value in cfg.items())


def write_libero_config(dest_dir: Path | None = None) -> Path:
    pkg = libero_pkg_dir()
    if not pkg.is_dir():
        raise FileNotFoundError(
            f"LIBERO sources missing at {pkg}. "
            "On the box, setup.sh clones vendor/LIBERO."
        )
    dest_dir = Path(dest_dir or os.environ.get("LIBERO_CONFIG_PATH") or repo_libero_config_dir())
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "config.yaml"
    dest.write_text(_dump_yaml(libero_config_payload()), encoding="utf-8")
    home_dir = Path.home() / ".libero"
    if dest_dir.resolve() != home_dir.resolve():
        home_dir.mkdir(parents=True, exist_ok=True)
        (home_dir / "config.yaml").write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def prepare_libero() -> Path:
    """Write config and put vendor/LIBERO on sys.path before importing libero."""
    dest = write_libero_config(repo_libero_config_dir())
    os.environ["LIBERO_CONFIG_PATH"] = str(dest.parent)
    vendor = vendor_libero_root()
    vendor_str = str(vendor)
    if vendor.is_dir() and vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    return dest


def ensure_libero_ready() -> str:
    dest = prepare_libero()
    try:
        from libero.libero import get_libero_path
    except EOFError as exc:
        raise RuntimeError(
            "LIBERO blocked on its first-import dataset prompt and stdin was closed. "
            f"Config was written to {dest}. {exc}"
        ) from exc
    except Exception as exc:
        vendor = vendor_libero_root()
        raise RuntimeError(
            f"Cannot import libero with {sys.executable}. "
            f"vendor={vendor} exists={vendor.is_dir()}. "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    path = get_libero_path("bddl_files")
    if not Path(path).exists():
        raise RuntimeError(f"LIBERO bddl_files path does not exist: {path}")
    return path
