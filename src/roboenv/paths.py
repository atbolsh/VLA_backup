from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "setup.sh").exists():
            return parent
        if (parent / "pyproject.toml").exists() and (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()


def configs_dir() -> Path:
    return repo_root() / "configs"


def weights_dir() -> Path:
    return repo_root() / "weights"


def default_ckpt() -> Path:
    return weights_dir() / "EO-1-3B"


def robot_config_path() -> Path:
    return configs_dir() / "libero_robot_config.json"


def tasks_yaml_path() -> Path:
    return configs_dir() / "tasks.yaml"


def scenes_dir() -> Path:
    d = repo_root() / "scenes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vendor_eo1_libero() -> Path:
    return repo_root() / "vendor" / "eo1_libero"
