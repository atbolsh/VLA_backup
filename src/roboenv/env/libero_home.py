"""Make LIBERO importable without its interactive ~/.libero prompt."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from roboenv.paths import repo_root

# robosuite 1.4.x + LIBERO eval images. 3.3.3 darkens floors; 3.4+ breaks
# libero_spatial init; 3.10+ changes mj_fullM and joint enums.
PINNED_MUJOCO = "3.3.2"
_MUJOCO_TOO_NEW = (3, 10, 0)


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


def _mujoco_version() -> tuple[int, ...]:
    import mujoco

    parts: list[int] = []
    for token in mujoco.__version__.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts[:3] or [0])


def _patch_robosuite_joint_addrs() -> None:
    """Newer mujoco enums no longer compare equal to numpy jnt_type ints."""
    import mujoco
    from robosuite.utils.binding_utils import MjModel

    if getattr(MjModel.get_joint_qpos_addr, "_roboenv_patched", False):
        return

    free = int(mujoco.mjtJoint.mjJNT_FREE)
    ball = int(mujoco.mjtJoint.mjJNT_BALL)
    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)

    def _addr(self, name: str, *, vel: bool):
        joint_id = self.joint_name2id(name)
        joint_type = int(self.jnt_type[joint_id])
        if vel:
            joint_addr = int(self.jnt_dofadr[joint_id])
            n_free, n_ball = 6, 3
        else:
            joint_addr = int(self.jnt_qposadr[joint_id])
            n_free, n_ball = 7, 4
        if joint_type == free:
            return (joint_addr, joint_addr + n_free)
        if joint_type == ball:
            return (joint_addr, joint_addr + n_ball)
        if joint_type not in (hinge, slide):
            raise AssertionError(
                f"joint {name!r} has type {joint_type}, expected hinge/slide/ball/free"
            )
        return joint_addr

    def get_joint_qpos_addr(self, name):
        return _addr(self, name, vel=False)

    def get_joint_qvel_addr(self, name):
        return _addr(self, name, vel=True)

    get_joint_qpos_addr._roboenv_patched = True  # type: ignore[attr-defined]
    MjModel.get_joint_qpos_addr = get_joint_qpos_addr
    MjModel.get_joint_qvel_addr = get_joint_qvel_addr


def ensure_mujoco_for_libero() -> None:
    import mujoco

    if _mujoco_version() >= _MUJOCO_TOO_NEW:
        raise RuntimeError(
            f"mujoco {mujoco.__version__} is too new for robosuite 1.4 / LIBERO. "
            "Do not rerun setup.sh. On the box:\n"
            f"  .venv/bin/python -m pip install 'mujoco=={PINNED_MUJOCO}'\n"
            "then bash launch.sh"
        )
    _patch_robosuite_joint_addrs()


def _patch_libero_init_state_load() -> None:
    """LIBERO pickle init files; torch 2.6+ defaults weights_only=True."""
    import torch
    from libero.libero import get_libero_path
    from libero.libero.benchmark import Benchmark

    if getattr(Benchmark.get_task_init_states, "_roboenv_patched", False):
        return

    def get_task_init_states(self, i):
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        return torch.load(init_states_path, map_location="cpu", weights_only=False)

    get_task_init_states._roboenv_patched = True  # type: ignore[attr-defined]
    Benchmark.get_task_init_states = get_task_init_states


def ensure_libero_ready() -> str:
    dest = prepare_libero()
    ensure_mujoco_for_libero()
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
    _patch_libero_init_state_load()
    return path
