from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from roboenv.env.libero import LiberoEnv
from roboenv.env.transforms import quat_wxyz_to_yaw, yaw_to_quat_wxyz
from roboenv.paths import scenes_dir


@dataclass
class SceneObject:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    joint: str | None
    movable: bool


def _objects_dict(env: LiberoEnv) -> dict[str, Any]:
    problem = env.problem
    out: dict[str, Any] = {}
    for attr in ("objects_dict", "fixtures_dict"):
        mapping = getattr(problem, attr, None)
        if isinstance(mapping, dict):
            out.update(mapping)
    return out


def _free_joint(obj: Any) -> str | None:
    joints = list(getattr(obj, "joints", None) or [])
    if not joints:
        return None
    return str(joints[0])


class SceneEditor:
    """Move free-joint objects, save/load flattened MuJoCo init states."""

    def __init__(self, env: LiberoEnv):
        self.env = env

    def list_objects(self) -> list[SceneObject]:
        sim = self.env.sim
        items: list[SceneObject] = []
        for name, obj in _objects_dict(self.env).items():
            joint = _free_joint(obj)
            movable = False
            x = y = z = yaw = 0.0
            if joint:
                try:
                    qpos = np.asarray(sim.data.get_joint_qpos(joint), dtype=np.float64)
                    if qpos.size >= 7:
                        movable = True
                        x, y, z = (float(qpos[0]), float(qpos[1]), float(qpos[2]))
                        yaw = quat_wxyz_to_yaw(qpos[3:7])
                except Exception:
                    joint = None
            if not movable:
                body = getattr(obj, "root_body", None)
                if body:
                    try:
                        bid = sim.model.body_name2id(body)
                        pos = sim.data.body_xpos[bid]
                        quat = sim.data.body_xquat[bid]
                        x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
                        yaw = quat_wxyz_to_yaw(quat)
                    except Exception:
                        pass
            items.append(
                SceneObject(
                    name=name,
                    x=x,
                    y=y,
                    z=z,
                    yaw=yaw,
                    joint=joint,
                    movable=movable,
                )
            )
        items.sort(key=lambda o: o.name)
        return items

    def object_names(self, movable_only: bool = True) -> list[str]:
        return [o.name for o in self.list_objects() if (o.movable or not movable_only)]

    def get(self, name: str) -> SceneObject:
        for obj in self.list_objects():
            if obj.name == name:
                return obj
        raise KeyError(name)

    def apply(self, name: str, x: float, y: float, z: float, yaw: float) -> None:
        obj = self.get(name)
        if not obj.movable or not obj.joint:
            raise ValueError(f"{name} is not a free-joint object")
        sim = self.env.sim
        qpos = np.asarray(sim.data.get_joint_qpos(obj.joint), dtype=np.float64).copy()
        qpos[0], qpos[1], qpos[2] = float(x), float(y), float(z)
        qpos[3:7] = yaw_to_quat_wxyz(float(yaw))
        sim.data.set_joint_qpos(obj.joint, qpos)
        sim.forward()
        if hasattr(self.env._env, "_post_process"):
            self.env._env._post_process()
        if hasattr(self.env._env, "_update_observables"):
            self.env._env._update_observables(force=True)
        self.env._obs = self.env.problem._get_observations()

    def save(self, name: str, dest: Path | None = None) -> Path:
        dest = dest or (scenes_dir() / f"{name}.npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        objects = [o.__dict__ for o in self.list_objects()]
        meta = {
            "name": name,
            "suite": self.env.suite,
            "task_id": self.env.task_id,
            "language": self.env.task_language,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "objects": objects,
        }
        np.savez_compressed(
            dest,
            sim_state=self.env.get_sim_state(),
            meta=json.dumps(meta),
        )
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return dest

    def load(self, path: Path | str) -> dict:
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        state = np.asarray(data["sim_state"])
        meta: dict = {}
        if "meta" in data.files:
            raw = data["meta"]
            if isinstance(raw, np.ndarray):
                raw = raw.item()
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            if isinstance(raw, str) and raw:
                meta = json.loads(raw)
        sidecar = path.with_suffix(".json")
        if not meta and sidecar.is_file():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        self.env.set_sim_state(state)
        return meta

    @staticmethod
    def list_saves() -> list[Path]:
        return sorted(scenes_dir().glob("*.npz"))
