from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from roboenv.catalog import load_catalog
from roboenv.env.libero_home import ensure_libero_ready
from roboenv.env.transforms import dummy_action, proprio_state, rotate180


@dataclass
class LiberoObs:
    agent_view: np.ndarray
    wrist_view: np.ndarray
    state: np.ndarray
    raw: dict[str, Any]


def _ensure_libero_ready() -> None:
    ensure_libero_ready()


def _load_task_init_states(suite_obj, task_id: int) -> np.ndarray:
    import torch
    from libero.libero import get_libero_path

    task = suite_obj.get_task(task_id)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    if path.is_file():
        return np.asarray(torch.load(path, map_location="cpu", weights_only=False))
    return np.asarray(suite_obj.get_task_init_states(task_id))


class LiberoEnv:
    """Thin wrapper around official OffScreenRenderEnv (eval_libero.py)."""

    def __init__(
        self,
        suite: str = "libero_object",
        task_id: int = 0,
        resolution: int | None = None,
        seed: int | None = None,
    ):
        _ensure_libero_ready()
        cat = load_catalog()
        self.catalog = cat
        self.resolution = int(resolution or cat.resolution)
        self.seed = int(cat.seed if seed is None else seed)
        self._env: Any = None
        self._initial_states: np.ndarray | None = None
        self.suite = suite
        self.task_id = task_id
        self.task_language = ""
        self.task_name = ""
        self.steps = 0
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self._obs: dict[str, Any] | None = None
        self.open(suite, task_id)

    @property
    def repo_id(self) -> str:
        return self.catalog.suite(self.suite).repo_id

    @property
    def max_steps(self) -> int:
        return self.catalog.suite(self.suite).max_steps

    @property
    def problem(self) -> Any:
        if self._env is None:
            raise RuntimeError("env is closed")
        return self._env.env

    @property
    def sim(self) -> Any:
        return self._env.sim

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def open(self, suite: str, task_id: int) -> LiberoObs:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.close()
        suite_obj = benchmark.get_benchmark_dict()[suite]()
        if task_id < 0 or task_id >= suite_obj.n_tasks:
            raise IndexError(f"task_id {task_id} out of range for {suite} (n={suite_obj.n_tasks})")
        task = suite_obj.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=self.resolution,
            camera_widths=self.resolution,
        )
        env.seed(self.seed)
        self._env = env
        self._initial_states = _load_task_init_states(suite_obj, task_id)
        self.suite = suite
        self.task_id = task_id
        self.task_language = str(task.language)
        self.task_name = str(getattr(task, "name", "") or f"task_{task_id:02d}")
        return self.reset(init_index=0)

    def reset(self, init_index: int = 0, state: np.ndarray | None = None) -> LiberoObs:
        if self._env is None:
            raise RuntimeError("env is closed")
        self._env.reset()
        if state is not None:
            raw = self._env.set_init_state(np.asarray(state))
        else:
            if self._initial_states is None or len(self._initial_states) == 0:
                raw = self._env.reset()
            else:
                idx = int(init_index) % len(self._initial_states)
                raw = self._env.set_init_state(self._initial_states[idx])
        self.steps = 0
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self._obs = raw
        return self.observation()

    def wait_for_objects(self, n: int | None = None) -> LiberoObs:
        """Official eval: dummy steps so dropped objects settle."""
        n = self.catalog.num_steps_wait if n is None else int(n)
        hold = dummy_action()
        for _ in range(n):
            self.step(hold, count=False)
        return self.observation()

    def step(self, action, *, count: bool = True) -> LiberoObs:
        if self._env is None:
            raise RuntimeError("env is closed")
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.size < 7:
            raise ValueError(f"LIBERO action must be 7-d, got {act.shape}")
        raw, reward, done, info = self._env.step(act[:7].tolist())
        self._obs = raw
        self.reward = float(reward)
        self.done = bool(done)
        self.truncated = bool(info.get("truncated", False)) if isinstance(info, dict) else False
        if count:
            self.steps += 1
            if self.steps >= self.max_steps:
                self.truncated = True
        return self.observation()

    def set_sim_state(self, state: np.ndarray) -> LiberoObs:
        if self._env is None:
            raise RuntimeError("env is closed")
        self._obs = self._env.set_init_state(np.asarray(state))
        return self.observation()

    def get_sim_state(self) -> np.ndarray:
        return np.asarray(self._env.get_sim_state(), dtype=np.float64)

    def observation(self) -> LiberoObs:
        if self._obs is None:
            raise RuntimeError("no observation yet")
        raw = self._obs
        agent = rotate180(np.asarray(raw["agentview_image"]))
        wrist = rotate180(np.asarray(raw["robot0_eye_in_hand_image"]))
        return LiberoObs(
            agent_view=agent,
            wrist_view=wrist,
            state=proprio_state(raw),
            raw=raw,
        )

    def agent_pil(self) -> Image.Image:
        return Image.fromarray(self.observation().agent_view).convert("RGB")

    def wrist_pil(self) -> Image.Image:
        return Image.fromarray(self.observation().wrist_view).convert("RGB")
