"""Official EO-1 inference: processor.generate → text + action.

Matches SHAILAB-IPEC/EO1 experiments/2_libero/eval_libero.py for the batch
keys, 180° image flip (done in LiberoEnv), repo_id, and gripper post-process.
The Hub card's unified call is processor.generate. Official eval uses
select_action (act only). We prefer generate so the mouth is visible; if that
path fails we fall back to select_action and say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from roboenv.env.transforms import postprocess_gripper
from roboenv.paths import default_ckpt, robot_config_path


@dataclass
class GenerateResult:
    text: str
    action_chunk: np.ndarray | None
    path: str
    repo_id: str
    task: str


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _first_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        return "" if not text else _first_text(text[0])
    return str(text).strip()


def _as_chunk(action: Any) -> np.ndarray | None:
    if action is None:
        return None
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    arr = np.asarray(action)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return None
    return arr.astype(np.float32)


def load_robot_config(path: Path | None = None) -> dict[str, Any] | None:
    path = path or robot_config_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class EO1Policy:
    def __init__(
        self,
        weights: Path | str | None = None,
        device: str = "cuda",
        post_process_action: bool = True,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.weights = Path(weights) if weights else default_ckpt()
        if not self.weights.exists():
            raise FileNotFoundError(
                f"EO-1 weights not found at {self.weights}. Run bash setup.sh."
            )
        self.device = device
        self.post_process_action = post_process_action
        self.processor = AutoProcessor.from_pretrained(
            str(self.weights), trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                str(self.weights),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .to(device)
        )
        cfg = load_robot_config()
        if (
            cfg
            and cfg.get("features")
            and hasattr(self.processor, "set_normalization")
        ):
            self.processor.set_normalization(cfg)

    def _batch(self, image, wrist, state, task: str, repo_id: str) -> dict[str, Any]:
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        return {
            "observation.images.image": [_as_pil(image)],
            "observation.images.wrist_image": [_as_pil(wrist)],
            "observation.state": [state_arr],
            "task": [str(task)],
            "repo_id": [str(repo_id)],
        }

    def generate(
        self,
        image,
        wrist,
        state,
        task: str,
        repo_id: str,
    ) -> GenerateResult:
        import torch

        batch = self._batch(image, wrist, state, task, repo_id)
        text = ""
        action = None
        path = "processor.generate"
        try:
            with torch.inference_mode():
                out = self.processor.generate(self.model, batch)
            text = _first_text(getattr(out, "text", None))
            action = _as_chunk(getattr(out, "action", None))
            if action is None:
                raise RuntimeError("processor.generate returned no action")
        except Exception as exc:  # noqa: BLE001
            path = f"processor.generate failed ({type(exc).__name__}: {exc}); select_action"
            with torch.inference_mode():
                out = self.processor.select_action(self.model, batch)
            action = _as_chunk(getattr(out, "action", out))
        if action is not None and self.post_process_action:
            action = postprocess_gripper(action)
        return GenerateResult(
            text=text,
            action_chunk=action,
            path=path,
            repo_id=repo_id,
            task=task,
        )
