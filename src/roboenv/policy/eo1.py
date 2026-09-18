"""Official EO-1 inference: mouth = model.generate, arm = processor.select_action.

The Hub processor is EO1VisionProcessor. It has no .generate. Official eval
(experiments/2_libero/eval_libero.py) only calls select_action. Language uses
the README VL path: apply_chat_template + model.generate + processor.decode.
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

ASK_SUFFIX = "In one English sentence, what is the next subtask you will execute?"
# Official VL-eval sampling, but short enough for a live loop.
ASK_GEN = dict(max_new_tokens=128, repetition_penalty=1.0)


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
    arr = np.ascontiguousarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax <= 1.0:
            arr = np.clip(arr, 0, 1) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
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
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


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
                dtype=torch.bfloat16,
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

    def _to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _batch(self, image, wrist, state, task: str, repo_id: str) -> dict[str, Any]:
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        return {
            "observation.images.image": [_as_pil(image)],
            "observation.images.wrist_image": [_as_pil(wrist)],
            "observation.state": [state_arr],
            "task": [str(task)],
            "repo_id": [str(repo_id)],
        }

    def _reason_messages(self, image, wrist, task: str, extra: list | None = None) -> list[dict]:
        content = [
            {"type": "image", "image": _as_pil(image)},
            {"type": "image", "image": _as_pil(wrist)},
            {"type": "text", "text": f"{str(task).strip()}\n{ASK_SUFFIX}"},
        ]
        messages: list[dict] = [{"role": "user", "content": content}]
        for msg in extra or []:
            messages.append(msg)
        return messages

    def _pack_reason_inputs(self, messages: list[dict]) -> Any:
        try:
            packed = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            return packed
        except Exception:
            from qwen_vl_utils import process_vision_info

            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos = process_vision_info(messages)
            return self.processor(
                text=prompt,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )

    def ask(self, image, wrist, task: str, extra_messages: list | None = None) -> str:
        import torch

        messages = self._reason_messages(image, wrist, task, extra_messages)
        inputs = self._to_device(self._pack_reason_inputs(messages))
        if isinstance(inputs, dict):
            in_len = int(inputs["input_ids"].shape[1])
            kwargs = inputs
        else:
            in_len = int(inputs["input_ids"].shape[1])
            kwargs = dict(inputs)
        with torch.inference_mode():
            seq = self.model.generate(**kwargs, **ASK_GEN)
        if hasattr(seq, "sequences"):
            seq = seq.sequences
        decode = getattr(self.processor, "decode", None) or self.processor.tokenizer.decode
        return decode(seq[0, in_len:], skip_special_tokens=True).strip()

    def select_action(self, image, wrist, state, task: str, repo_id: str) -> np.ndarray | None:
        import torch

        batch = self._batch(image, wrist, state, task, repo_id)
        with torch.inference_mode():
            out = self.processor.select_action(self.model, batch)
        action = _as_chunk(getattr(out, "action", out))
        if action is not None and self.post_process_action:
            action = postprocess_gripper(action)
        return action

    def generate(
        self,
        image,
        wrist,
        state,
        task: str,
        repo_id: str,
        extra_messages: list | None = None,
    ) -> GenerateResult:
        text = ""
        bits: list[str] = []
        try:
            text = _first_text(self.ask(image, wrist, task, extra_messages))
            bits.append("model.generate")
        except Exception as exc:  # noqa: BLE001
            bits.append(f"model.generate failed ({type(exc).__name__}: {exc})")
        action = self.select_action(image, wrist, state, task, repo_id)
        bits.append("select_action")
        return GenerateResult(
            text=text,
            action_chunk=action,
            path=" ; ".join(bits),
            repo_id=repo_id,
            task=task,
        )
