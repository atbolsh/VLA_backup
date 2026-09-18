from __future__ import annotations

import math

import numpy as np

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def dummy_action() -> list[float]:
    """Official eval wait action: hold still, gripper closed convention."""
    return list(DUMMY_ACTION)


def rotate180(image: np.ndarray) -> np.ndarray:
    """LIBERO train/eval preprocess: flip agent and wrist views 180 degrees."""
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1].copy())


def to_display_image(image: np.ndarray):
    """Stable RGB uint8 for Gradio. Copies so the sim cannot rewrite the frame."""
    from PIL import Image

    arr = np.ascontiguousarray(np.asarray(image)).copy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
    if arr.dtype != np.uint8:
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if np.isfinite(amax) and amax <= 1.0:
            arr = np.clip(arr, 0, 1) * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr).convert("RGB")


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite transform used by EO-1 eval_libero.py."""
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def proprio_state(obs: dict) -> np.ndarray:
    """8-d state: eef xyz + axis-angle + gripper qpos. Official eval_libero.py."""
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def quat_wxyz_to_yaw(quat: np.ndarray) -> float:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    w, x, y, z = q.tolist()
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def postprocess_gripper(action: np.ndarray) -> np.ndarray:
    """eval_libero.py: action[..., -1] = 2 * (1 - raw) - 1."""
    out = np.array(action, dtype=np.float32, copy=True)
    out[..., -1] = 2.0 * (1.0 - out[..., -1]) - 1.0
    return out
