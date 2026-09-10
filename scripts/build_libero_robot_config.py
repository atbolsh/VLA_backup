#!/usr/bin/env python3
"""Build EO-1 robot_config JSON from IPEC LIBERO LeRobot dataset meta (no videos)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "libero_robot_config.json"

DATASETS = {
    "libero_spatial_no_noops_1.0.0_lerobot": "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot": "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot": "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot": "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
    "libero_90_no_noops_1.0.0_lerobot": "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
    "libero_90_no_noops_lerobot": "IPEC-COMMUNITY/libero_90_no_noops_lerobot",
}

FEATURE_KEYS = (
    "action",
    "observation.state",
    "observation.images.image",
    "observation.images.wrist_image",
)


def _download_meta(repo_id: str, name: str) -> Path | None:
    try:
        return Path(hf_hub_download(repo_id, name, repo_type="dataset"))
    except Exception as exc:  # noqa: BLE001
        print(f"skip {repo_id}/{name}: {exc}", file=sys.stderr)
        return None


def _load_info(repo: str) -> dict | None:
    path = _download_meta(repo, "meta/info.json")
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_stats(repo: str) -> dict | None:
    path = _download_meta(repo, "meta/stats.json")
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    path = _download_meta(repo, "meta/episodes_stats.jsonl")
    if path is None:
        return None
    sums: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            stats = rec.get("stats") or {}
            for key in ("action", "observation.state"):
                block = stats.get(key) or {}
                for stat_name in ("mean", "std", "min", "max"):
                    if stat_name in block:
                        sums[key][stat_name].append(block[stat_name])
    if not sums:
        return None
    import numpy as np

    out = {}
    for key, stat_map in sums.items():
        out[key] = {}
        for stat_name, rows in stat_map.items():
            arr = np.asarray(rows, dtype=np.float64)
            if stat_name in ("min",):
                out[key][stat_name] = arr.min(axis=0).tolist()
            elif stat_name in ("max",):
                out[key][stat_name] = arr.max(axis=0).tolist()
            else:
                out[key][stat_name] = arr.mean(axis=0).tolist()
    return out


def _feature_entry(spec: dict) -> dict:
    names = spec.get("names")
    return {
        "dtype": spec.get("dtype", "float32"),
        "names": names,
        "shape": spec.get("shape"),
    }


def _unit_stats(shape) -> dict:
    n = int(shape[0]) if shape else 1
    zeros = [0.0] * n
    ones = [1.0] * n
    return {"mean": zeros, "std": ones, "min": zeros, "max": ones}


def main() -> int:
    features: dict = {}
    stats: dict = {}
    select_video = {}
    select_state = {}
    select_action = {}
    any_ok = False
    for local_id, repo in DATASETS.items():
        info = _load_info(repo)
        if info is None:
            continue
        any_ok = True
        feat = info.get("features") or {}
        features[local_id] = {
            key: _feature_entry(feat[key]) for key in FEATURE_KEYS if key in feat
        }
        got = _aggregate_stats(repo) or {}
        stats[local_id] = {
            "action": got.get("action") or _unit_stats(feat.get("action", {}).get("shape")),
            "observation.state": got.get("observation.state")
            or _unit_stats(feat.get("observation.state", {}).get("shape")),
        }
        select_video[local_id] = [
            k
            for k in ("observation.images.image", "observation.images.wrist_image")
            if k in feat
        ]
        select_state[local_id] = ["observation.state"]
        select_action[local_id] = ["action"]
        print(f"ok {local_id}")

    if not any_ok:
        print("no dataset meta downloaded; writing shape-only fallback", file=sys.stderr)
        fallback_ids = [
            "libero_spatial_no_noops_1.0.0_lerobot",
            "libero_object_no_noops_1.0.0_lerobot",
            "libero_goal_no_noops_1.0.0_lerobot",
            "libero_10_no_noops_1.0.0_lerobot",
            "libero_90_no_noops_1.0.0_lerobot",
        ]
        image = {
            "dtype": "video",
            "names": ["height", "width", "rgb"],
            "shape": [256, 256, 3],
        }
        for local_id in fallback_ids:
            features[local_id] = {
                "action": {
                    "dtype": "float32",
                    "names": {
                        "motors": [
                            "x",
                            "y",
                            "z",
                            "axis_angle1",
                            "axis_angle2",
                            "axis_angle3",
                            "gripper",
                        ]
                    },
                    "shape": [7],
                },
                "observation.state": {
                    "dtype": "float32",
                    "names": {
                        "motors": [
                            "x",
                            "y",
                            "z",
                            "axis_angle1",
                            "axis_angle2",
                            "axis_angle3",
                            "gripper",
                            "gripper",
                        ]
                    },
                    "shape": [8],
                },
                "observation.images.image": image,
                "observation.images.wrist_image": image,
            }
            stats[local_id] = {
                "action": _unit_stats([7]),
                "observation.state": _unit_stats([8]),
            }
            select_video[local_id] = [
                "observation.images.image",
                "observation.images.wrist_image",
            ]
            select_state[local_id] = ["observation.state"]
            select_action[local_id] = ["action"]

    cfg = {
        "action_chunk_size": 16,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "state_mode": "MEAN_STD",
        "features": features,
        "select_action_keys": select_action,
        "select_state_keys": select_state,
        "select_video_keys": select_video,
        "stats": stats,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
