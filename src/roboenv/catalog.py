from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from roboenv.paths import tasks_yaml_path


@dataclass(frozen=True)
class SuiteInfo:
    name: str
    label: str
    max_steps: int
    repo_id: str
    repo_id_alts: tuple[str, ...] = ()
    fallback_tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskInfo:
    suite: str
    task_id: int
    language: str
    name: str = ""
    label: str = ""


@dataclass
class Catalog:
    resolution: int = 256
    num_steps_wait: int = 10
    replan_steps: int = 8
    seed: int = 7
    suites: dict[str, SuiteInfo] = field(default_factory=dict)

    def suite_names(self) -> list[str]:
        return list(self.suites.keys())

    def suite(self, name: str) -> SuiteInfo:
        if name not in self.suites:
            raise KeyError(f"Unknown suite {name!r}. Known: {self.suite_names()}")
        return self.suites[name]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def load_catalog(path: Path | None = None) -> Catalog:
    raw = _load_yaml(path or tasks_yaml_path())
    suites: dict[str, SuiteInfo] = {}
    for name, spec in (raw.get("suites") or {}).items():
        suites[name] = SuiteInfo(
            name=name,
            label=spec.get("label", name),
            max_steps=int(spec["max_steps"]),
            repo_id=spec["repo_id"],
            repo_id_alts=tuple(spec.get("repo_id_alts") or ()),
            fallback_tasks=tuple(spec.get("tasks") or ()),
        )
    return Catalog(
        resolution=int(raw.get("resolution", 256)),
        num_steps_wait=int(raw.get("num_steps_wait", 10)),
        replan_steps=int(raw.get("replan_steps", 8)),
        seed=int(raw.get("seed", 7)),
        suites=suites,
    )


def discover_tasks(suite: str) -> list[TaskInfo]:
    """Prefer live LIBERO benchmark language; fall back to configs/tasks.yaml."""
    cat = load_catalog()
    info = cat.suite(suite)
    try:
        from libero.libero import benchmark

        suite_obj = benchmark.get_benchmark_dict()[suite]()
        out: list[TaskInfo] = []
        for task_id in range(suite_obj.n_tasks):
            task = suite_obj.get_task(task_id)
            language = str(task.language)
            name = str(getattr(task, "name", "") or f"task_{task_id:02d}")
            out.append(
                TaskInfo(
                    suite=suite,
                    task_id=task_id,
                    language=language,
                    name=name,
                    label=f"{task_id:02d}. {language}",
                )
            )
        if out:
            return out
    except Exception:
        pass
    return [
        TaskInfo(
            suite=suite,
            task_id=i,
            language=lang,
            name=f"task_{i:02d}",
            label=f"{i:02d}. {lang}",
        )
        for i, lang in enumerate(info.fallback_tasks)
    ]


def task_choices(suite: str) -> list[str]:
    return [t.label for t in discover_tasks(suite)]


def task_by_label(suite: str, label: str) -> TaskInfo:
    for task in discover_tasks(suite):
        if task.label == label or task.language == label:
            return task
    # "00. language" or a raw int
    if "." in label:
        head = label.split(".", 1)[0].strip()
        if head.isdigit():
            return task_by_id(suite, int(head))
    raise KeyError(f"No task {label!r} in {suite}")


def task_by_id(suite: str, task_id: int) -> TaskInfo:
    tasks = discover_tasks(suite)
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(f"No task_id {task_id} in {suite}")
