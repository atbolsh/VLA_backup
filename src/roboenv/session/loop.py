from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from roboenv.catalog import load_catalog
from roboenv.env.libero import LiberoEnv
from roboenv.env.transforms import to_display_image
from roboenv.policy.eo1 import EO1Policy
from roboenv.session.conversation import Conversation

PHASE_IDLE = "idle"
PHASE_GENERATING = "generating text"
PHASE_ACTING = "acting"
PHASE_FROZEN = "frozen"


@dataclass
class Tick:
    phase: str
    agent_view: Image.Image
    wrist_view: Image.Image
    language_log: str
    instruction: str
    steps: int
    done: bool
    reward: float
    note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


class Session:
    """Cooperative generate → act → interrupt → inject loop."""

    def __init__(
        self,
        suite: str = "libero_object",
        task_id: int = 0,
        weights: Path | str | None = None,
        load_policy: bool = True,
        device: str = "cuda",
    ):
        self.catalog = load_catalog()
        self.env = LiberoEnv(suite=suite, task_id=task_id)
        self.conversation = Conversation(self.env.task_language)
        self.instruction = self.env.task_language
        self.language_entries: list[str] = []
        self.phase = PHASE_IDLE
        self.policy: EO1Policy | None = None
        ckpt = weights or os.environ.get("ROBOENV_CKPT")
        self._policy_kwargs = {"weights": ckpt, "device": device}
        self._interrupt = threading.Event()
        self._lock = threading.RLock()
        self._running = threading.Event()
        self.init_index = 0
        if load_policy:
            self.ensure_policy()
        self.env.wait_for_objects()

    def ensure_policy(self) -> EO1Policy:
        if self.policy is None:
            self.policy = EO1Policy(**self._policy_kwargs)
        return self.policy

    def request_interrupt(self) -> None:
        self._interrupt.set()

    def _tick(self, note: str = "", **extras) -> Tick:
        obs = self.env.observation()
        return Tick(
            phase=self.phase,
            agent_view=to_display_image(obs.agent_view),
            wrist_view=to_display_image(obs.wrist_view),
            language_log=self.language_markdown(),
            instruction=self.instruction,
            steps=self.env.steps,
            done=self.env.done,
            reward=self.env.reward,
            note=note,
            extras=extras,
        )

    def language_markdown(self) -> str:
        if not self.language_entries:
            return "_no language yet_"
        return "\n\n".join(self.language_entries)

    def _append_language(self, text: str, path: str) -> None:
        body = (text or "").strip() or "_(empty)_"
        n = len(self.language_entries) + 1
        self.language_entries.append(f"**{n}.** {body}\n\n`{path}`")
        self.conversation.append_assistant(text)

    def open_task(self, suite: str, task_id: int, instruction: str | None = None) -> Tick:
        with self._lock:
            self._interrupt.set()
            self.env.open(suite, task_id)
            self.env.wait_for_objects()
            lang = instruction if instruction is not None else self.env.task_language
            self.instruction = lang
            self.conversation.reset(lang)
            self.language_entries = []
            self.phase = PHASE_IDLE
            self.init_index = 0
            return self._tick(note=f"opened {suite}[{task_id}]")

    def reset(self, instruction: str | None = None) -> Tick:
        with self._lock:
            self._interrupt.set()
            self.env.reset(init_index=self.init_index)
            self.env.wait_for_objects()
            lang = instruction if instruction is not None else self.env.task_language
            self.instruction = lang
            self.conversation.reset(lang)
            self.language_entries = []
            self.phase = PHASE_IDLE
            return self._tick(note="reset")

    def set_instruction(self, text: str) -> None:
        self.instruction = text
        self.conversation.task = text

    def inject(self, text: str) -> Tick:
        with self._lock:
            image = self.env.agent_pil()
            self.instruction = self.conversation.inject_user(text, image=image)
            self.phase = PHASE_FROZEN
            return self._tick(note="injected user turn")

    def iter_run(self, instruction: str | None = None) -> Iterator[Tick]:
        if self._running.is_set():
            yield self._tick(note="already running")
            return
        self._running.set()
        self._interrupt.clear()
        try:
            with self._lock:
                if instruction is not None:
                    self.set_instruction(instruction)
            policy = self.ensure_policy()
            self.env.rebind_gl(force=True)
            replan = self.catalog.replan_steps
            while True:
                if self._interrupt.is_set():
                    self.phase = PHASE_FROZEN
                    yield self._tick(note="interrupted")
                    return
                if self.env.done or self.env.truncated:
                    self.phase = PHASE_IDLE
                    yield self._tick(note="episode finished")
                    return

                self.phase = PHASE_GENERATING
                yield self._tick(note="model.generate (language) then select_action")

                obs = self.env.observation()
                try:
                    result = policy.generate(
                        image=obs.agent_view,
                        wrist=obs.wrist_view,
                        state=obs.state,
                        task=self.conversation.task,
                        repo_id=self.env.repo_id,
                        extra_messages=self.conversation.messages,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.phase = PHASE_FROZEN
                    yield self._tick(note=f"generate failed: {type(exc).__name__}: {exc}")
                    return
                self._append_language(result.text, result.path)
                self.env.rebind_gl(force=True)
                yield self._tick(note="language ready", extras={"path": result.path})

                if self._interrupt.is_set():
                    self.phase = PHASE_FROZEN
                    yield self._tick(note="interrupted after generate")
                    return
                if result.action_chunk is None or len(result.action_chunk) == 0:
                    self.phase = PHASE_FROZEN
                    yield self._tick(note="no action chunk")
                    return

                chunk = result.action_chunk[:replan]
                self.phase = PHASE_ACTING
                for row in chunk:
                    if self._interrupt.is_set():
                        self.phase = PHASE_FROZEN
                        yield self._tick(note="interrupted during acting")
                        return
                    with self._lock:
                        self.env.step(row)
                    yield self._tick()
                    if self.env.done or self.env.truncated:
                        self.phase = PHASE_IDLE
                        yield self._tick(note="episode finished")
                        return
        finally:
            self._running.clear()
