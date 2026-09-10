"""EO-1 instruction channel.

Official LIBERO eval only sends `task` in the generate/select_action batch.
Qwen chat (`role: user` / `assistant`, apply_chat_template) is the same model's
VL railroad. Injection updates `task` (what acting sees) and appends a user
turn (what a later tool/chat layer can reuse). No invented roles.
"""

from __future__ import annotations

from typing import Any

from PIL import Image


def _image_note(image: Any) -> str:
    if isinstance(image, Image.Image):
        return f"<PIL {image.size[0]}x{image.size[1]} {image.mode}>"
    return "<image>"


class Conversation:
    def __init__(self, task: str = ""):
        self.task = task
        self.messages: list[dict[str, Any]] = []

    def reset(self, task: str) -> None:
        self.task = task
        self.messages = []

    def inject_user(self, text: str, image: Image.Image | None = None) -> str:
        text = (text or "").strip()
        if not text:
            return self.task
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})
        if self.task.strip():
            self.task = f"{self.task.rstrip()}\n\nAdditional user instruction: {text}"
        else:
            self.task = text
        return self.task

    def append_assistant(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.messages.append({"role": "assistant", "content": text})

    def for_print(self) -> list[dict[str, Any]]:
        out = []
        for msg in self.messages:
            item: dict[str, Any] = {"role": msg.get("role")}
            content = msg.get("content")
            if isinstance(content, str):
                item["content"] = content
            else:
                item["content"] = []
                for part in content or []:
                    p = dict(part)
                    if "image" in p:
                        p["image"] = _image_note(p["image"])
                    item["content"].append(p)
            out.append(item)
        return out
