from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import gradio as gr

from roboenv.catalog import load_catalog, task_by_label, task_choices
from roboenv.paths import scenes_dir
from roboenv.scene.editor import SceneEditor
from roboenv.session.loop import Session
from roboenv.ui.auth import launch_auth

STATUS_COLOR = {
    "idle": "#586e75",
    "generating text": "#b58900",
    "acting": "#268bd2",
    "frozen": "#d33682",
}


def status_html(phase: str) -> str:
    color = STATUS_COLOR.get(phase, "#586e75")
    return (
        f"<div style='font-size:1.5rem;font-weight:700;letter-spacing:0.02em;"
        f"padding:14px 16px;border-radius:8px;background:{color};color:#fff'>"
        f"{phase}</div>"
    )


def _save_choices() -> list[str]:
    return [p.name for p in SceneEditor.list_saves()]


def _error_demo(message: str) -> gr.Blocks:
    with gr.Blocks(title="robo-env error") as demo:
        gr.Markdown("# robo-env failed to start")
        gr.Markdown(f"```\n{message}\n```")
        gr.Markdown(
            "This page is the real exception, not a missing-setup guess. "
            "`bash launch.sh` must use `.venv/bin/python` (conda `(main)` will steal `python`). "
            "Re-run setup only if `vendor/LIBERO` is actually missing."
        )
    return demo


def build_demo(*, scene_only: bool = False) -> gr.Blocks:
    cat = load_catalog()
    default_suite = "libero_object" if "libero_object" in cat.suites else cat.suite_names()[0]
    try:
        session = Session(suite=default_suite, task_id=0, load_policy=False)
    except Exception:  # noqa: BLE001
        return _error_demo(traceback.format_exc())
    editor = SceneEditor(session.env)
    default_tasks = task_choices(default_suite)
    default_label = default_tasks[0] if default_tasks else ""
    start = session._tick(note="ready")
    objs = editor.list_objects()
    movable = [o.name for o in objs if o.movable] or [o.name for o in objs]
    first = editor.get(movable[0]) if movable else None

    def current_views():
        tick = session._tick()
        return tick.agent_view, tick.wrist_view

    def refresh_object_sliders(name: str):
        if not name:
            return 0.0, 0.0, 0.0, 0.0
        obj = editor.get(name)
        return obj.x, obj.y, obj.z, obj.yaw

    def on_suite(suite: str):
        choices = task_choices(suite)
        label = choices[0] if choices else ""
        return gr.update(choices=choices, value=label)

    def on_open(suite: str, label: str, instruction: str):
        task = task_by_label(suite, label)
        tick = session.open_task(suite, task.task_id, instruction=task.language)
        names = editor.object_names(movable_only=True) or editor.object_names(False)
        name = names[0] if names else ""
        xs = refresh_object_sliders(name) if name else (0.0, 0.0, 0.0, 0.0)
        return (
            tick.agent_view,
            tick.wrist_view,
            status_html(tick.phase),
            tick.language_log,
            tick.instruction,
            gr.update(choices=names, value=name),
            xs[0],
            xs[1],
            xs[2],
            xs[3],
            gr.update(choices=_save_choices()),
        )

    def on_reset(instruction: str):
        tick = session.reset(instruction=instruction)
        names = editor.object_names(movable_only=True) or editor.object_names(False)
        name = names[0] if names else ""
        xs = refresh_object_sliders(name) if name else (0.0, 0.0, 0.0, 0.0)
        return (
            tick.agent_view,
            tick.wrist_view,
            status_html(tick.phase),
            tick.language_log,
            tick.instruction,
            gr.update(choices=names, value=name),
            xs[0],
            xs[1],
            xs[2],
            xs[3],
        )

    def on_apply(name, x, y, z, yaw):
        editor.apply(name, float(x), float(y), float(z), float(yaw))
        agent, wrist = current_views()
        return agent, wrist, status_html(session.phase), agent

    def on_save(save_name: str):
        name = (save_name or "").strip() or f"{session.env.suite}_{session.env.task_id:02d}"
        path = editor.save(name)
        return f"saved {path.name}", gr.update(choices=_save_choices(), value=path.name)

    def on_load(save_name: str):
        if not save_name:
            agent, wrist = current_views()
            return agent, wrist, status_html(session.phase), session.instruction, "no save selected"
        path = Path(save_name)
        if not path.is_file():
            path = scenes_dir() / save_name
        meta = editor.load(path)
        note = f"loaded {path.name}"
        if meta.get("suite") and meta.get("suite") != session.env.suite:
            note += f" (saved suite was {meta.get('suite')}[{meta.get('task_id')}])"
        agent, wrist = current_views()
        return agent, wrist, status_html(session.phase), session.instruction, note

    def on_run(instruction: str):
        for tick in session.iter_run(instruction=instruction):
            yield (
                tick.agent_view,
                tick.wrist_view,
                status_html(tick.phase),
                tick.language_log,
                tick.instruction,
                tick.note,
            )

    def on_interrupt():
        session.request_interrupt()
        return status_html("frozen")

    def on_inject(hint: str, instruction: str):
        session.set_instruction(instruction)
        tick = session.inject(hint)
        return (
            tick.agent_view,
            tick.wrist_view,
            status_html(tick.phase),
            tick.language_log,
            tick.instruction,
            "injected — press Run to resume",
        )

    title = "LIBERO scene editor" if scene_only else "EO-1 / LIBERO playground"
    with gr.Blocks(title=title, fill_height=True) as demo:
        gr.Markdown(f"# {title}")
        if scene_only:
            gr.Markdown(
                "Edit the table the robot will first see. Save a scene, then open "
                "the main app (`bash launch.sh`) and Load it."
            )
        else:
            gr.Markdown(
                "Public Hub weights are the EO-1-3B generalist. Official 98.2% "
                "LIBERO is an unpublished 50-epoch tune — drop a checkpoint later "
                "with `ROBOENV_CKPT`. First **Run** loads the 3B."
            )

        with gr.Group():
            gr.Markdown("### Scene (what the robot first sees)")
            with gr.Row():
                scene_preview = gr.Image(
                    value=start.agent_view,
                    label="scene preview (agent view)",
                    height=220,
                )
                with gr.Column():
                    obj_dd = gr.Dropdown(
                        choices=movable,
                        value=movable[0] if movable else None,
                        label="object",
                    )
                    sl_x = gr.Slider(-0.6, 0.8, value=first.x if first else 0, label="x", step=0.005)
                    sl_y = gr.Slider(-0.6, 0.6, value=first.y if first else 0, label="y", step=0.005)
                    sl_z = gr.Slider(0.0, 0.7, value=first.z if first else 0, label="z", step=0.005)
                    sl_yaw = gr.Slider(-3.14, 3.14, value=first.yaw if first else 0, label="yaw", step=0.01)
            with gr.Row():
                apply_btn = gr.Button("Apply pose")
                save_name = gr.Textbox(value="custom_scene", label="save as", scale=2)
                save_btn = gr.Button("Save scene")
                load_dd = gr.Dropdown(choices=_save_choices(), label="saved scenes", scale=2)
                load_btn = gr.Button("Load scene")
            scene_note = gr.Markdown("")

        agent = gr.Image(value=start.agent_view, label="robot view (agent)", height=420)
        wrist = gr.Image(value=start.wrist_view, label="wrist", height=200)
        status = gr.HTML(status_html(start.phase))
        note = gr.Markdown(start.note or "")
        language = gr.Markdown(start.language_log, label="model language")

        with gr.Row():
            suite = gr.Dropdown(
                choices=cat.suite_names(),
                value=default_suite,
                label="LIBERO suite",
            )
            task = gr.Dropdown(choices=default_tasks, value=default_label, label="task")
        instruction = gr.Textbox(
            value=session.instruction,
            lines=3,
            label="instruction (sent as EO-1 `task`)",
        )

        if not scene_only:
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary")
                interrupt_btn = gr.Button("Interrupt")
                reset_btn = gr.Button("Reset")
            inject_box = gr.Textbox(
                lines=2,
                label="inject hint (appended as a user turn + additional `task` line)",
            )
            inject_btn = gr.Button("Inject (while frozen)")
        else:
            reset_btn = gr.Button("Reset to canonical init")

        obj_dd.change(refresh_object_sliders, [obj_dd], [sl_x, sl_y, sl_z, sl_yaw])
        suite.change(on_suite, [suite], [task])
        open_outs = [
            agent,
            wrist,
            status,
            language,
            instruction,
            obj_dd,
            sl_x,
            sl_y,
            sl_z,
            sl_yaw,
            load_dd,
        ]
        task.change(on_open, [suite, task, instruction], open_outs)
        apply_btn.click(
            on_apply, [obj_dd, sl_x, sl_y, sl_z, sl_yaw], [agent, wrist, status, scene_preview]
        )
        save_btn.click(on_save, [save_name], [scene_note, load_dd])
        load_btn.click(on_load, [load_dd], [agent, wrist, status, instruction, scene_note])
        load_btn.click(lambda: current_views()[0], outputs=[scene_preview])

        if scene_only:
            reset_btn.click(
                on_reset,
                [instruction],
                [agent, wrist, status, language, instruction, obj_dd, sl_x, sl_y, sl_z, sl_yaw],
            )
        else:
            run_btn.click(
                on_run,
                [instruction],
                [agent, wrist, status, language, instruction, note],
            )
            interrupt_btn.click(on_interrupt, outputs=[status])
            reset_btn.click(
                on_reset,
                [instruction],
                [agent, wrist, status, language, instruction, obj_dd, sl_x, sl_y, sl_z, sl_yaw],
            )
            inject_btn.click(
                on_inject,
                [inject_box, instruction],
                [agent, wrist, status, language, instruction, note],
            )

    return demo


def launch(*, scene_only: bool = False) -> None:
    host = os.environ.get("ROBOENV_HOST", "0.0.0.0")
    port = int(os.environ.get("ROBOENV_PORT") or os.environ.get("PORT") or 7860)
    auth, message = launch_auth()
    demo = build_demo(scene_only=scene_only)
    demo.queue()
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        auth=auth,
        auth_message=message,
        inbrowser=False,
        show_error=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="store_true", help="scene editor only (no 3B)")
    args = parser.parse_args()
    launch(scene_only=args.scene)


if __name__ == "__main__":
    main()
