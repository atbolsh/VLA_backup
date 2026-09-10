# EO-1 / LIBERO playground

From-scratch playground for [EO-1](https://huggingface.co/IPEC-COMMUNITY/EO-1-3B) on [LIBERO](https://libero-project.github.io/). Gradio UI: live robot view, suite/task dropdown, instruction edit, scene sliders, language log, cooperative interrupt + inject.

This is **not** a port of the LVA_shopping WidowX demo. The control loop matches official [`experiments/2_libero/eval_libero.py`](https://github.com/SHAILAB-IPEC/EO1/blob/main/experiments/2_libero/eval_libero.py).

The public Hub card is the **3B generalist**. The paper’s 98.2% LIBERO number is after a 50-epoch finetune that IPEC did not publish as a named card. Point `ROBOENV_CKPT` at a later tune if you have one.

## Vast.ai

1. Rent a CUDA box (5090 / 6090 / 5000 / 6000). Open **internal port 7860** (or set `ROBOENV_PORT`).
2. Clone this repo on the box.
3. Optional: `export HF_TOKEN=...` in `.env` to avoid Hub rate limits.
4. `bash setup.sh` — venv, torch, LIBERO, EO-1-3B (~6.5 GB). Does **not** start the UI.
5. `bash launch.sh` — prints the URL and login token, then serves Gradio.
6. Open the printed public URL (or the instance **Open** button if `OPEN_BUTTON_PORT=7860`). Log in with username `vastai` and the printed token (same as `$OPEN_BUTTON_TOKEN` / `$JUPYTER_TOKEN` when those exist).

Scene editor only (no 3B load):

```bash
bash launch.sh --scene
```

There is no `share=True` / `*.gradio.live` link.

## Layout

- `src/roboenv/policy/eo1.py` — `processor.generate` (text + action); `select_action` fallback
- `src/roboenv/env/libero.py` — `OffScreenRenderEnv`
- `src/roboenv/session/` — generate / act / freeze / inject
- `src/roboenv/scene/` — object sliders, save/load init state
- `src/roboenv/ui/` — Gradio
- `notes/LIBERO_SCENES.md` — BDDL / init-file railroad

## Interrupt and inject

EO-1 is one decoder. Interrupt stops **after** the current generate returns, or **between** action steps, and freezes the sim. Inject appends a Qwen `user` turn and an `Additional user instruction:` line on the official `task` string, then press Run.
