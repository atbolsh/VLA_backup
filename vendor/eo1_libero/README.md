# Official EO-1 LIBERO railroad (vendored)

Fetched by `setup.sh` from [SHAILAB-IPEC/EO1 `experiments/2_libero`](https://github.com/SHAILAB-IPEC/EO1/tree/main/experiments/2_libero).

`eval_libero.py` is the closed-loop we match: `OffScreenRenderEnv`, 256px, 180° image flip, `task` + `repo_id` batch, gripper post-process, replan 8.

This playground does **not** shell out to that script. We own the session loop so interrupt / inject / Gradio streaming work. The copy is here so the policy wrapper can stay honest if upstream moves.
