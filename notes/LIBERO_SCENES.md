# LIBERO scenes: load, modify, save

The Gradio strip at the top of the playground (and `bash launch.sh --scene`) moves free-joint objects and writes a flattened MuJoCo state. That is enough for “change what is on the table before the robot starts.”

This note is the programmer railroad: official LIBERO BDDL + init files.

## What a LIBERO task is

A task is three things:

1. **BDDL** — objects, fixtures, regions, init predicates, goal, language. Lives under the LIBERO package `bddl_files/<suite>/`.
2. **Init states** — arrays of `sim.get_state().flatten()` used for fair eval. `task_suite.get_task_init_states(task_id)`. Stored as `.init` / `.pruned_init` under `init_files/`.
3. **Language** — `task.language`, which we send to EO-1 as `task`.

Official walkthrough: [LIBERO `notebooks/procedural_creation_walkthrough.ipynb`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/notebooks/procedural_creation_walkthrough.ipynb). Docs: [Task generation](https://lifelong-robot-learning.github.io/LIBERO/html/procedural_generation/task_generation.html).

## Our saved scenes

`scenes/<name>.npz` contains:

- `sim_state` — `OffScreenRenderEnv.get_sim_state()`
- `meta` — suite, task_id, language, object poses at save time

Sidecar `scenes/<name>.json` is the same meta, readable without numpy.

Load in the UI, or:

```python
from pathlib import Path
from roboenv.env.libero import LiberoEnv
from roboenv.scene.editor import SceneEditor

env = LiberoEnv("libero_object", 0)
editor = SceneEditor(env)
editor.load(Path("scenes/custom_scene.npz"))
```

The state vector is tied to that task’s XML. Loading a kitchen save into an object-suite env will not match.

## New BDDL tasks (programmer)

1. Clone LIBERO (setup.sh already does this into `vendor/LIBERO`).
2. Register a scene class (`@register_mu`) with fixtures, objects, regions, `init_states`.
3. `register_task_info(language, scene_name=..., goal_states=...)`.
4. Generate the `.bddl` file with their procedural scripts / notebook.
5. Collect init states: reset many seeds, `env.get_sim_state()` (add that helper on `ControlEnv` if missing — see [LIBERO#34](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/34)).
6. Point the playground at the new suite/task or drop the language into `configs/tasks.yaml`.

You do **not** need the human demonstration hdf5 datasets to *play*. Those are for training.

## Official EO-1 eval (reference only)

```text
vendor/eo1_libero/eval_libero.py
```

Batch keys: `observation.images.image`, `observation.images.wrist_image`, `observation.state`, `task`, `repo_id=<suite>_no_noops_1.0.0_lerobot`. Images are rotated 180°. Gripper is remapped with `2 * (1 - g) - 1`.
