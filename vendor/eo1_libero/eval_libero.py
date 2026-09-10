# Vendored from SHAILAB-IPEC/EO1 experiments/2_libero/eval_libero.py
# setup.sh refreshes this file. Kept so the playground can match the railroad
# without cloning the full EO1 repo. Not imported by roboenv at runtime.

import collections
import dataclasses
import datetime as dt
import json
import logging
import math
import pathlib
from pathlib import Path

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    replan_steps: int = 8
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    video_out_path: str = "experiments/2_libero/logs/videos"
    seed: int = 7
    pretrained_path: str = ""
    post_process_action: bool = True
    job_name: str = "test"
    debug: bool = False


def eval_libero(args: Args) -> None:
    date_base = Path("experiments/2_libero/logs", dt.datetime.now().strftime("%Y-%m-%d/%H-%M-%S"))
    date_base.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=f"{date_base}+{args.job_name}.log")
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    args.video_out_path = f"{date_base}+{args.job_name}"
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520 + 50
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    from transformers import AutoModel, AutoProcessor

    model = (
        AutoModel.from_pretrained(
            args.pretrained_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(
        args.pretrained_path, trust_remote_code=True, local_files_only=True
    )

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []
            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_images.append(img)
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    if not action_plan:
                        img = Image.fromarray(img)
                        wrist_img = Image.fromarray(wrist_img)
                        batch = {
                            "observation.images.image": [img],
                            "observation.images.wrist_image": [wrist_img],
                            "observation.state": [state],
                            "task": [str(task_description)],
                            "repo_id": [f"{args.task_suite_name}_no_noops_1.0.0_lerobot"],
                        }
                        ov_out = processor.select_action(model, batch)
                        action_chunk = ov_out.action[0].numpy()
                        if args.post_process_action:
                            action_chunk[..., -1] = 2 * (1 - action_chunk[..., -1]) - 1
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    if args.debug:
                        raise e
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    tyro.cli(eval_libero)
