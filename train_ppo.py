import inspect
import functools
import time
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from absl import logging
import tqdm
import tyro
import numpy as np
import jax

from mujoco_playground import wrapper
from brax.training.agents.ppo.networks import make_ppo_networks

import cat_ppo
from cat_ppo import update_file_handler
from cat_ppo.constant import PATH_LOG
from cat_ppo.learning.policy.ppo import train as ppo # brax.training.agents.ppo
from cat_ppo.learning.train.pf_utils import wrap_for_brax_training_reset

if os.environ.get("SWANLAB_SYNC_WANDB", "").strip().lower() in ("1", "true", "yes"):
    import swanlab

    swanlab.sync_wandb()

import wandb  # noqa: E402

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"  # allows more GPU matrix multiplications to use the Triton GEMM path
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # Stops JAX from grabbing most GPU memory upfront at startup.
os.environ["MUJOCO_GL"] = "egl" # for headless rendering

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")  # Project name in WandB
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

@dataclass
class Args:
    """
    Args for training the policy.
    """
    task: str
    exp_name: str = "debug"   # experiment name, used to identify the experiment in WandB and log directory
    num_timesteps: int = 400_000_000    # total number of environment steps to use during training
    seed: int = 42   # random seed for training
    convert_onnx: bool = True   # convert the trained policy to ONNX format for deployment
    restore_name: str = "none"   # name of the checkpoint to restore from, used to resume training
    ground: float = 0    # reward scale for feet body group: GF guidance alignment + SDF penalty vs ground-level obstacles
    lateral: float = 0   # reward scale for hands/knees/shoulders body group: GF guidance alignment + SDF penalty vs side obstacles
    overhead: float = 0  # reward scale for head body group: GF guidance alignment + SDF penalty vs overhead obstacles
    term_collision_threshold: float = 0.04  # SDF below -threshold triggers collision termination
    obs_path: str = 'data/assets/TypiObs/empty'  # path to the obstacle grid files: sdf.npy, bf.npy, gf.npy.
    stage1_steps: int = -1  # for G1CaTra: number of steps in stage 1 (pickup); -1 = use task default
    warmstart_states_path: str = ""  # path to pre-generated .npz from generate_warmstart_states.py; enables file-load warm-start in G1CaTra
    def generate_exp_name(self):
        # generate a unique experiment name based on the task, difficulty, and seed
        exp_name_parts = [self.exp_name]

        if self.ground != 0:
            exp_name_parts.append('G'+str(self.ground).replace('.', ''))

        if self.lateral != 0:
            exp_name_parts.append('L'+str(self.lateral).replace('.', ''))

        if self.overhead != 0:
            exp_name_parts.append('O'+str(self.overhead).replace('.', ''))

        exp_name_parts.append(f"T{str(self.term_collision_threshold).replace('.', '')}")

        if self.obs_path:
            exp_name_parts.append(self.obs_path.replace('/', '').replace('_', ''))

        return "x".join(exp_name_parts)

def _prepare_exp_name(task: str, exp_name: str) -> str:
    # add a timestamp to the experiment name
    timestamp = datetime.now().strftime("%m%d%H%M")
    return f"{timestamp}_{task}_{exp_name}"

def _validate_exp_name_format(exp_name: str, debug_mode: bool):
    # validate the experiment name format
    if not debug_mode and len(exp_name.split("_")) != 4:
        raise ValueError(
            f"exp_name should be in the format <task>_<tag>_<version>, got {exp_name}"
        )


def _setup_paths(exp_name: str) -> tuple[Path, Path]:
    # setup the log directory and checkpoint directory
    logdir = Path(PATH_LOG) / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    update_file_handler(filename=f"{logdir}/info.log")
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return logdir, ckpt_path


def _log_checkpoint_path(ckpt_path: Path):
    # log the checkpoint path
    logging.info(f"Checkpoint path: {ckpt_path}")


def _apply_args_to_config(args: Args, policy_cfg, env_config, debug: bool):
    """
    Apply command line arguments to the policy and environment configurations.
    """
    policy_cfg.num_timesteps = args.num_timesteps  # total env interaction budget; controls number of PPO training epochs and acts as the hard stop condition
    if debug:
        policy_cfg.training_metrics_steps = 1000
        policy_cfg.num_evals = 5
        policy_cfg.batch_size = 8
        policy_cfg.num_minibatches = 2
        policy_cfg.num_envs = policy_cfg.batch_size * policy_cfg.num_minibatches
        policy_cfg.episode_length = 200
        policy_cfg.unroll_length = 10
        policy_cfg.num_updates_per_batch = 1
        policy_cfg.action_repeat = 1
        policy_cfg.num_timesteps = 100_000
        policy_cfg.num_resets_per_eval = 1
        policy_cfg.num_eval_envs = 128
    # cfg.restore_checkpoint_path = Path(args.restore_checkpoint_path)
    if args.restore_name != "none":
        # get the latest checkpoint from the log directory if resuming training
        from cat_ppo.constant import get_latest_ckpt
        policy_cfg.restore_checkpoint_path = str(get_latest_ckpt(args.restore_name))
    env_config.reward_config.scales.feetgf = args.ground  # scale: align feet velocity with HumanoidPF guidance (ground)
    env_config.reward_config.scales.feetdf = args.ground  # scale: SDF penalty for feet vs ground obstacles
    env_config.reward_config.scales.headgf = args.overhead  # scale: align head motion with guidance (overhead)
    env_config.reward_config.scales.headdf = args.overhead  # scale: SDF penalty for head vs overhead obstacles
    env_config.reward_config.scales.handsgf = args.lateral  # scale: align hands with guidance (lateral / narrow)
    env_config.reward_config.scales.handsdf = args.lateral  # scale: SDF penalty for hands vs side obstacles
    env_config.reward_config.scales.kneesdf = args.lateral  # scale: SDF penalty for knees vs obstacles
    env_config.reward_config.scales.shldsdf = args.lateral  # scale: SDF penalty for shoulders vs obstacles
    env_config.term_collision_threshold = args.term_collision_threshold  # SDF below -threshold triggers collision termination
    if args.stage1_steps >= 0 and hasattr(env_config, "stage1_steps"):
        env_config.stage1_steps = args.stage1_steps  # override stage 1 length (G1CaTra only)
    if args.warmstart_states_path and hasattr(env_config, "warmstart_states_path"):
        env_config.warmstart_states_path = args.warmstart_states_path
        if args.stage1_steps < 0:
            env_config.stage1_steps = 0
        # Swap in a warmstart-aware DR function that carries box mass/size from the state file
        from cat_ppo.envs.g1.env_catra import make_warmstart_domain_randomize_catra
        policy_cfg.randomization_fn = make_warmstart_domain_randomize_catra(args.warmstart_states_path)
    env_config.pf_config.path = args.obs_path  # directory with sdf.npy, bf.npy, gf.npy for HumanoidPF

def _prepare_training_params(cfg, ckpt_path: Path):
    # Convert config to a **kwargs dict for Brax PPO train(), injecting the custom env wrapper and network factory callable.
    params = cfg.to_dict()
    params.pop("network_factory", None)  # remove the network factory from the parameters
    params["wrap_env_fn"] = wrap_for_brax_training_reset #wrapper.wrap_for_brax_training
    network_fn = make_ppo_networks  # make_ppo_networks is an available function from brax
    params["network_factory"] = (
        functools.partial(network_fn, **cfg.network_factory)
        if hasattr(cfg, "network_factory")
        else network_fn
    )
    params["save_checkpoint_path"] = ckpt_path
    return params


def _init_wandb(
    args: Args, exp_name, env_class, task_cfg, ckpt_path, config_fname="config.json"
):
    """
    WanDB initialization.
    """
    wandb.init(
        name=exp_name,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "num_timesteps": args.num_timesteps,
            "task": args.task,
        },
        dir=PATH_LOG,
    )
    wandb.config.update(task_cfg.to_dict())
    wandb.save(inspect.getfile(env_class))
    config_path = ckpt_path / config_fname
    config_path.write_text(task_cfg.to_json_best_effort(indent=4))


def _progress(num_steps, metrics, times, total_steps, debug_mode, exp_name):
    # progres callback function to log training progress and metrics, and estimate time remaining
    now = time.monotonic()
    times.append(now)
    if metrics and not debug_mode:
        try:
            wandb.log(metrics, step=num_steps)
        except Exception as e:
            logging.warning(f"wandb.log failed: {e}")

    if len(times) < 2 or num_steps == 0:
        return
    step_times = np.diff(times)
    median_step_time = np.median(step_times)
    if median_step_time <= 0:
        return
    steps_logged = num_steps / len(step_times)
    est_seconds_left = (total_steps - num_steps) / steps_logged * median_step_time
    logging.info(f"NumSteps {num_steps} - EstTimeLeft {est_seconds_left:.1f}[s]")
    logging.info(exp_name)


def _report_training_time(times):
    # report the time taken for JIT compilation and total training time, the JIT is a one-time cost that happens at the beginning of training when JAX compiles the training function for the first time, which can take a few minutes but is necessary for fast execution in subsequent steps
    if len(times) > 1:
        logging.info("Done training.")
        logging.info(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
        logging.info(f"Time to train: {times[-1] - times[1]:.2f}s")


def train(args: Args):
    """
    Main training function.
    """
    env_class = cat_ppo.registry.get(args.task, "train_env_class")  # args.task can be "G1Cat" or "G1CatPri". This returns the class of the environment, e.g. G1CatEnv or G1CatPriEnv.
    task_cfg = cat_ppo.registry.get(args.task, "config")  # ConfigDict with env_config, policy_config, eval_config for the given task.
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config
    eval_config = task_cfg.eval_config

    exp_name = _prepare_exp_name(args.task, args.generate_exp_name())
    debug_mode = "debug" in exp_name    # check if we are in the debug mode

    _validate_exp_name_format(exp_name, debug_mode)

    logdir, ckpt_path = _setup_paths(exp_name)
    _log_checkpoint_path(ckpt_path)

    _apply_args_to_config(args, policy_cfg, env_cfg, debug_mode)
    task_cfg.env_config = env_cfg
    policy_params = _prepare_training_params(policy_cfg, ckpt_path)

    # initialize wandb
    if not debug_mode:
        _init_wandb(args, exp_name, env_class, task_cfg, ckpt_path)

    train_fn = functools.partial(ppo.train, **policy_params)    # prefill arguments in policy_params. functols.partial(func, **kwargs) returns a new function with the arguments prefilled in the original function func.
    times = [time.monotonic()]

    env = env_class(task_type=env_cfg.task_type, config=env_cfg)
    _eval_env = env_class(task_type=env_cfg.task_type, config=env_cfg)

    # process
    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
        pass

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=lambda s, m: _progress(s, m, times, policy_cfg.num_timesteps, debug_mode, exp_name),
        eval_env=_eval_env,
        policy_params_fn=policy_params_fn,
    )

    _report_training_time(times)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    logging.info(f"Run {exp_name} Train done.")

    if args.convert_onnx:
        try:
            from cat_ppo.eval.brax2onnx import convert_jax2onnx, get_latest_ckpt

            ckpt_dir = get_latest_ckpt(ckpt_path)
            obs_size = {
                'privileged_state': (env_cfg.num_pri,),
                'state': (env_cfg.num_obs,),
            }
            act_size = env_cfg.num_act
            policy_obs_key = policy_cfg.network_factory.policy_obs_key
            convert_jax2onnx(
                ckpt_dir=ckpt_dir,
                output_path=f"{ckpt_dir}/policy.onnx",
                inference_fn=inference_fn,
                hidden_layer_sizes=policy_cfg.network_factory.policy_hidden_layer_sizes,
                obs_size=obs_size,
                action_size=act_size,
                policy_obs_key=policy_obs_key,
                jax_params=params,
                activation="swish",
            )
        except ImportError:
            logging.warning(
                "TensorFlow is not installed. Please install TensorFlow to use ONNX conversion."
            )


if __name__ == "__main__":
    train(tyro.cli(Args))
