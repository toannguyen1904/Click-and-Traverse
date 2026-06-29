"""DAgger x RL distillation entry point for G1CaTra.

Distills several privileged specialists into one generalist single-agent G1CaTra
policy. Teachers are either all single-agent `G1CaTraPri` or all two-agent
`G1CaTra2APri` (never mixed in one run); each specialist owns one obstacle scene
and supervises the student envs routed to that scene (`pf_id`). The schedule is
two-phase: KL imitation for the first `--dagger_timesteps` env steps, then PPO.

Example:
    python train_ppo_dagger.py --task G1CaTra \\
        --teacher_restore_names runA runB runC \\
        --teacher_kind single \\
        --dagger_timesteps 100_000_000 \\
        --warmstart_states_path data/warmstart/catra.npz \\
        --exp_name dagger_v1
"""

import functools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import tyro
from absl import logging

import cat_ppo
from cat_ppo.constant import get_latest_ckpt
from cat_ppo.learning.policy.ppo import train as ppo
from cat_ppo.learning.policy.ppo import train_2a as ppo_2a
from train_ppo import (
    Args,
    _apply_args_to_config,
    _init_wandb,
    _log_checkpoint_path,
    _prepare_exp_name,
    _prepare_training_params,
    _progress,
    _report_training_time,
    _setup_paths,
    _validate_exp_name_format,
)

# Student task -> its DAgger env. Distill into a single-agent G1CaTra student
# (G1CaTraDagger) or a two-agent G1CaTra2A student (G1CaTra2ADagger).
_SINGLE_STUDENT_TASKS = ("G1CaTra", "G1CaTraPri", "G1CaTraDagger")
_TWO_AGENT_STUDENT_TASKS = ("G1CaTra2A", "G1CaTra2APri", "G1CaTra2ADagger")


@dataclass
class DaggerArgs(Args):
    teacher_restore_names: list[str] = field(default_factory=list)
    # "single" -> G1CaTraPri teachers; "2a" -> G1CaTra2APri teachers; "auto"
    # detects from the first teacher's config.json (and asserts all agree).
    teacher_kind: str = "auto"
    # "two_phase": DAgger KL until --dagger_timesteps, then PPO.
    # "blend": every step lambda_ppo*L_PPO + lambda_dagger*L_DAgger, lambda_dagger
    #          annealed max(floor, 1 - env_steps/blend_anneal_timesteps) (never off).
    dagger_mode: str = "two_phase"
    dagger_timesteps: int = 0
    blend_lambda_floor: float = 0.1
    blend_anneal_timesteps: int = 0  # 0 -> num_timesteps // 2 (paper's K/2)
    dagger_actor_loss_scale: float = 1.0
    dagger_value_loss_scale: float = 1.0
    pf_sampling_weights: list[float] = field(default_factory=list)
    pf_sampling_alpha: float = 1.0
    pf_sampling_ema_decay: float = 0.95


def _dagger_task_name(task: str) -> str:
    if task in _TWO_AGENT_STUDENT_TASKS:
        return "G1CaTra2ADagger"
    if task in _SINGLE_STUDENT_TASKS:
        return "G1CaTraDagger"
    raise ValueError(
        f"train_ppo_dagger distills into G1CaTra / G1CaTra2A; got task={task!r}"
    )


def _checkpoint_config_path(ckpt_path: Path) -> Path:
    if ckpt_path.name.isdigit():
        return ckpt_path.parent / "config.json"
    if ckpt_path.name == "checkpoints":
        return ckpt_path / "config.json"
    return ckpt_path / "checkpoints" / "config.json"


def _load_teacher_config(ckpt_path: str) -> dict:
    config_path = _checkpoint_config_path(Path(ckpt_path))
    if not config_path.exists():
        raise ValueError(f"Missing teacher config.json for DAgger checkpoint: {ckpt_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _detect_teacher_kind(teacher_config: dict) -> str:
    """A teacher is two-agent iff its env_config carries the lower/upper split."""
    env_cfg = teacher_config.get("env_config", {})
    return "2a" if "num_act_lower" in env_cfg else "single"


def _prepare_dagger_config(policy_cfg, env_config, args: DaggerArgs, student_is_2a: bool):
    dagger_cfg = getattr(policy_cfg, "dagger_config", None)
    if dagger_cfg is None or not getattr(dagger_cfg, "enable", False):
        raise ValueError("train_ppo_dagger requires a task with enabled dagger_config")

    teacher_names = list(args.teacher_restore_names) or list(
        getattr(dagger_cfg, "teacher_restore_names", [])
    )
    if not teacher_names:
        raise ValueError("Set --teacher_restore_names")

    teacher_checkpoint_paths = []
    for teacher_name in teacher_names:
        ckpt = get_latest_ckpt(teacher_name)
        if ckpt is None:
            raise ValueError(f"No checkpoint found for DAgger teacher run: {teacher_name}")
        teacher_checkpoint_paths.append(str(ckpt))

    # Resolve each teacher's scene, validate a shared grid, and detect the kind.
    scene_paths = []
    kinds = []
    box_inflations = []
    teacher_configs = []
    for ckpt in teacher_checkpoint_paths:
        teacher_config = _load_teacher_config(ckpt)
        teacher_configs.append(teacher_config)
        teacher_env = teacher_config["env_config"]
        teacher_pf_config = teacher_env["pf_config"]
        teacher_dx = teacher_pf_config.get("dx", env_config.pf_config.dx)
        if float(teacher_dx) != float(env_config.pf_config.dx):
            raise ValueError(
                f"Teacher scene dx mismatch for {ckpt}: teacher dx={teacher_dx}, "
                f"student dx={env_config.pf_config.dx}"
            )
        scene_paths.append(teacher_pf_config["path"])
        kinds.append(_detect_teacher_kind(teacher_config))
        box_inflations.append(
            bool(teacher_env.get("box_use_inflation", env_config.box_use_inflation))
        )

    if len(set(kinds)) != 1:
        raise ValueError(
            f"All teachers must be the same kind, got a mix: {dict(zip(teacher_names, kinds))}"
        )
    detected_kind = kinds[0]

    # The box guidance field feeds the teacher observation (hands + box-corner PF
    # blocks), so the student must build it the same way every teacher did.
    if len(set(box_inflations)) != 1:
        raise ValueError(
            "All teachers must share box_use_inflation, got a mix: "
            f"{dict(zip(teacher_names, box_inflations))}"
        )
    env_config.box_use_inflation = box_inflations[0]
    if args.teacher_kind != "auto" and args.teacher_kind != detected_kind:
        raise ValueError(
            f"--teacher_kind={args.teacher_kind!r} but teachers look like {detected_kind!r}"
        )
    # Enforce single-single / two-two: the student's agent structure must match the
    # teachers'. A two-agent student needs two-agent teachers and vice versa.
    expected_kind = "2a" if student_is_2a else "single"
    if detected_kind != expected_kind:
        raise ValueError(
            f"Student task is {'two-agent' if student_is_2a else 'single-agent'} "
            f"(expects {expected_kind!r} teachers) but teachers are {detected_kind!r}. "
            "Use single-agent teachers for a single-agent student and two-agent "
            "teachers for a two-agent student."
        )
    teacher_kind = detected_kind

    env_config.pf_config.paths = scene_paths
    env_config.pf_config.path = scene_paths[0]
    if args.pf_sampling_weights:
        if len(args.pf_sampling_weights) != len(scene_paths):
            raise ValueError(
                f"--pf_sampling_weights length must match --teacher_restore_names: "
                f"{len(args.pf_sampling_weights)} != {len(scene_paths)}"
            )
        env_config.pf_config.sampling_weights = args.pf_sampling_weights
    else:
        env_config.pf_config.sampling_weights = [1.0] * len(scene_paths)
    env_config.pf_config.sampling_alpha = args.pf_sampling_alpha
    env_config.pf_config.sampling_ema_decay = args.pf_sampling_ema_decay

    dagger_cfg.teacher_restore_names = teacher_names
    dagger_cfg.teacher_checkpoint_paths = teacher_checkpoint_paths
    dagger_cfg.teacher_kind = teacher_kind
    if args.dagger_mode not in ("two_phase", "blend"):
        raise ValueError(f"--dagger_mode must be 'two_phase' or 'blend', got {args.dagger_mode!r}")
    dagger_cfg.dagger_mode = args.dagger_mode
    dagger_cfg.dagger_timesteps = args.dagger_timesteps or (policy_cfg.num_timesteps // 2)
    dagger_cfg.blend_lambda_floor = args.blend_lambda_floor
    dagger_cfg.blend_anneal_timesteps = args.blend_anneal_timesteps or (policy_cfg.num_timesteps // 2)
    if args.dagger_actor_loss_scale <= 0:
        raise ValueError("dagger_actor_loss_scale must be > 0 (DAgger phase trains the actor).")
    dagger_cfg.actor_loss_scale = args.dagger_actor_loss_scale
    dagger_cfg.value_loss_scale = args.dagger_value_loss_scale

    if teacher_kind == "2a":
        t_env = teacher_configs[0]["env_config"]
        t_net = teacher_configs[0]["policy_config"]["network_factory"]
        dagger_cfg.teacher_action_size_lower = int(t_env["num_act_lower"])
        dagger_cfg.teacher_action_size_upper = int(t_env["num_act_upper"])
        dagger_cfg.teacher_policy_hidden_layer_sizes = list(
            t_net["policy_hidden_layer_sizes"]
        )


def train(args: DaggerArgs):
    task_name = _dagger_task_name(args.task)
    env_class = cat_ppo.registry.get(task_name, "train_env_class")
    task_cfg = cat_ppo.registry.get(task_name, "config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    dagger_keys = (
        "teacher_restore_names",
        "teacher_kind",
        "dagger_mode",
        "dagger_timesteps",
        "blend_lambda_floor",
        "blend_anneal_timesteps",
        "dagger_actor_loss_scale",
        "dagger_value_loss_scale",
        "pf_sampling_weights",
        "pf_sampling_alpha",
        "pf_sampling_ema_decay",
    )
    base_args = {k: v for k, v in args.__dict__.items() if k not in dagger_keys}
    train_args = Args(**{**base_args, "task": task_name})

    exp_name = _prepare_exp_name(task_name, train_args.generate_exp_name())
    debug_mode = "debug" in exp_name
    _validate_exp_name_format(exp_name, debug_mode)

    logdir, ckpt_path = _setup_paths(exp_name)
    _log_checkpoint_path(ckpt_path)

    is_2a = args.task in _TWO_AGENT_STUDENT_TASKS

    _apply_args_to_config(train_args, policy_cfg, env_cfg, debug_mode)
    _prepare_dagger_config(policy_cfg, env_cfg, args, student_is_2a=is_2a)
    task_cfg.env_config = env_cfg
    policy_params = _prepare_training_params(policy_cfg, ckpt_path)

    # Two-agent student: swap in the 2A trainer + network factory (separate
    # upper/lower actor+critic), mirroring train_ppo.py's dispatch.
    if is_2a:
        from cat_ppo.learning.policy.ppo.networks_2a import make_ppo_networks_2a

        policy_params["network_factory"] = functools.partial(
            make_ppo_networks_2a,
            action_size_lower=env_cfg.num_act_lower,
            action_size_upper=env_cfg.num_act_upper,
            **task_cfg.policy_config.network_factory,
        )
    trainer = ppo_2a.train if is_2a else ppo.train

    if not debug_mode:
        _init_wandb(train_args, exp_name, env_class, task_cfg, ckpt_path)

    train_fn = functools.partial(trainer, **policy_params)
    times = [time.monotonic()]

    env = env_class(task_type=env_cfg.task_type, config=env_cfg)
    eval_env = env_class(task_type=env_cfg.task_type, config=env_cfg)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=lambda s, m: _progress(
            s, m, times, policy_cfg.num_timesteps, debug_mode, exp_name
        ),
        eval_env=eval_env,
        policy_params_fn=lambda *a: None,
    )

    _report_training_time(times)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    logging.info(f"Run {exp_name} DAgger train done.")

    # Export the distilled student to ONNX (two-agent or single-agent actor).
    if train_args.convert_onnx:
        try:
            if is_2a:
                from cat_ppo.eval.brax2onnx import export_2a_onnx, get_latest_ckpt

                ckpt_dir = get_latest_ckpt(ckpt_path)
                obs_size = {
                    "privileged_state": (env_cfg.num_pri,),
                    "state": (env_cfg.num_obs,),
                }
                export_2a_onnx(
                    ckpt_dir, params, obs_size,
                    env_cfg.num_act_lower, env_cfg.num_act_upper,
                    policy_cfg.network_factory,
                )
            else:
                from cat_ppo.eval.brax2onnx import convert_jax2onnx, get_latest_ckpt

                ckpt_dir = get_latest_ckpt(ckpt_path)
                obs_size = {
                    "privileged_state": (env_cfg.num_pri,),
                    "state": (env_cfg.num_obs,),
                }
                convert_jax2onnx(
                    ckpt_dir=ckpt_dir,
                    output_path=f"{ckpt_dir}/policy.onnx",
                    inference_fn=inference_fn,
                    hidden_layer_sizes=policy_cfg.network_factory.policy_hidden_layer_sizes,
                    obs_size=obs_size,
                    action_size=env_cfg.num_act,
                    policy_obs_key=policy_cfg.network_factory.policy_obs_key,
                    jax_params=params,
                    activation="swish",
                )
        except ImportError:
            logging.warning(
                "TensorFlow is not installed. Please install TensorFlow to use ONNX conversion."
            )


if __name__ == "__main__":
    train(tyro.cli(DaggerArgs))
