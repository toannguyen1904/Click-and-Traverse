"""Headless ONNX evaluation for the single-stage G1CaTra (box carry & traverse) policy.

Mirrors cat_ppo/eval/mj_onnx_test.py for the Cat task, adapted to CaTra:
  * the play env (play_catra.PlayG1CaTraEnv) does NOT compute success/termination, so the
    scoring here replicates env_catra._get_termination (fall + body/box SDF collision +
    box-drop), plus a goal-reached success check.
  * supports both single-agent (one policy.onnx) and two-agent (policy_lower.onnx +
    policy_upper.onnx, outputs concatenated [lower, upper]) CaTra policies. Two-agent is
    auto-detected from the task name.
  * supports warm-start reset and init-pose perturbation for generalization testing.

Examples:
    # single-agent
    python -m cat_ppo.eval.mj_onnx_test --task G1CaTra --exp-name <run> \
        --obs-path data/assets/TypiObs/empty --num-episodes 50
    # two-agent (defaults to <ckpt>/policy_lower.onnx and <ckpt>/policy_upper.onnx)
    python -m cat_ppo.eval.mj_onnx_test --task G1CaTra2A --exp-name <run> \
        --obs-path data/assets/TypiObs/empty
"""
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=true"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import json
from dataclasses import dataclass
from collections import Counter
from typing import Optional

import numpy as np
import onnxruntime as rt
import tqdm
import tyro

import cat_ppo


# Two-agent CaTra tasks (lower-body + upper-body actors). Kept in sync with
# brax2onnx._TWO_AGENT_TASKS; "...Pri"/"...Dagger" variants are also matched by suffix below.
_TWO_AGENT_TASKS = ("G1CaTra2A", "G1CaTra2APri", "G1CaTra2ADagger")


def _is_two_agent(task: str) -> bool:
    return task in _TWO_AGENT_TASKS or task.startswith("G1CaTra2A")


# env_config fields that are baked into a training run and must match at playback time.
# Read from the checkpoint's config.json (registry defaults can differ from the run).
_CKPT_OVERRIDE_FIELDS = (
    "box_use_inflation",
    "episode_length",
    "term_collision_threshold",
    "box_drop_threshold",
    "warmstart_states_path",
)


@dataclass
class Args:
    task: str = "G1CaTra"
    exp_name: Optional[str] = None
    # single-agent policy (used when the task is not a two-agent task)
    onnx_path: Optional[str] = None     # default -> <ckpt>/policy.onnx
    # two-agent policies (used when the task is a *2A* task)
    onnx_path_lower: Optional[str] = None   # default -> <ckpt>/policy_lower.onnx
    onnx_path_upper: Optional[str] = None   # default -> <ckpt>/policy_upper.onnx
    obs_path: str = "data/assets/TypiObs/empty"
    pri: bool = False
    num_episodes: int = 50
    seed: int = 42
    goal_x: float = 1.6                 # base x (m) counted as a completed traversal
    use_ckpt_config: bool = True        # apply the run's config.json over the registry config
    # generalization-test knobs (passed straight to env.reset)
    warmstart_path: Optional[str] = None    # override config.warmstart_states_path; loads holding-box states
    warmstart_idx: int = -1             # -1 -> random warm-start each episode
    pos_offset: float = 0.0             # random init xy displacement [-v, v] m
    ang_offset_deg: float = 0.0         # random init yaw [-v, v] deg
    max_steps: Optional[int] = None     # default -> env_config.episode_length
    render: bool = False                # launch the MuJoCo viewer (needs a display)
    box_noise: bool = True              # add box position/orientation tracking noise to the deployable obs (False -> ground-truth box)


def _set_box_noise(env_cfg, enabled: bool):
    """Toggle box position/orientation observation noise for playback/eval.

    When disabled, zero the box noise scales (box_pos/box_ori) so the deployable obs sees the
    ground-truth box pose + PF. Proprio noise (gyro/gravity/joint) is left untouched. No-op if
    the config predates the box noise scales."""
    scales = getattr(getattr(env_cfg, "noise_config", None), "scales", None)
    if scales is None or not hasattr(scales, "box_pos"):
        return
    if enabled:
        print(f"[eval] box tracking noise ENABLED (box_pos={scales.box_pos}, box_ori={scales.box_ori})")
        return
    scales.box_pos = 0.0
    scales.box_ori = 0.0
    print("[eval] box tracking noise DISABLED (box_pos/box_ori scales set to 0)")


def _apply_ckpt_config(env_cfg, exp_name):
    """Override playback-relevant env_config fields from the run's checkpoints/config.json,
    so scoring (thresholds), the box field (box_use_inflation) and warm-start match how the
    policy was actually trained rather than the registry defaults."""
    if exp_name is None:
        print("[mj_onnx_test] no --exp-name; using registry config (pass explicit onnx paths).")
        return
    cfg_path = cat_ppo.get_path_log(exp_name) / "checkpoints" / "config.json"
    if not cfg_path.exists():
        print(f"[mj_onnx_test] config.json not found at {cfg_path}; using registry config.")
        return
    saved = json.loads(cfg_path.read_text()).get("env_config", {})
    applied = {}
    for field in _CKPT_OVERRIDE_FIELDS:
        if field in saved:
            setattr(env_cfg, field, saved[field])
            applied[field] = saved[field]
    print(f"[mj_onnx_test] applied checkpoint config: {applied}")


class _Policy:
    """ONNX action provider. Wraps either a single actor or two actors (lower+upper)
    whose 'continuous_actions' outputs are concatenated [lower, upper]. Both actors
    consume the same actor observation."""

    _OUT = ["continuous_actions"]

    def __init__(self, sessions):
        self._sessions = sessions  # list of rt.InferenceSession, in concat order

    @classmethod
    def load(cls, paths):
        sessions = []
        for p in paths:
            print(f"[mj_onnx_test] loading policy: {p}")
            sessions.append(rt.InferenceSession(str(p), providers=["CPUExecutionProvider"]))
        return cls(sessions)

    def act(self, obs: np.ndarray) -> np.ndarray:
        onnx_input = {"obs": obs}
        parts = [s.run(self._OUT, onnx_input)[0][0] for s in self._sessions]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _episode_status(env, state, cfg, goal_x, max_steps):
    """Replicate env_catra._get_termination for the CPU play env, plus a goal check.

    Returns (status, base_x) where status is one of {"success", "box_drop", "fall",
    "robot_collision", "box_collision", "timeout"} or None if still running.
    Geom-contact terms in the training env (foot-foot, box-thigh, box-head) are omitted
    because the play env does not expose those geom ids; the SDF/fall/box-drop terms below
    cover the dominant failure modes.
    """
    info = state.info
    step = int(info["step"])
    base_x = float(env.mj_data.qpos[0])
    box_z = float(env.mj_data.xpos[env._box_body_id][2])

    # Goal reached: traversed far enough in +x while still alive and holding the box.
    if base_x >= goal_x:
        return "success", base_x

    # Box dropped — active throughout the whole episode.
    if box_z < cfg.box_drop_threshold:
        return "box_drop", base_x

    # Fall (pelvis flipped or head too low). Uses the same gravity sensor as
    # env_catra._get_termination: it reads +1 in z when upright, so < 0 means flipped.
    if float(env.get_gravity("pelvis")[2]) < 0.0 or float(info["head_pos"][2]) < 0.7:
        return "fall", base_x

    # Obstacle SDF penetration — active only after a short settle window (step >= 50),
    # matching env_catra._get_termination. Split into robot-body collision and
    # carried-box collision (boxdf). Only G1CaTra (warm-start box transport) traverses
    # obstacles; obstacle-less tasks (e.g. G1Pickup) lack `warmstart_states_path` and
    # skip this check.
    is_catra = hasattr(cfg, "warmstart_states_path")
    if is_catra and step >= 50:
        thr = cfg.term_collision_threshold
        robot_keys = ("headdf", "pelvdf", "torsdf", "feetdf", "handsdf", "kneesdf", "shldsdf")
        if any(np.any(info[k] < -thr) for k in robot_keys):
            return "robot_collision", base_x
        if np.any(info["boxdf"] < -thr):
            return "box_collision", base_x

    if step >= max_steps:
        return "timeout", base_x

    return None, base_x


def play(args: Args):
    np.random.seed(args.seed)

    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    if args.use_ckpt_config:
        _apply_ckpt_config(env_cfg, args.exp_name)
    env_cfg.pf_config.path = args.obs_path
    if args.warmstart_path is not None:
        env_cfg.warmstart_states_path = args.warmstart_path
    _set_box_noise(env_cfg, args.box_noise)

    env_class = cat_ppo.registry.get(args.task, "play_env_class")
    env = env_class(task_type=env_cfg.task_type, config=env_cfg, headless=not args.render)
    env.pri = args.pri

    max_steps = args.max_steps if args.max_steps is not None else int(env_cfg.episode_length)

    two_agent = _is_two_agent(args.task)
    if two_agent:
        ckpt_path = None
        lower = args.onnx_path_lower
        upper = args.onnx_path_upper
        if lower is None or upper is None:
            ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
            lower = lower or ckpt_path / "policy_lower.onnx"
            upper = upper or ckpt_path / "policy_upper.onnx"
        policy = _Policy.load([lower, upper])
    else:
        onnx_path = args.onnx_path
        if onnx_path is None:
            onnx_path = cat_ppo.get_latest_ckpt(args.exp_name) / "policy.onnx"
        policy = _Policy.load([onnx_path])

    # Pre-select the warm-start index for every episode from a DEDICATED rng, independent of
    # the global np.random stream consumed by per-step obs noise. This guarantees the same
    # set of initial states across runs with the same --seed, regardless of the policy being
    # evaluated or flags like --box-noise (which change how much global RNG each rollout uses).
    if getattr(env, "_ws_qpos", None) is not None and args.warmstart_idx < 0:
        n_ws = env._ws_qpos.shape[0]
        episode_indices = np.random.default_rng(args.seed).integers(0, n_ws, size=args.num_episodes)
        print(f"[mj_onnx_test] pre-selected {args.num_episodes} warm-start indices "
              f"(seed={args.seed}, N={n_ws}): first 8 = {episode_indices[:8].tolist()}")
    else:
        episode_indices = np.full(args.num_episodes, args.warmstart_idx, dtype=int)

    list_succ = []
    list_completed = []
    reasons = Counter()

    for ep in tqdm.tqdm(range(args.num_episodes), desc="Evaluating"):
        # Re-seed the global rng per episode so any init-pose perturbation and obs noise are
        # reproducible run-to-run and independent of the previous episode's trajectory length.
        np.random.seed(args.seed + ep)
        state = env.reset(
            warmstart_idx=int(episode_indices[ep]),
            pos_offset=args.pos_offset,
            ang_offset_deg=args.ang_offset_deg,
        )
        status, base_x = None, float(env.mj_data.qpos[0])
        for _ in range(max_steps + 1):
            obs = state.obs["state"].reshape(1, -1).astype(np.float32)
            action = policy.act(obs)
            if action.shape[0] != env.action_size:
                raise ValueError(
                    f"policy produced {action.shape[0]} action dims but env expects "
                    f"{env.action_size} (two_agent={two_agent}). Check the ONNX file(s) / task."
                )
            state = env.step(state, action)
            status, base_x = _episode_status(env, state, env_cfg, args.goal_x, max_steps)
            if status is not None:
                break

        succ = int(status == "success")
        list_succ.append(succ)
        list_completed.append(min(base_x, args.goal_x) / args.goal_x)
        reasons[status or "timeout"] += 1

    n = len(list_succ)
    print("\n==================== CaTra evaluation ====================")
    print(f"task={args.task}  mode={'two-agent' if two_agent else 'single-agent'}  "
          f"obs_path={args.obs_path}  pri={args.pri}  episodes={n}")
    print(f"success rate : {np.mean(list_succ):.3f}  ({sum(list_succ)}/{n})")
    print(f"mean completion (x/goal): {np.mean(list_completed):.3f}")
    print("outcome breakdown:")
    for reason, count in reasons.most_common():
        print(f"  {reason:10s}: {count:3d}  ({count / n:.2%})")
    print("=========================================================")


if __name__ == "__main__":
    play(tyro.cli(Args))
