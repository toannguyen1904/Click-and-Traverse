import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=true"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass

import numpy as np
import onnxruntime as rt
import tyro

import cat_ppo
import tqdm


@dataclass
class Args:
    task: str
    exp_name: str = None
    seed: int = 42
    onnx_path: str = None
    pri: bool = False
    obs_path: str = 'data/assets/TypiObs/empty'


@dataclass
class State:
    info: dict
    obs: dict
    

def play(args: Args):
    env_class = cat_ppo.registry.get(args.task, "play_env_class")
    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    env_cfg.pf_config.path = args.obs_path
    env = env_class(task_type=env_cfg.task_type, config=env_cfg)
    env.pri = args.pri
    if args.onnx_path is not None:
        onnx_path = args.onnx_path
    else:
        ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
        onnx_path = ckpt_path / "policy.onnx"
    output_names = ["continuous_actions"]
    policy = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    state = env.reset()
    _ctr = 0
    # from plot import ActionPlotter
    #
    # plotter = ActionPlotter(num_dim=5)

    list_succ = []
    list_completed = []
    # while num_try < 42:
    for _ in tqdm.tqdm(range(10000), desc="Playing"):
        obs = state.obs["state"].reshape(1, -1).astype(np.float32)
        
        onnx_input = {"obs": obs}
        action = policy.run(output_names, onnx_input)[0]
        
        action = action[0]
        state = env.step(state, action)
        # if state.info['step'] >= 250:
        # print(env.mj_data.qpos[0])
        if env.mj_data.qpos[0] >= 1.6:
            list_completed.append(min(env.mj_data.qpos[0],1.6)/1.6)
            list_succ.append(1)
            state = env.reset()
        elif state.info['step'] >= 300:
            list_completed.append(min(env.mj_data.qpos[0],1.6)/1.6)
            list_succ.append(0)
            state = env.reset()
        elif (not state.info['succ']):
            list_completed.append(min(env.mj_data.qpos[0],1.6)/1.6)
            list_succ.append(0)
            state = env.reset()
        _ctr += 1


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)
