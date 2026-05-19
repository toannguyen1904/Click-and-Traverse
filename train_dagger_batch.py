import os
import shlex
import subprocess
import sys
from datetime import datetime


python_executable = sys.executable

# Shared teacher config for all tasks below.
# Use run names under data/logs/$WANDB_PROJECT. The latest checkpoint is selected
# automatically, and each teacher's checkpoints/config.json provides its scene.
teacher_restore_names = [
    "04171800_G1Cat_0417testV0_xT0p0xempty",
    "05130748_G1Cat_V0_xG1p0xL1p0xT0p0xside-hurdle2",
    "05130748_G1Cat_V0_xG1p0xL1p0xT0p0xside-hurdle3",
    "05130748_G1Cat_V0_xG1p0xL1p0xT0p0xside-hurdle4",
    "05141019_G1Cat_V0_xG1p0xT0p0xhurdle0",
    "05141019_G1Cat_V0_xG1p0xT0p0xhurdle1",
    "05141019_G1Cat_V0_xG1p0xT0p0xhurdle2",
    "05130748_G1Cat_V0_xO1p0xT0p0xcrouch0",
    "05130748_G1Cat_V0_xO1p0xT0p0xcrouch1",
    "05141019_G1Cat_V0_xG1p0xL1p0xO1p0xT0p0xside-crouch0",
    "05130748_G1Cat_V0_xG1p0xL1p0xO1p0xT0p0xside-hurdle-crouch3",
]

tasks = [
    # cuda_devices, task, exp_name, restore_name, ground, lateral, overhead, obs_path, term_collision_threshold, dagger_timesteps
    # cuda_devices can be 0, "0,1,2,3", or [0, 1, 2, 3].
    ([0,1], "G1Cat", "dagger_v4DG1", "05180809_G1CatDagger_dagger_v3thenrlxG1p0xL1p0xO1p0xT0p0", 1.0, 1.0, 1.0, "", 0.0, 100_000_000),
    ([2,3], "G1Cat", "dagger_v4DG2", "05180809_G1CatDagger_dagger_v3thenrlxG1p0xL1p0xO1p0xT0p0", 1.0, 1.0, 1.0, "", 0.0, 200_000_000),
    ([4,5], "G1Cat", "dagger_v4DG3", "05180809_G1CatDagger_dagger_v3thenrlxG1p0xL1p0xO1p0xT0p0", 1.0, 1.0, 1.0, "", 0.0, 300_000_000),
    ([6,7], "G1Cat", "dagger_v4DG4", "05180809_G1CatDagger_dagger_v3thenrlxG1p0xL1p0xO1p0xT0p0", 1.0, 1.0, 1.0, "", 0.0, 400_000_000),
    # ([0, 1, 2, 3], "G1Cat", "dagger_4gpu", "none", 0.0, 0.0, 0.0, "data/assets/TypiObs/empty", 0.0),
]

processes = []


def _cuda_visible_devices(cuda_devices):
    if isinstance(cuda_devices, (list, tuple)):
        return ",".join(str(device) for device in cuda_devices)
    return str(cuda_devices)


def _safe_log_token(value):
    return _cuda_visible_devices(value).replace(",", "-")


if __name__ == "__main__":
    if not teacher_restore_names:
        raise ValueError("Set teacher_restore_names before running DAgger batch training.")

    output_dir = "./output_logs"
    os.makedirs(output_dir, exist_ok=True)
    process_cmd_map = {}

    for cuda_devices, task, exp_name, restore_name, ground, lateral, overhead, obs_path, term_collision_threshold, dagger_timesteps in tasks:
        cuda_visible_devices = _cuda_visible_devices(cuda_devices)
        cmd = [
            python_executable,
            "-m",
            "train_ppo_dagger",
            "--task",
            task,
            "--restore_name",
            restore_name,
            "--exp_name",
            exp_name,
            "--ground",
            str(ground),
            "--lateral",
            str(lateral),
            "--overhead",
            str(overhead),
            "--term_collision_threshold",
            str(term_collision_threshold),
            "--dagger_timesteps",
            str(dagger_timesteps),
            "--obs_path",
            obs_path,
            "--teacher_restore_names",
            *teacher_restore_names,
        ]
        cmd_display = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} " + " ".join(
            shlex.quote(part) for part in cmd
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cuda_log_token = _safe_log_token(cuda_devices)
        stdout_file = os.path.join(output_dir, f"{timestamp}_{cuda_log_token}_dagger_stdout.log")
        stderr_file = os.path.join(output_dir, f"{timestamp}_{cuda_log_token}_dagger_stderr.log")

        with open(stdout_file, "w") as out_file, open(stderr_file, "w") as err_file:
            print(f"Executing: {cmd_display}")
            out_file.write(f"{cmd_display}\n")
            err_file.write(f"{cmd_display}\n")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            process = subprocess.Popen(cmd, env=env, stdout=out_file, stderr=err_file)
            processes.append(process)
            process_cmd_map[process] = cmd_display

    while processes:
        for process in processes[:]:
            retcode = process.poll()
            if retcode is not None:
                if retcode != 0:
                    cmd = process_cmd_map[process]
                    print(f"\033[91mReturn code {retcode}.\nCommand: {cmd}\033[0m")
                processes.remove(process)

    print("All DAgger tasks completed.")
