import os
# Must run before any transitive `import mujoco` — mujoco.gl_context reads
# MUJOCO_GL at module load time and binds GLContext permanently.
os.environ.setdefault("MUJOCO_GL", "egl")

from cat_ppo.utils import registry
from cat_ppo.utils.logger import LOGGER, update_file_handler
from cat_ppo.constant import get_path_log, get_latest_ckpt

import cat_ppo.envs