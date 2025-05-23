from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np

# Load model configuration and checkpoint
cfg = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")
policy = policy_config.create_trained_policy(cfg, checkpoint_dir)

# Construct a minimal example input for inference
dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

example = {
    "observation/joint_position": [0.0] * 6,
    "observation/gripper_position": [0.0],
    "observation/exterior_image_1_left": dummy_image,
    "observation/wrist_image_left": dummy_image,
    "prompt": "pick up the fork"
}

# Run inference
action_chunk = policy.infer(example)["actions"]
print("Inference successful. Output actions:")
print(action_chunk)
