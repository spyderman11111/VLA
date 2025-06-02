import dataclasses
import numpy as np
from openpi import transforms

def _parse_image(img):
    return img.astype(np.uint8)

@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    action_dim: int
    model_type: str = "pi0"

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["joints"], data["gripper"]])
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        return {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            },
            "prompt": data.get("prompt", "pick the object")
        }
