import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class UR5Outputs:
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}