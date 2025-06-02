import dataclasses
import pathlib
from openpi import transforms as _transforms
from openpi import model as _model
from openpi.transforms import ModelTransformFactory, DataConfig, DataConfigFactory
from .input_adapter import UR5Inputs
from .output_adapter import UR5Outputs


@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(inputs=[
            _transforms.RepackTransform({
                "base_rgb": "image",
                "wrist_rgb": "wrist_image",
                "joints": "joints",
                "gripper": "gripper",
                "prompt": "prompt",
            })
        ])
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR5Outputs()]
        )
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)]
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
