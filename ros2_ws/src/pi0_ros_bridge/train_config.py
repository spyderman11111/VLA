from openpi import pi0, weight_loaders
from openpi.train_config import TrainConfig, AssetsConfig, DataConfig
from .data_config import LeRobotUR5DataConfig

train_config = TrainConfig(
    name="pi0_ur5",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_dataset",
        assets=AssetsConfig(
            assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(
            local_files_only=True,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
