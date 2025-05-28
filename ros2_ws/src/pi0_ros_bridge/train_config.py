from pathlib import Path
from openpi.models import pi0
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import TrainConfig, AssetsConfig, DataConfig
from .data_config import LeRobotUR5DataConfig 

def get_custom_config(exp_name: str = "default") -> TrainConfig:
    return TrainConfig(
        name="pi0_ur5",
        exp_name=exp_name,
        model=pi0.Pi0Config(),

        data=LeRobotUR5DataConfig(
            repo_id="your_username/ur5_dataset",
            assets=AssetsConfig(
                assets_dir=str(Path("checkpoints/pi0_base/assets").resolve()),
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(Path("checkpoints/pi0_base/params").resolve())
        ),

        checkpoint_base_dir=str(Path("checkpoints").resolve()),  
        num_train_steps=30_000,
    )
