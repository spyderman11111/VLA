from openpi.training import config as _config
from openpi.shared import download
from openpi.models import model as _model
import jax
import dataclasses

def run_loss_eval():
    config = _config.get_config("pi0_aloha_sim")

    # 可选：减少 batch size
    config = dataclasses.replace(config, batch_size=2)

    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    key = jax.random.key(0)

    model = config.model.load(_model.restore_params(checkpoint_dir / "params"))

    # 构造假的 obs, act
    obs, act = config.model.fake_obs(), config.model.fake_act()
    loss = model.compute_loss(key, obs, act)

    print("Loss shape:", loss.shape)

    del model

if __name__ == "__main__":
    run_loss_eval()
