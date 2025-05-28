from openpi.shared import download

checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base")

print("checkpoint_path：", checkpoint_dir)
