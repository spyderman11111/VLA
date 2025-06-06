from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lerobot/smolvla_base",
    revision="main",  
    local_dir="/home/ipk/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/smolvla_clean",
    local_dir_use_symlinks=False
)
