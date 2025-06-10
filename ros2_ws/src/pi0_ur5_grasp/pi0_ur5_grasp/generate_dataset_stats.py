import torch
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    
    repo_id = "lerobot/berkeley_autolab_ur5"

    dataset = LeRobotDataset(repo_id, delta_timestamps={
        "observation.state": [0],
        "action": [0],
    })

    print(f"Dataset frames: {len(dataset)}")

    all_states = []
    all_actions = []

    for sample in tqdm(dataset, desc="Collecting observation.state and action"):
        state = sample["observation.state"][0]  # shape: (state_dim,)
        action = sample["action"][0]            # shape: (action_dim,)
        all_states.append(state)
        all_actions.append(action)

    states_tensor = torch.stack(all_states)   # (N, state_dim)
    actions_tensor = torch.stack(all_actions) # (N, action_dim)

    dataset_stats = {
        "observation.state": {
            "mean": states_tensor.mean(dim=0),
            "std": states_tensor.std(dim=0).clamp(min=1e-8)
        },
        "action": {
            "mean": actions_tensor.mean(dim=0),
            "std": actions_tensor.std(dim=0).clamp(min=1e-8)
        }
    }

    torch.save(dataset_stats, "berkeley_ur5_dataset_stats.pt")
    print("Saved dataset_stats to berkeley_ur5_dataset_stats.pt")

if __name__ == "__main__":
    main()
