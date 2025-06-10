import torch

model = torch.load("/mnt/HDD/shuo/VLA/berkeley_ur5_dataset_stats.pt")
print(type(model))
print(model)