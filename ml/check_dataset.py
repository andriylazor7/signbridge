from pathlib import Path
from data.dataset import WLASLDataset

ds = WLASLDataset(Path("data/landmarks"), Path("data/wlasl100_split.json"), split="train")
x, mask, y = ds[0]
print("landmarks:", x.shape)   
print("mask:", mask.shape)     
print("label:", y.item())      
print("Кількість прикладів у train:", len(ds))