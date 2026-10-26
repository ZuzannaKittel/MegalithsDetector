import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torchvision.transforms.v2 as transforms

class MegalithDataset(Dataset):
    def __init__(self, data_dir: Path, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        
        self.pos_files = list((self.data_dir / "positive").glob("*.npy"))
        self.neg_files = list((self.data_dir / "negative").glob("*.npy"))
        
        # Oversample real positive files if synthetic files exist
        real_pos = [f for f in self.pos_files if "synth" not in f.name]
        synth_pos = [f for f in self.pos_files if "synth" in f.name]
        
        # Multiply real samples 5x so the model prioritizes real barrows
        self.pos_files = (real_pos * 5) + synth_pos
        
        self.all_files = self.pos_files + self.neg_files
        self.labels = [1] * len(self.pos_files) + [0] * len(self.neg_files)
        
        print(f"Dataset Loaded: {len(self.pos_files)} positives ({len(real_pos)} real x5 + {len(synth_pos)} synth), {len(self.neg_files)} negatives.")

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomChoice([
                    transforms.Lambda(lambda x: x),
                    transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
                    transforms.Lambda(lambda x: torch.rot90(x, 2, [1, 2])),
                    transforms.Lambda(lambda x: torch.rot90(x, 3, [1, 2]))
                ])
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        npy_path = self.all_files[idx]
        lrm_array = np.load(npy_path)
        
        # Z-score Normalization (Zero mean, unit variance)
        std = np.std(lrm_array)
        if std > 1e-5:
            lrm_array = (lrm_array - np.mean(lrm_array)) / std
        else:
            lrm_array = lrm_array - np.mean(lrm_array)
            
        tensor = torch.from_numpy(lrm_array).float().unsqueeze(0)
        
        if self.transform:
            tensor = self.transform(tensor)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label