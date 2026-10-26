import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torchvision.transforms.v2 as transforms

class MegalithDataset(Dataset):
    def __init__(self, data_dir: Path, is_train: bool = True):
        """
        Loads .npy LRM arrays and applies aggressive geometric augmentations
        to artificially expand the limited positive samples.
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        
        # Load file paths
        self.pos_files = list((self.data_dir / "positive").glob("*.npy"))
        self.neg_files = list((self.data_dir / "negative").glob("*.npy"))
        
        self.all_files = self.pos_files + self.neg_files
        self.labels = [1] * len(self.pos_files) + [0] * len(self.neg_files)
        
        print(f"Dataset Loaded: {len(self.pos_files)} positives, {len(self.neg_files)} negatives.")

        # Aggressive augmentation for training
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Random 90, 180, 270 degree rotations (barrows can face any direction)
                transforms.RandomChoice([
                    transforms.Lambda(lambda x: x),  # 0 deg
                    transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])), # 90 deg
                    transforms.Lambda(lambda x: torch.rot90(x, 2, [1, 2])), # 180 deg
                    transforms.Lambda(lambda x: torch.rot90(x, 3, [1, 2]))  # 270 deg
                ])
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # 1. Load the raw numpy array
        npy_path = self.all_files[idx]
        lrm_array = np.load(npy_path)
        
        # 2. Convert to PyTorch Tensor and add channel dimension (C, H, W) -> (1, 512, 512)
        tensor = torch.from_numpy(lrm_array).float().unsqueeze(0)
        
        # 3. Apply geometric augmentations
        if self.transform:
            tensor = self.transform(tensor)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return tensor, label

if __name__ == "__main__":
    # Quick test to verify it works
    test_dir = Path(__file__).resolve().parent.parent / "data" / "dataset" / "train"
    
    # Create dummy folders so the script doesn't crash if your dad hasn't finished
    (test_dir / "positive").mkdir(parents=True, exist_ok=True)
    (test_dir / "negative").mkdir(parents=True, exist_ok=True)
    
    dataset = MegalithDataset(test_dir, is_train=True)