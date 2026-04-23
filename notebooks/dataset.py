import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    def __init__(self, path, image_size=256, split="train"):
        self.path       = path
        self.image_size = image_size
        self.split      = split

        self.files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {path}")

        print(f" {split} dataset loaded: {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.files))

        img    = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        img_np = np.array(img) / 255.0
        lab    = rgb2lab(img_np).astype("float32")

        L  = (lab[:, :, 0] / 50.0) - 1.0
        AB = lab[:, :, 1:] / 128.0

        L  = torch.tensor(L).unsqueeze(0)
        AB = torch.tensor(AB).permute(2, 0, 1)

        return L, AB

    def __repr__(self):
        return f"ColorizationDataset(split={self.split}, size={len(self.files)})"