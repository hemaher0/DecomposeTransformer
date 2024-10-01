from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class CustomTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        return {
            "input_ids": self.data["input_ids"][index],
            "labels": self.data["labels"][index],
            "attention_mask": self.data.get("attention_mask", [None])[index],
        }


class CustomEmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["embeddings"])

    def __getitem__(self, index):
        return {
            "embeddings": self.data["embeddings"][index],
            "labels": self.data["labels"][index],
            "attention_mask": self.data.get("attention_mask", [None])[index],
        }


class CustomCodeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        return {
            "input_ids": self.data["input_ids"][index],
            "summaries": self.data["summaries"][index],
            "attention_mask": self.data.get("attention_mask", [None])[index],
        }


class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data["image"])

    def __getitem__(self, index):
        image = self.data["image"][index]
        labels = self.data["labels"][index]

        if not isinstance(image, np.ndarray):
            image = Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "labels": labels}
