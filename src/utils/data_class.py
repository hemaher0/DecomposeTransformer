from torch.utils.data import Dataset
import torch
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
        embedding = self.data["embeddings"][index]
        attention_mask = self.data["attention_mask"][index]
        if isinstance(embedding, torch.Tensor):
            if embedding.dim() == 3:
                embedding = torch.squeeze(embedding)
        elif isinstance(embedding, np.ndarray):
            if embedding.ndim == 3:
                embedding = np.squeeze(embedding)

        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.dim() == 2:
                attention_mask = torch.squeeze(attention_mask)
        elif isinstance(attention_mask, np.ndarray):
            if attention_mask.ndim == 2:
                attention_mask = np.squeeze(attention_mask)
        return {
            "embeddings": embedding,
            "labels": self.data["labels"][index],
            "attention_mask": attention_mask,
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
