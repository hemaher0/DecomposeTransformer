import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Dict, Generator
import random


class SamplingDataset(IterableDataset):
    def __init__(
            self,
            dataloader: DataLoader,
            target_class: int,
            num_samples: int,
            num_class: int,
            positive_sample: bool = True,
            batch_size: int = 4,
            device: torch.device = torch.device("cuda:0"),
            seed: int = 42,
            resample: bool = True,
    ) -> None:
        if num_samples % batch_size != 0:
            raise ValueError("num_samples must be divisible by batch_size")

        self.dataloader = dataloader
        self.target_class = target_class
        self.num_samples = num_samples
        self.num_class = num_class
        self.batch_size = batch_size
        self.positive_sample = positive_sample
        self.device = device
        self.resample = resample

        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.sampled_data = None

    def _sample_data(self) -> None:
        sampled_ids = []
        sampled_masks = []
        sampled_labels = []

        class_sample_counts = {}
        if self.positive_sample:
            class_sample_counts[self.target_class] = self.num_samples
        else:
            available_classes = [
                i for i in range(self.num_class) if i != self.target_class
            ]
            samples_per_class = self.num_samples // len(available_classes)
            remainder = self.num_samples % len(available_classes)

            for class_id in available_classes:
                class_sample_counts[class_id] = samples_per_class + (
                    1 if remainder > 0 else 0
                )
                remainder -= 1

        total_sampled = {class_id: 0 for class_id in class_sample_counts}

        # Loop through batches and collect data
        for batch in self.dataloader:
            b_input_ids = batch["input_ids"]
            b_attention_mask = batch["attention_mask"]
            b_labels = batch["labels"]

            for class_id, target_count in class_sample_counts.items():
                if total_sampled[class_id] >= target_count:
                    continue

                mask = b_labels == class_id
                selected_input_ids = b_input_ids[mask]
                selected_attention_mask = b_attention_mask[mask]
                selected_labels = b_labels[mask]

                num_selected = selected_labels.size(0)
                if num_selected == 0:
                    continue

                remaining_samples = target_count - total_sampled[class_id]
                num_selected = min(num_selected, remaining_samples)

                selected_input_ids = selected_input_ids[:num_selected]
                selected_attention_mask = selected_attention_mask[:num_selected]
                selected_labels = selected_labels[:num_selected]

                total_sampled[class_id] += num_selected

                if num_selected > 0:
                    sampled_ids.append(selected_input_ids)
                    sampled_masks.append(selected_attention_mask)
                    sampled_labels.append(selected_labels)

        # Concatenate all collected data outside the loop
        if sampled_ids:
            sampled_ids = torch.cat(sampled_ids)
            sampled_masks = torch.cat(sampled_masks)
            sampled_labels = torch.cat(sampled_labels)

        self.sampled_data = []
        num_batches = len(sampled_ids) // self.batch_size
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size

            batch_input_ids = sampled_ids[start_idx:end_idx]
            batch_masks = sampled_masks[start_idx:end_idx]
            batch_labels = sampled_labels[start_idx:end_idx]

            self.sampled_data.append({
                "input_ids": batch_input_ids,
                "attention_mask": batch_masks,
                "labels": batch_labels,
            })

        if len(sampled_ids) % self.batch_size != 0:
            remaining_batch = {
                "input_ids": sampled_ids[num_batches * self.batch_size:],
                "attention_mask": sampled_masks[num_batches * self.batch_size:],
                "labels": sampled_labels[num_batches * self.batch_size:],
            }
            self.sampled_data.append(remaining_batch)

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        if self.sampled_data is None or self.resample:
            self._sample_data()
        for batch in self.sampled_data:
            yield batch
