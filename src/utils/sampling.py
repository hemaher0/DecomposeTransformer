import torch
from typing import Dict, Generator
from torch.utils.data import IterableDataset, DataLoader
from .helper import Config


class SamplingDataset(IterableDataset):
    def __init__(
        self,
        dataloader: DataLoader,
        config: Config,
        target_class: int,
        num_samples: int,
        positive_sample: bool = True,
        batch_size: int = 4,
        resample: bool = True,
    ) -> None:
        self.dataloader = dataloader
        self.target_class = target_class
        self.num_samples = num_samples
        self.num_class = config.num_labels
        self.positive_sample = positive_sample
        self.resample = resample
        self.batch_size = batch_size
        first_batch = next(iter(dataloader))
        self.keys = first_batch.keys()
        self.is_embeds = "embeddings" in first_batch
        config.init_seed()

        self.sampled_data = None

    def _sample_data(self) -> None:
        sampled_data = {key: [] for key in self.keys}

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

        for batch in self.dataloader:
            labels = batch.get("labels", batch.get("summaries", None))

            if labels is None:
                raise ValueError(
                    "The dataset must contain either 'labels' or 'summaries'."
                )

            for class_id, target_count in class_sample_counts.items():
                if total_sampled[class_id] >= target_count:
                    continue

                mask = labels == class_id

                num_selected = mask.sum().item()

                if num_selected == 0:
                    continue

                remaining_samples = target_count - total_sampled[class_id]
                num_selected = min(num_selected, remaining_samples)

                for key in self.keys:
                    selected_items = batch[key][mask][:num_selected]
                    sampled_data[key].append(selected_items)

                total_sampled[class_id] += num_selected

        for key in self.keys:
            if sampled_data[key]:
                sampled_data[key] = torch.cat(sampled_data[key])

        self.sampled_data = []
        num_batches = len(sampled_data["labels"]) // self.batch_size
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size

            batch_dict = {
                key: sampled_data[key][start_idx:end_idx] for key in self.keys
            }
            self.sampled_data.append(batch_dict)

        if len(sampled_data["labels"]) % self.batch_size != 0:
            remaining_batch = {
                key: sampled_data[key][num_batches * self.batch_size :]
                for key in self.keys
            }
            self.sampled_data.append(remaining_batch)

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        if self.sampled_data is None or self.resample:
            self._sample_data()
        for batch in self.sampled_data:
            yield batch
