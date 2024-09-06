from os.path import join
import torch
import random
import numpy as np
import torch.utils.data as data_utils
from datasets import load_dataset
from utils.helper import DataConfig, ModelConfig, color_print, Paths
from transformers import AutoTokenizer
from tqdm.auto import tqdm


class CustomDataset(data_utils.Dataset):
    def __init__(self, data, data_config):
        self.data = data
        self.data_config = data_config

    def __len__(self):
        return len(self.data[self.data_config.return_fields[0]])

    def __getitem__(self, index):
        return {key: self.data[key][index] for key in self.data_config.return_fields}


class Embedding(data_utils.Dataset):
    def __init__(self, embeddings, labels=None, attention_mask=None):
        self.embeddings = embeddings
        self.labels = labels
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return {"embeddings": self.embeddings[index],
                "labels": self.labels[index],
                "attention_mask": self.attention_mask[index]}


def tokenize_dataset(raw_dataset, tokenizer, data_config):
    tokenized_datasets = {field: [] for field in data_config.return_fields}
    for example in tqdm(raw_dataset, desc="Tokenizing dataset", ascii=True):
        if data_config.task_type == "seq2seq":
            inputs = tokenizer(
                example[data_config.text_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            targets = tokenizer(
                example[data_config.label_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            tokenized_datasets["input_ids"].append(inputs["input_ids"][0])
            tokenized_datasets["attention_mask"].append(inputs["attention_mask"][0])
            tokenized_datasets["labels"].append(targets["input_ids"][0])
        else:
            tokens = tokenizer(
                example[data_config.text_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            tokenized_datasets["input_ids"].append(tokens["input_ids"][0])
            tokenized_datasets["attention_mask"].append(tokens["attention_mask"][0])
            tokenized_datasets["labels"].append(example[data_config.label_column])
    return CustomDataset(tokenized_datasets, data_config)


# In[]: Define load datasets for pretrained
def load_dataloader(dataset, tokenizer, data_config, shuffle=False, is_valid=False):
    if is_valid:
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, data_config)
        valid_size = int(len(tokenized_dataset) * data_config.valid_size)
        train_size = len(tokenized_dataset) - valid_size
        train_dataset, valid_dataset = data_utils.random_split(
            tokenized_dataset, [train_size, valid_size]
        )

        train_dataloader = data_utils.DataLoader(
            train_dataset, batch_size=data_config.batch_size, shuffle=True, num_workers=data_config.num_workers
        )
        valid_dataloader = data_utils.DataLoader(
            valid_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.num_workers
        )
        return train_dataloader, valid_dataloader
    else:
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, data_config)
        dataloader = data_utils.DataLoader(
            tokenized_dataset, batch_size=data_config.batch_size, shuffle=shuffle, num_workers=data_config.num_workers
        )
        return dataloader


def load_cached_dataset(data_config):
    model_config = ModelConfig(data_config.dataset_name, data_config.device)

    torch.manual_seed(data_config.seed)
    np.random.seed(data_config.seed)
    random.seed(data_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, cache_dir=model_config.cache_dir)
    cached_dataset_path = data_config.cache_dir

    if (
            not data_config.is_cached() or not data_config.do_cache
    ):  # If not cached, generate caches
        color_print(f"Downloading the Dataset {data_config.dataset_name}")
        dataset = load_dataset(**data_config.dataset_args)

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        if data_config.dataset_name == "YahooAnswersTopics":
            train_size = len(train_dataset)
            test_size = len(test_dataset)

            train_size //= 20
            test_size //= 2
            train_dataset, _ = data_utils.random_split(train_dataset, [train_size, len(train_dataset) - train_size],
                                                       generator=torch.Generator().manual_seed(data_config.seed))
            test_dataset, _ = data_utils.random_split(test_dataset, [test_size, len(test_dataset) - test_size],
                                                      generator=torch.Generator().manual_seed(data_config.seed))

        if "validation" in dataset:
            valid_dataset = dataset["validation"]
            train_dataloader = load_dataloader(
                train_dataset, tokenizer, data_config, shuffle=True
            )
            valid_dataloader = load_dataloader(valid_dataset, tokenizer, data_config)
        elif "valid" in dataset:
            valid_dataset = dataset["valid"]
            train_dataloader = load_dataloader(
                train_dataset, tokenizer, data_config, shuffle=True
            )
            valid_dataloader = load_dataloader(valid_dataset, tokenizer, data_config)
        else:
            train_dataloader, valid_dataloader = load_dataloader(
                train_dataset, tokenizer, data_config, is_valid=True
            )

        test_dataloader = load_dataloader(test_dataset, tokenizer, data_config)
        if data_config.do_cache:
            Paths.get_dir(cached_dataset_path)
            torch.save(train_dataloader, join(cached_dataset_path, "train.pt"))
            torch.save(valid_dataloader, join(cached_dataset_path, "valid.pt"))
            torch.save(test_dataloader, join(cached_dataset_path, "test.pt"))
            color_print("Caching is completed.")
    else:
        color_print(f"Loading cached dataset {data_config.dataset_name}.")
        train_dataloader = torch.load(join(cached_dataset_path, "train.pt"), weights_only=True)
        valid_dataloader = torch.load(join(cached_dataset_path, "valid.pt"), weights_only=True)
        test_dataloader = torch.load(join(cached_dataset_path, "test.pt"), weights_only=True)
    color_print(f"The dataset {data_config.dataset_name} is loaded")
    data_config.summary()
    return train_dataloader, valid_dataloader, test_dataloader


def load_data(dataset_name, batch_size=32, valid_size=0.1, num_workers=4, pin_memory=True, seed=44, do_cache=True):
    data_config = DataConfig(
        dataset_name=dataset_name,
        max_length=512,
        batch_size=batch_size,
        valid_size=valid_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        do_cache=do_cache,
    )
    return load_cached_dataset(data_config)


# def convert_dataset_labels_to_binary(dataloader, target_class, is_stratified=False):
#     input_ids, attention_mask, labels = [], [], []
#     for batch in dataloader:
#         input_ids.append(batch["input_ids"])
#         attention_mask.append(batch["attention_mask"])
#
#         binary_labels = (batch["labels"] == target_class).long()
#         labels.append(binary_labels)
#
#     input_ids = torch.cat(input_ids)
#     attention_mask = torch.cat(attention_mask)
#     labels = torch.cat(labels)
#
#     if is_stratified:
#         # Count the number of samples for each class
#         class_0_indices = [i for i, label in enumerate(labels) if label == 0]
#         class_1_indices = [i for i, label in enumerate(labels) if label == 1]
#
#         # Find the minimum class size
#         min_class_size = min(len(class_0_indices), len(class_1_indices))
#
#         # Convert to tensors and shuffle
#         class_0_indices = torch.tensor(class_0_indices)
#         class_1_indices = torch.tensor(class_1_indices)
#
#         class_0_indices = class_0_indices[
#             torch.randperm(len(class_0_indices))[:min_class_size]
#         ]
#         class_1_indices = class_1_indices[
#             torch.randperm(len(class_1_indices))[:min_class_size]
#         ]
#
#         # Combine indices and shuffle them
#         balanced_indices = torch.cat([class_0_indices, class_1_indices]).long()
#         balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]
#
#         # Subset the data to the balanced indices
#         input_ids = input_ids[balanced_indices]
#         attention_mask = attention_mask[balanced_indices]
#         labels = labels[balanced_indices]
#
#     transformed_dataset = CustomDataset(input_ids, attention_mask, labels)
#     transformed_dataloader = DataLoader(
#         transformed_dataset, batch_size=dataloader.batch_size
#     )
#
#     return transformed_dataloader


# def extract_and_convert_dataloader(dataloader, true_index, false_index):
#     # Extract the data using the provided indices
#
#     input_ids, attention_mask, labels = [], [], []
#
#     for batch in dataloader:
#         mask = (batch["labels"] == true_index) | (batch["labels"] == false_index)
#         if mask.any():
#             input_ids.append(batch["input_ids"][mask])
#             attention_mask.append(batch["attention_mask"][mask])
#             labels.append(batch["labels"][mask])
#
#     input_ids = torch.cat(input_ids, dim=0)
#     attention_mask = torch.cat(attention_mask, dim=0)
#     labels = torch.cat(labels, dim=0)
#
#     subset_dataset = CustomDataset(input_ids, attention_mask, labels)
#     subset_dataloader = DataLoader(subset_dataset, batch_size=dataloader.batch_size)
#
#     # Apply convert_dataset_labels_to_binary
#     binary_dataloader = convert_dataset_labels_to_binary(subset_dataloader, true_index)
#
#     return binary_dataloader


torch.serialization.add_safe_globals(
    [data_utils.DataLoader, data_utils.Subset, data_utils.RandomSampler, data_utils.BatchSampler,
     data_utils.SequentialSampler, data_utils.default_collate, CustomDataset, DataConfig])
