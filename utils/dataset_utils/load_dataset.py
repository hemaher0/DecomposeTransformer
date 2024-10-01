import os.path as path
import torch
import random
import numpy as np
import pickle
import torch.utils.data as data_utils
from datasets import load_dataset, DatasetDict
from utils.helper import DataConfig, ModelConfig, color_print, Paths
from utils.dataset_utils.dataset import (
    CustomEmbeddingDataset,
    CustomImageDataset,
    CustomTextDataset,
    CustomCodeDataset,
)
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def tokenize_dataset(raw_dataset: DatasetDict, data_config: DataConfig):
    tokenized_datasets = {field: [] for field in data_config.return_fields}
    model_config = data_config.model_config
    if model_config.model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_name, cache_dir=model_config.cache_dir
        )
    for example in tqdm(raw_dataset, desc="Processing dataset", ascii=True):
        if data_config.task_type == "text_generation":
            tokens = tokenizer(
                example[data_config.first_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            tokenized_datasets["input_ids"].append(tokens["input_ids"][0])
            tokenized_datasets["attention_mask"].append(tokens["attention_mask"][0])
            targets = tokenizer(
                example[data_config.second_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            tokenized_datasets["summaries"].append(targets["input_ids"][0])
        elif data_config.task_type == "text_classification":
            tokens = tokenizer(
                example[data_config.first_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            tokenized_datasets["input_ids"].append(tokens["input_ids"][0])
            tokenized_datasets["attention_mask"].append(tokens["attention_mask"][0])
            tokenized_datasets["labels"].append(example[data_config.second_column])
        elif data_config.task_type == "image_classification":
            image = example[data_config.first_column]
            image = image.resize((28, 28))
            image = np.array(image)
            tokenized_datasets["image"].append(image)
            tokenized_datasets["labels"].append(example[data_config.second_column])

    if data_config.task_type == "text_classification":
        return CustomTextDataset(tokenized_datasets)
    elif data_config.task_type == "text_generation":
        return CustomTextDataset(tokenized_datasets)
    elif data_config.task_type == "image_classification":
        return CustomImageDataset(tokenized_datasets)


def load_dataloader(
    dataset: DatasetDict,
    data_config: DataConfig,
    shuffle: bool = False,
    is_valid: bool = False,
):
    """
    Define load datasets for pretrained.
    """
    if is_valid:
        tokenized_dataset = tokenize_dataset(dataset, data_config)
        valid_size = int(len(tokenized_dataset) * data_config.valid_size)
        train_size = len(tokenized_dataset) - valid_size
        train_dataset, valid_dataset = data_utils.random_split(
            tokenized_dataset, [train_size, valid_size]
        )

        train_dataloader = data_utils.DataLoader(
            train_dataset,
            batch_size=data_config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
        )
        valid_dataloader = data_utils.DataLoader(
            valid_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
        )
        return train_dataloader, valid_dataloader
    else:
        tokenized_dataset = tokenize_dataset(dataset, data_config)
        dataloader = data_utils.DataLoader(
            tokenized_dataset,
            batch_size=data_config.batch_size,
            shuffle=shuffle,
            num_workers=data_config.num_workers,
        )
        return dataloader


def load_cached_dataset(data_config: DataConfig):
    cached_dataset_path = data_config.cache_dir

    if (
        not data_config.is_cached() or not data_config.do_cache
    ):  # If not cached, generate caches
        color_print(f"Downloading the Dataset {data_config.dataset_name}")
        dataset = load_dataset(**data_config.dataset_args)

        torch.manual_seed(data_config.seed)
        np.random.seed(data_config.seed)
        random.seed(data_config.seed)

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        if data_config.dataset_name == "YahooAnswersTopics":
            train_size = len(train_dataset)
            test_size = len(test_dataset)

            train_size //= 20
            test_size //= 2
            train_dataset, _ = data_utils.random_split(
                train_dataset,
                [train_size, len(train_dataset) - train_size],
                generator=torch.Generator().manual_seed(data_config.seed),
            )
            test_dataset, _ = data_utils.random_split(
                test_dataset,
                [test_size, len(test_dataset) - test_size],
                generator=torch.Generator().manual_seed(data_config.seed),
            )

        if "validation" in dataset:
            valid_dataset = dataset["validation"]
            train_dataloader = load_dataloader(train_dataset, data_config, shuffle=True)
            valid_dataloader = load_dataloader(valid_dataset, data_config)
        elif "valid" in dataset:
            valid_dataset = dataset["valid"]
            train_dataloader = load_dataloader(train_dataset, data_config, shuffle=True)
            valid_dataloader = load_dataloader(valid_dataset, data_config)
        else:
            train_dataloader, valid_dataloader = load_dataloader(
                train_dataset, data_config, is_valid=True
            )

        test_dataloader = load_dataloader(test_dataset, data_config)

        if data_config.do_cache:
            Paths.get_dir(cached_dataset_path)
            save_cache(train_dataloader, cached_dataset_path, "train.pkl")
            save_cache(valid_dataloader, cached_dataset_path, "valid.pkl")
            save_cache(test_dataloader, cached_dataset_path, "test.pkl")
            color_print("Caching is completed.")
    else:
        color_print(f"Loading cached dataset {data_config.dataset_name}.")
        train_dataloader = load_from_cache(cached_dataset_path, "train.pkl")
        valid_dataloader = load_from_cache(cached_dataset_path, "valid.pkl")
        test_dataloader = load_from_cache(cached_dataset_path, "test.pkl")
    color_print(f"The dataset {data_config.dataset_name} is loaded")
    data_config.summary()
    return train_dataloader, valid_dataloader, test_dataloader


def load_data(
    model_config: ModelConfig,
    batch_size: int = 32,
    valid_size: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 44,
    do_cache: bool = True,
    streaming: bool = False,
):
    data_config = DataConfig(
        model_config=model_config,
        max_length=512,
        batch_size=batch_size,
        valid_size=valid_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        do_cache=do_cache,
        streaming=streaming,
    )
    return load_cached_dataset(data_config)


def save_cache(data, cache_dir, filename):
    with open(path.join(cache_dir, filename), "wb") as f:
        pickle.dump(data, f)
    print(f"{filename} is cached.")


def load_from_cache(cache_dir, filename):
    cache_path = path.join(cache_dir, filename)
    if path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"{filename} is loaded from cache.")
        return data
    else:
        raise FileNotFoundError(f"{filename} not found in {cache_dir}")


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
    [
        data_utils.DataLoader,
        data_utils.Subset,
        data_utils.RandomSampler,
        data_utils.BatchSampler,
        data_utils.SequentialSampler,
        data_utils.default_collate,
        CustomTextDataset,
        CustomImageDataset,
        CustomEmbeddingDataset,
        CustomCodeDataset,
        DataConfig,
    ]
)
