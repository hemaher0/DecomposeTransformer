import os
import torch
import warnings
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import (
    random_split,
    DataLoader,
    Subset,
    RandomSampler,
    BatchSampler,
    SequentialSampler,
    default_collate,
)
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from .helper import Config, Paths, color_print, save_cache, load_cache
from .data_class import *

warnings.filterwarnings("ignore", category=FutureWarning)


class DataConfig:
    def __init__(
        self,
        config: Config,
        batch_size: int = 32,
        valid_size: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


def preprocssing_dataset(raw_dataset: DatasetDict, config: Config):
    tokenized_datasets = {field: [] for field in config.return_fields}
    if "text" in config.task_type:
        max_length = config.model_config.max_position_embeddings
        tokenizer = load_tokenizer(config)
    for example in tqdm(raw_dataset, desc="Processing dataset", ascii=True):
        if config.task_type == "text_generation":
            tokens = tokenizer(
                example[config.first_column],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized_datasets["input_ids"].append(tokens["input_ids"][0])
            tokenized_datasets["attention_mask"].append(tokens["attention_mask"][0])
            targets = tokenizer(
                example[config.second_column],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized_datasets["summaries"].append(targets["input_ids"][0])
        elif config.task_type == "text_classification":
            tokens = tokenizer(
                example[config.first_column],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized_datasets["input_ids"].append(tokens["input_ids"][0])
            tokenized_datasets["attention_mask"].append(tokens["attention_mask"][0])
            tokenized_datasets["labels"].append(example[config.second_column])
        elif config.task_type == "image_classification":
            image = example[config.first_column]
            image = image.resize((28, 28))
            image = np.array(image)
            tokenized_datasets["image"].append(image)
            tokenized_datasets["labels"].append(example[config.second_column])

    if config.task_type == "text_classification":
        return CustomTextDataset(tokenized_datasets)
    elif config.task_type == "text_generation":
        return CustomTextDataset(tokenized_datasets)
    elif config.task_type == "image_classification":
        return CustomImageDataset(tokenized_datasets)


def load_dataloader(
    dataset: DatasetDict,
    data_config: DataConfig,
    shuffle: bool = False,
):
    """
    Define load datasets for pretrained.
    """
    tokenized_dataset = preprocssing_dataset(dataset, data_config.config)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=data_config.batch_size,
        shuffle=shuffle,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )
    return dataloader


def is_cached(config: Config) -> bool:
    cache_dir = config.data_cache_dir
    for file in ["train.pkl", "valid.pkl", "test.pkl"]:
        file_path = os.path.join(cache_dir, file)
        if not os.path.exists(file_path):
            return False
    return True


def load_data(
    config: Config,
    batch_size: int = 32,
    valid_size: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    do_cache: bool = True,
):
    cache_dir = config.data_cache_dir
    data_config = DataConfig(config, batch_size, valid_size, num_workers, pin_memory)

    if do_cache and not is_cached(config):
        color_print(f"Downloading the Dataset {config.dataset_name}")
        config.init_seed()
        dataset = load_dataset(**config.dataset_args)

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        generator = torch.Generator().manual_seed(config.seed)

        if config.dataset_name == "YahooAnswersTopics":
            train_size = len(train_dataset)
            test_size = len(test_dataset)

            train_size //= 20
            test_size //= 2

            train_dataset, _ = random_split(
                train_dataset,
                [train_size, len(train_dataset) - train_size],
                generator=generator,
            )
            test_dataset, _ = random_split(
                test_dataset,
                [test_size, len(test_dataset) - test_size],
                generator=generator,
            )

        if "validation" in dataset or "valid" in dataset:
            valid_dataset = dataset.get("validation", dataset.get("valid"))
        else:
            valid_size = int(len(train_dataset) * data_config.valid_size)
            train_size = len(train_dataset) - valid_size
            train_dataset, valid_dataset = random_split(
                train_dataset, [train_size, valid_size]
            )
        train_dataloader = load_dataloader(
            train_dataset,
            data_config,
            True,
        )
        valid_dataloader = load_dataloader(
            valid_dataset,
            data_config,
        )
        test_dataloader = load_dataloader(test_dataset, data_config)

        if do_cache:
            save_cache(train_dataloader, cache_dir, "train.pkl")
            save_cache(valid_dataloader, cache_dir, "valid.pkl")
            save_cache(test_dataloader, cache_dir, "test.pkl")
            color_print("Caching is completed.")
    else:
        color_print(f"Loading cached dataset {config.dataset_name}.")
        train_dataloader = load_cache(cache_dir, "train.pkl")
        valid_dataloader = load_cache(cache_dir, "valid.pkl")
        test_dataloader = load_cache(cache_dir, "test.pkl")
    color_print(f"The dataset {config.dataset_name} is loaded")
    config.dataset_summary()
    return train_dataloader, valid_dataloader, test_dataloader


def save_checkpoint(model, save_point):
    path = os.path.join(Paths.get_dir(save_point), "model.pth")
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    color_print(f"Checkpoint saved at {path}")


def load_checkpoint(model, check_point: str):
    path = os.path.join(check_point, "model.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path)

    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise KeyError(
            "Checkpoint structure is unrecognized. 'model_state_dict' key not found."
        )

    model.load_state_dict(model_state_dict, strict=True)

    return model


def load_model(config: Config, checkpoint: str = None):
    cache_dir = config.model_cache_dir
    color_print(f"Loading the model.")
    config.model_summary()

    if config.task_type == "text_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, cache_dir=cache_dir
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name, cache_dir=cache_dir
        )

    # load check point
    if checkpoint is not None:
        checkpoint_path = os.path.join(config.Checkpoints, checkpoint)
        model = load_checkpoint(model, checkpoint_path)

    color_print(f"The model {config.model_name} is loaded.")
    return model


def load_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    return tokenizer


torch.serialization.add_safe_globals(
    [
        DataLoader,
        Subset,
        RandomSampler,
        BatchSampler,
        SequentialSampler,
        default_collate,
        CustomTextDataset,
        CustomImageDataset,
        CustomEmbeddingDataset,
        CustomCodeDataset,
        Config,
    ]
)
