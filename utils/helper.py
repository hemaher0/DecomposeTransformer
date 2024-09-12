import os
import torch
import json
import pyrootutils
from os.path import join as join
from os.path import exists
from os import makedirs as makedirs
from colorama import Fore, Style, init
from typing import *


class Paths:
    """Class for managing directory paths."""

    def __init__(self) -> None:
        self.root = pyrootutils.find_root()
        os.chdir(self.root)

        # Set roots
        self.Datasets = Paths.get_dir("Datasets")
        self.Models = Paths.get_dir("Models")
        self.Modules = Paths.get_dir("Modules")
        self.Logs = Paths.get_dir("Logs")
        self.Checkpoints = Paths.get_dir("Checkpoint")

    @staticmethod
    def get_dir(path: str) -> str:
        makedirs(path, exist_ok=True)
        return path


class ModelConfig:
    def __init__(
            self,
            model_name: str,
            device: torch.device = torch.device("cuda:0"),
    ) -> None:
        """
        Initialize the configuration for the model.

        Args:
            model_name (str): The name of the model.
            device (torch.device): The device to run the model on. Defaults to "cuda:0".
        """
        # Specific directories

        self.config: Dict[str, Any] = ModelConfig.load_config(model_name)
        self.model_name: str = self.config["model_name"]
        self.tokenizer_name: str = self.config["tokenizer_name"]
        self.task_type: str = self.config["task_type"]
        self.architectures: List[str] = self.config["architectures"]
        self.dataset_name: str = self.config["dataset_name"]
        self.num_labels: int = self.config["num_labels"]
        self.cache_dir: str = self.config["cache_dir"]
        self.device: torch.device = device

        self.Datasets: str = Paths.get_dir("Datasets")
        self.Models: str = Paths.get_dir("Models")
        self.Modules: str = Paths.get_dir("Modules")
        self.Checkpoints: str = Paths.get_dir("Checkpoint")

    @staticmethod
    def load_config(model_name: str) -> Dict[str, Any]:
        with open("utils/config.json", "r") as json_file:
            config = json.load(json_file)
            model_config = config["model"][model_name]
        return model_config

    def summary(self) -> None:
        color_print(self.config)


class DataConfig:
    def __init__(
            self,
            dataset_name: str,
            device: torch.device = torch.device("cuda:0"),
            max_length: int = 512,
            batch_size: int = 4,
            valid_size: float = 0.1,
            num_workers: int = 4,
            pin_memory: bool = True,
            seed: int = 42,
            return_fields: Optional[List[str]] = None,
            do_cache: bool = True,
    ) -> None:
        if return_fields is None:
            return_fields = ["input_ids", "attention_mask", "labels"]

        self.config: Dict[str, Any] = DataConfig.load_config(dataset_name)
        self.device: torch.device = device
        self.dataset_name: str = self.config["dataset_name"]
        self.cache_dir: str = self.config["cache_dir"]
        self.text_column: str = self.config["text_column"]
        self.label_column: str = self.config["label_column"]
        self.task_type: str = self.config["task_type"]
        self.dataset_args: Dict[str, str] = {
            "path": self.config["path"],
            "name": self.config["config_name"],
            "cache_dir": self.config["cache_dir"],
        }
        self.max_length: int = max_length
        self.batch_size: int = batch_size
        self.valid_size: float = valid_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.seed: int = seed
        self.return_fields: List[str] = return_fields

        self.do_cache: bool = do_cache

    def is_cached(self) -> bool:
        train = join(self.cache_dir, "train.pkl")
        valid = join(self.cache_dir, "valid.pkl")
        test = join(self.cache_dir, "test.pkl")
        return exists(train) and exists(valid) and exists(test)

    @staticmethod
    def load_config(dataset_name: str) -> Dict[str, Any]:
        with open("utils/config.json", "r") as json_file:
            config = json.load(json_file)
            data_config = config["dataset"][dataset_name]
        return data_config

    def summary(self) -> None:
        color_print(self.config)


def color_print(data: Any) -> None:
    init(autoreset=True)
    print(f"{Fore.CYAN}{data}{Style.RESET_ALL}")
