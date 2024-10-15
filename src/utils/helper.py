import re
import os
import torch
import json
import pickle
import random
import numpy as np
import pandas as pd
from typing import *
from colorama import Fore, Style, init
from transformers import AutoConfig
from pprint import pprint
from .torchtest import check_device


class Paths:
    """Class for managing directory paths."""

    def __init__(self) -> None:
        self.root = self.find_root()
        os.chdir(self.root)
        self.Datasets = Paths.get_dir("datasets")
        self.Models = Paths.get_dir("models")
        self.Results = Paths.get_dir("results")
        self.Checkpoints = Paths.get_dir("checkpoints")

    def find_root(self):
        current_dir = os.getcwd()
        while current_dir != "/":
            if ".project-root" in os.listdir(current_dir):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        return None

    @staticmethod
    def get_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path


class Config:
    def __init__(
        self,
        model_name: str,
        device: torch.device = torch.device("cuda:0"),
        seed: int = 42,
    ) -> None:
        self.task_type: str
        self.config: Dict

        self.task_type, self.config = Config.load_model_config(model_name)
        self.model_name: str = self.config["model_name"]
        try:
            self.model_config = AutoConfig.from_pretrained(self.model_name)
        except:
            self.model_config = None

        self.tokenizer_name: Optional[str] = self.config.get("tokenizer_name", None)
        self.architectures: List[str] = self.config["architectures"]
        self.dataset_name: str = self.config["dataset_name"]
        self.num_labels: int = self.config["num_labels"]
        self.model_cache_dir: str = Paths.get_dir(f"models/{model_name}")
        self.data_cache_dir: str = Paths.get_dir(f"datasets/{self.dataset_name}")
        self.device: torch.device = device
        check_device()

        # Paths for directories
        self.Datasets: str = Paths.get_dir("datasets")
        self.Models: str = Paths.get_dir("models")
        self.Outputs: str = Paths.get_dir("results")
        self.Checkpoints: str = Paths.get_dir("checkpoints")

        # Load dataset configuration
        self.dataset_config: Dict[str, Any] = Config.load_data_config(self.dataset_name)
        self.dataset_path: str = self.dataset_config["path"]
        self.config_name: str = self.dataset_config["config_name"]
        self.first_column: str = self.dataset_config["features"]["first_column"]
        self.second_column: str = self.dataset_config["features"]["second_column"]

        self.return_fields: List[str] = self.get_return_fields()
        self.seed: int = seed
        self.init_seed()

        # Dataset arguments for loading
        self.dataset_args: Dict[str, str] = {
            "path": self.dataset_path,
            "name": self.config_name,
            "cache_dir": self.data_cache_dir,
        }

    @staticmethod
    def load_model_config(model_name: str) -> Dict[str, Any]:
        with open("./config.json", "r") as json_file:
            config = json.load(json_file)
            for model_task, model_info in config["model"].items():
                for key, details in model_info.items():
                    if model_name in details.get("model_name", []):
                        details["model_name"] = details["model_name"][model_name]
                        return model_task, details

        raise NotImplementedError(f"{model_name} is not allowd model yet.")

    @staticmethod
    def load_data_config(dataset_name: str) -> Dict[str, Any]:
        with open("./config.json", "r") as json_file:
            config = json.load(json_file)
            return config["dataset"][dataset_name]

    def get_return_fields(self) -> List[str]:
        """Returns the appropriate return fields based on task type."""
        if self.task_type == "text_classification":
            return ["input_ids", "attention_mask", "labels"]
        elif self.task_type == "text_generation":
            return ["input_ids", "attention_mask", "summaries"]
        elif self.task_type == "image_classification":
            return ["image", "labels"]
        elif self.task_type == "embedding":
            return ["embedding", "attention_mask", "labels"]
        else:
            raise TypeError(f"{self.task_type} is unsupported dataset type.")

    def init_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def model_summary(self) -> None:
        pprint(self.config)

    def dataset_summary(self) -> None:
        pprint(self.dataset_config)


def color_print(data: Any) -> None:
    init(autoreset=True)
    print(f"{Fore.CYAN}{data}{Style.RESET_ALL}")


def save_cache(data, cache_dir, filename):
    with open(os.path.join(cache_dir, filename), "wb") as f:
        pickle.dump(data, f)
    print(f"{filename} is cached.")


def load_cache(cache_dir, filename):
    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"{filename} is loaded from cache.")
        return data
    else:
        raise FileNotFoundError(f"{filename} not found in {cache_dir}")


def report_to_df(report):
    report_str = report["report"].strip()
    lines = [line.strip() for line in report_str.split("\n") if line.strip()]

    data = []

    for line in lines[1:]:
        row_data = re.split(r"\s+", line)
        if len(row_data) >= 5 and row_data[0] not in ["accuracy", "macro", "weighted"]:
            data.append(
                {
                    "class": row_data[0],
                    "precision": float(row_data[1]),
                    "recall": float(row_data[2]),
                    "f1-score": float(row_data[3]),
                    "support": int(row_data[4]),
                }
            )

    report_df = pd.DataFrame(data)
    report_df.columns = ["class", "precision", "recall", "f1-score", "support"]

    return report_df


def append_nth_row(df_list):
    data = []

    for idx, df in enumerate(df_list):
        if idx < len(df):
            row = df.iloc[idx]
            data.append(row)
        else:
            data.append(pd.Series(dtype="float64"))

    new_df = pd.DataFrame(data).reset_index(drop=True)

    return new_df
