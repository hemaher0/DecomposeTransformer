import torch
import os
import copy


def save_module(module, save_path, module_name):
    save_path = os.path.join(save_path, module_name)
    os.makedirs(save_path, exist_ok=True)
    module.save_pretrained(save_path, safe_serialization=False)


def load_module(module_class, save_path, module_name, device="cpu"):
    load_path = os.path.join(save_path, module_name)
    return module_class.from_pretrained(save_path, map_location=device)
