import torch
import os
import copy


def save_module(module, save_path, module_name):
    save_path = os.path.join(save_path, module_name)
    torch.save(module.state_dict(), save_path)


def load_module(module, save_path, module_name, device="cpu"):
    copied_module = copy.deepcopy(module)
    load_path = os.path.join(save_path, module_name)
    copied_module.load_state_dict(torch.load(load_path, map_location=device))
    return copied_module
