import os


def save_model(module, save_path):
    os.makedirs(save_path, exist_ok=True)
    module.save_pretrained(save_path, safe_serialization=False)
