import torch


def check_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        return (
            True,
            f"CUDA is available with {device_count} device(s): {', '.join(device_names)}",
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True, "MPS (Metal Performance Shaders) is available."
    else:
        return True, "CPU is available."


def main():
    is_available, message = check_device()

    if is_available:
        print(f"Device check successful: {message}")
    else:
        print("No suitable device found. Using CPU.")

    return is_available


if __name__ == "__main__":
    main()
