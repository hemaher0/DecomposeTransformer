import torch


def check_device(device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return False, "CUDA is not available."
        device_id = int(device.split(":")[1])
        if device_id >= torch.cuda.device_count():
            return False, f"CUDA device {device_id} is not available."
        return True, torch.cuda.get_device_name(device_id)
    elif device == "cpu":
        return True, "CPU"
    elif device.startswith("mps"):
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            return False, "MPS (Metal Performance Shaders) is not available."
        return True, "MPS (Metal Performance Shaders)"
    else:
        return False, "Unknown device type."


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 torchtest.py <device>")
        sys.exit(1)

    device = sys.argv[1]
    is_available, message = check_device(device)

    if is_available:
        print(f"Device {device} is available: {message}")
    else:
        print(f"Device {device} is not available: {message}")
    return is_available


if __name__ == "__main__":
    main()
