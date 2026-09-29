"""Run on an allocated compute node, not the login node."""
import json
import platform
import socket
import torch


def gpu_info():
    return {
        "hostname": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
        "gpus": [{"name": torch.cuda.get_device_name(i),
                  "vram_gib": torch.cuda.get_device_properties(i).total_memory / 2**30}
                 for i in range(torch.cuda.device_count())],
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }


if __name__ == "__main__":
    print(json.dumps(gpu_info(), indent=2))
