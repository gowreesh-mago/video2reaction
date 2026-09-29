"""Import and version checks; no model downloads or GPU allocation."""
import importlib
import importlib.metadata
import json
import platform

modules = {"torch": "torch", "torchvision": "torchvision", "transformers": "transformers",
           "accelerate": "accelerate", "peft": "peft", "timm": "timm", "numpy": "numpy",
           "scipy": "scipy", "pandas": "pandas", "scikit-learn": "sklearn", "pillow": "PIL",
           "opencv-python-headless": "cv2", "matplotlib": "matplotlib", "tqdm": "tqdm",
           "pyyaml": "yaml", "einops": "einops", "safetensors": "safetensors",
           "huggingface-hub": "huggingface_hub"}
versions = {}
for package, module in modules.items():
    importlib.import_module(module)
    versions[package] = importlib.metadata.version(package)
print(json.dumps({"python": platform.python_version(), "packages": versions}, indent=2))
