from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "revflow_xl_256.yaml"


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_weights(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
