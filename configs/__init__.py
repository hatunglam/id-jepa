from functools import lru_cache
from typing import Any, Dict

from .load_config import load_config

# def load_config(config_path: str = "./config.json") -> Dict[str, Any]:
#     config_path = Path(config_path).resolve()
#     with open(config_path, encoding="utf-8") as f:
#         config: Dict[str, Any] = json.load(f)
#      return config

@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    return load_config()

# _________________________________________________________________________________________


@lru_cache(maxsize=1)
def get_image_config() -> Dict[str, Any]:
    return get_config()["image"]


@lru_cache(maxsize=1)
def get_image_model_config() -> Dict[str, Any]:
    return get_image_config()["model"]


@lru_cache(maxsize=1)
def get_image_dataset_config() -> Dict[str, Any]:
    return get_image_config()["dataset"]


@lru_cache(maxsize=1)
def get_image_profiling_config() -> Dict[str, Any]:
    return get_image_config()["profiling"]


@lru_cache(maxsize=1)
def get_image_experiment_config() -> Dict[str, Any]:
    return get_image_config()["experiment"]


@lru_cache(maxsize=1)
def get_image_runtime_config() -> Dict[str, Any]:
    return get_image_config()["runtime"]


@lru_cache(maxsize=1)
def get_image_tracking_config() -> Dict[str, Any]:
    return get_image_config()["tracking"]


@lru_cache(maxsize=1)
def get_image_dataset_transforms_config() -> Dict[str, Any]:
    return get_image_dataset_config()["transforms"]


