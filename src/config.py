from pathlib import Path
from typing import Any

import yaml

REQUIRED_CONFIG_KEYS = {
    "data": {
        "input_csv",
        "processed_csv",
        "stickered_csv",
        "sticker_metadata_jsonl",
        "sticker_image_dir",
        "model_cache_dir",
    },
    "preprocess": {"allowed_message_types"},
    "sticker": {"model_name", "batch_size", "phash_threshold", "max_new_tokens"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是 YAML 对象")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section, required_keys in REQUIRED_CONFIG_KEYS.items():
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"配置文件缺少有效 section：{section}")
        missing_keys = required_keys - set(config[section])
        if missing_keys:
            raise ValueError(f"配置项 {section} 缺少：{sorted(missing_keys)}")
