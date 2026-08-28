from pathlib import Path
from typing import Any

import yaml

REQUIRED_CONFIG_KEYS = {
    "data":{
        "input_csv",
        "processed_csv",
        "stickered_csv",
        "anonymized_messages_csv",
        "privacy_report_json",
        "turns_jsonl",
        "sticker_metadata_jsonl",
        "sticker_review_jsonl",
        "sticker_image_dir",
        "model_cache_dir"
    },
    "preprocess":{
        "allowed_message_types"
    },
    "sticker":{
        "model_name",
        "batch_size",
        "gif_batch_size",
        "gif_max_frames",
        "phash_threshold",
        "max_new_tokens",
        "retry_failed",
        "min_quality_score"
    },
    "privacy":{
        "enabled",
        "aliases_file"
    },
    "turn":{
        "merge_gap_seconds",
        "message_break_token",
        "self_speaker",
        "other_speaker"
    }
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是YAML对象")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section,required_keys in REQUIRED_CONFIG_KEYS.items():
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"配置文件缺少有效section：{section}")
        missing_keys = required_keys - set(config[section])
        if missing_keys:
            raise ValueError(f"配置项{section}缺少：{sorted(missing_keys)}")
    sticker = config["sticker"]
    if int(sticker["batch_size"]) <= 0 or int(sticker["gif_batch_size"]) <= 0:
        raise ValueError("batch_size和gif_batch_size必须大于0")
    if int(sticker["gif_max_frames"]) <= 0:
        raise ValueError("gif_max_frames必须大于0")
    score = float(sticker["min_quality_score"])
    if not 0 <= score <= 1:
        raise ValueError("min_quality_score必须位于0到1之间")
    
    turn = config["turn"]

    if int(turn["merge_gap_seconds"]) < 0:
        raise ValueError(
            "turn.merge_gap_seconds不能小于0"
        )

    if not str(turn["message_break_token"]).strip():
        raise ValueError(
            "turn.message_break_token不能为空"
        )
