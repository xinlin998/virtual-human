from pathlib import Path

import pandas as pd

from src.config import load_config
from src.normalize import normalize_message
from src.preprocessed import add_message_id,drop_empty_messages,filter_message_types,process_chat_csv
from src.sticker_pipeline import (
    QwenStickerAnalyzer,
    process_stickers,
    save_sticker_metadata,
    save_sticker_review_queue
)
from src.utils import ensure_parent_dir

PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(relative_path: str) -> Path:
    return (PROJECT_ROOT / relative_path).resolve()


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs/pipeline.yaml")
    data_config = config["data"]
    preprocess_config = config["preprocess"]
    sticker_config = config["sticker"]
    input_csv = _project_path(data_config["input_csv"])
    processed_csv = _project_path(data_config["processed_csv"])
    stickered_csv = _project_path(data_config["stickered_csv"])
    sticker_metadata_jsonl = _project_path(data_config["sticker_metadata_jsonl"])
    sticker_review_jsonl = _project_path(data_config["sticker_review_jsonl"])
    model_cache_dir = _project_path(data_config["model_cache_dir"])
    df = pd.read_csv(input_csv)
    df = process_chat_csv(df)
    df = filter_message_types(df, allowed_types=preprocess_config["allowed_message_types"])
    df = drop_empty_messages(df)
    df = add_message_id(df)
    ensure_parent_dir(processed_csv)
    df.to_csv(processed_csv, index=False)
    analyzer = QwenStickerAnalyzer(
        model_name=sticker_config["model_name"],
        model_dir=model_cache_dir,
        max_new_tokens=sticker_config["max_new_tokens"],
        gif_max_frames=sticker_config["gif_max_frames"]
    )
    df,sticker_metadata = process_stickers(
        df=df,
        analyzer=analyzer,
        phash_threshold=sticker_config["phash_threshold"],
        batch_size=sticker_config["batch_size"],
        gif_batch_size=sticker_config["gif_batch_size"],
        retry_failed=sticker_config["retry_failed"],
        min_quality_score=sticker_config["min_quality_score"],
        image_root=data_config["sticker_image_dir"]
    )
    df = normalize_message(df)
    ensure_parent_dir(stickered_csv)
    df.to_csv(stickered_csv, index=False)
    save_sticker_metadata(sticker_metadata, sticker_metadata_jsonl)
    save_sticker_review_queue(sticker_metadata, sticker_review_jsonl)
    print(f"处理完成：{stickered_csv}")
    print(f"表情包元数据：{sticker_metadata_jsonl}")
    print(f"人工复核队列：{sticker_review_jsonl}")


if __name__ == "__main__":
    main()
