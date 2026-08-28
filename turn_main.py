import json
from pathlib import Path

import pandas as pd

from src.build_turns import build_turns
from src.config import load_config
from src.privacy import (
    anonymize_dataframe,
    build_anonymized_message_table,
    load_privacy_aliases,
    is_export_noise,
)
from src.utils import ensure_parent_dir,save_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(relative_path: str) -> Path:
    return (PROJECT_ROOT/relative_path).resolve()


def main() -> None:
    config = load_config(PROJECT_ROOT/"configs/pipeline.yaml")

    data_config = config["data"]
    privacy_config = config["privacy"]
    turn_config = config["turn"]

    input_csv = _project_path(
        data_config["stickered_csv"]
    )

    anonymized_csv = _project_path(
        data_config["anonymized_messages_csv"]
    )

    privacy_report_json = _project_path(
        data_config["privacy_report_json"]
    )

    turns_jsonl = _project_path(
        data_config["turns_jsonl"]
    )

    print(f"读取标准消息表：{input_csv}")

    df = pd.read_csv(input_csv)

    if privacy_config["enabled"]:
        aliases = load_privacy_aliases(
            _project_path(
                privacy_config["aliases_file"]
            )
        )

        df,privacy_stats = anonymize_dataframe(
            df=df,
            text_col="normalized_text",
            aliases=aliases
        )
    else:
        df["privacy_changed"] = False
        df["privacy_types"] = "[]"

        privacy_stats = {
            "total_messages":len(df),
            "changed_messages":0,
            "unchanged_messages":len(df),
            "type_counts":{}
        }

    df = build_anonymized_message_table(df)

    ensure_parent_dir(anonymized_csv)

    df.to_csv(
        anonymized_csv,
        index=False
    )

    ensure_parent_dir(privacy_report_json)

    privacy_report_json.write_text(
        json.dumps(
            privacy_stats,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )
    noise_mask = df["normalized_text"].apply(is_export_noise)
    print(f"过滤微信导出污染消息：{int(noise_mask.sum())}")

    df = df[~noise_mask].copy().reset_index(drop=True)

    turns = build_turns(
        df=df,
        speaker_col="is_sender",
        text_col="normalized_text",
        timestamp_col="unix_seconds",
        message_id_col="message_id",
        merge_gap_seconds=turn_config["merge_gap_seconds"],
        break_token=turn_config["message_break_token"],
        self_speaker=turn_config["self_speaker"],
        other_speaker=turn_config["other_speaker"]
    )

    save_jsonl(
        turns,
        turns_jsonl
    )

    print("\n处理完成")
    print(f"脱敏消息表：{anonymized_csv}")
    print(f"脱敏统计：{privacy_report_json}")
    print(f"Turn数据：{turns_jsonl}")


if __name__ == "__main__":
    main()