from typing import Any

import pandas as pd
from pandas import DataFrame


def _speaker_from_sender(
        value: object,
        self_speaker: str,
        other_speaker: str
) -> str:
    if value is None or pd.isna(value):
        raise ValueError("is_sender存在空值，无法判断说话人")

    normalized = str(value).strip().lower()

    if normalized in {"1","1.0","true"}:
        return self_speaker

    if normalized in {"0","0.0","false"}:
        return other_speaker

    raise ValueError(f"无法识别is_sender值：{value}")


def build_turns(
        df: DataFrame,
        speaker_col: str = "is_sender",
        text_col: str = "normalized_text",
        timestamp_col: str = "unix_seconds",
        message_id_col: str = "message_id",
        merge_gap_seconds: int = 120,
        max_turn_duration_seconds: int = 300,
        max_messages_per_turn: int = 10,
        break_token: str = "<msg_break>",
        self_speaker: str = "assistant",
        other_speaker: str = "user"
) -> list[dict[str,Any]]:
    required_columns = {
        speaker_col,
        text_col,
        timestamp_col,
        message_id_col
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Turn构造缺少必要列：{sorted(missing_columns)}"
        )

    if merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds不能小于0")

    if not break_token:
        raise ValueError("message_break_token不能为空")

    sort_columns = [timestamp_col]

    if "raw_index" in df.columns:
        sort_columns.append("raw_index")
    else:
        sort_columns.append(message_id_col)

    ordered = (
        df.sort_values(
            sort_columns,
            kind="stable"
        )
        .reset_index(drop=True)
    )

    turns = []
    current = None

    for _,row in ordered.iterrows():
        speaker = _speaker_from_sender(
            value=row[speaker_col],
            self_speaker=self_speaker,
            other_speaker=other_speaker
        )

        if pd.isna(row[timestamp_col]):
            raise ValueError(
                f"message_id={row[message_id_col]} 时间戳为空"
            )

        timestamp = int(row[timestamp_col])
        message_id = str(row[message_id_col])

        text = (
            ""
            if pd.isna(row[text_col])
            else str(row[text_col]).strip()
        )

        if not text:
            continue

        if current is None:
            current = {
                "session_id":None,
                "speaker":speaker,
                "start_timestamp":timestamp,
                "end_timestamp":timestamp,
                "message_ids":[message_id],
                "messages":[text]
            }
            continue

        gap = timestamp - current["end_timestamp"]

        if gap < 0:
            raise ValueError(
                f"消息时间顺序异常：{message_id}"
            )

        turn_duration = timestamp - current["start_timestamp"]

        can_merge = (
            speaker == current["speaker"]
            and gap <= merge_gap_seconds
            and turn_duration <= max_turn_duration_seconds
            and len(current["message_ids"]) < max_messages_per_turn
        )

        if can_merge:
            current["end_timestamp"] = timestamp
            current["message_ids"].append(message_id)
            current["messages"].append(text)
            continue

        turns.append(current)

        current = {
            "session_id":None,
            "speaker":speaker,
            "start_timestamp":timestamp,
            "end_timestamp":timestamp,
            "message_ids":[message_id],
            "messages":[text]
        }

    if current is not None:
        turns.append(current)

    separator = f"\n{break_token}\n"

    for index,turn in enumerate(turns):
        turn["turn_id"] = f"turn_{index:08d}"
        turn["merged_text"] = separator.join(
            turn["messages"]
        )
        turn["message_count"] = len(
            turn["message_ids"]
        )

    print(f"输入消息数量：{len(ordered)}")
    print(f"Turn数量：{len(turns)}")

    if turns:
        average_messages = sum(
            turn["message_count"]
            for turn in turns
        ) / len(turns)

        print(
            f"平均每个Turn消息数："
            f"{average_messages:.2f}"
        )

    return turns