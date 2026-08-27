import re

import pandas as pd
from pandas import DataFrame

PENDING_STICKER_CAPTION = "[表情包:待处理]"


def _normalize_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    result = str(text).replace("\r\n", "\n").replace("\r", "\n")
    result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", result)
    return result.strip()


def _build_normalized_content(
        message_type: str,
        raw_content: object,
        sticker_caption: object = None
) -> tuple[str, bool]:
    if message_type == "text":
        return _normalize_text(raw_content), True
    if message_type == "sticker":
        caption = _normalize_text(sticker_caption)
        if caption and caption != PENDING_STICKER_CAPTION:
            return caption, True
        return PENDING_STICKER_CAPTION, False
    raise ValueError(f"不支持的内部消息类型：{message_type}")


def normalize_message(
        df: DataFrame,
        message_type_col: str = "message_type",
        content_col: str = "message",
        sticker_caption_col: str = "sticker_caption"
) -> DataFrame:
    required_columns = {message_type_col, content_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要列：{sorted(missing_columns)}")
    result = df.copy()
    if sticker_caption_col not in result.columns:
        result[sticker_caption_col] = None
    normalized_values = result.apply(
        lambda row: _build_normalized_content(
            message_type=row[message_type_col],
            raw_content=row[content_col],
            sticker_caption=row[sticker_caption_col]
        ),
        axis=1
    )
    result["normalized_text"] = [item[0] for item in normalized_values]
    result["sticker_resolved"] = [item[1] for item in normalized_values]
    return result
