import pandas as pd
from pandas import DataFrame

RAW_REQUIRED_COLUMNS = {"id", "type_name", "is_sender", "msg", "src", "CreateTime"}
SAVE_COLUMNS = [
    "raw_index",
    "message_type",
    "is_sender",
    "message",
    "sticker_path",
    "CreateTime",
    "unix_seconds",
    "time_gap",
]
TYPE_MAPPING = {
    "文本": "text",
    "表情包": "sticker",
    "text": "text",
    "sticker": "sticker",
}


def validate_dataframe_columns(df: DataFrame) -> None:
    missing_columns = RAW_REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"原始聊天数据缺少必要列：{sorted(missing_columns)}")


def process_chat_csv(df: DataFrame) -> DataFrame:
    validate_dataframe_columns(df)
    result = df.copy()
    result["CreateTime"] = pd.to_datetime(result["CreateTime"], errors="coerce")
    invalid_time_count = int(result["CreateTime"].isna().sum())
    if invalid_time_count:
        raise ValueError(f"CreateTime 中存在 {invalid_time_count} 条无法解析的时间")
    result = result.sort_values("CreateTime").reset_index(drop=True)
    result["unix_seconds"] = result["CreateTime"].astype("int64") // 10**9
    result["time_gap"] = result["unix_seconds"].diff()
    raw_types = result["type_name"].astype(str).str.strip()
    result["message_type"] = raw_types.map(TYPE_MAPPING).fillna(raw_types.str.lower())
    result["raw_index"] = result["id"]
    result["message"] = result["msg"]
    result["sticker_path"] = result["src"]
    return result[SAVE_COLUMNS].copy()


def filter_message_types(df: DataFrame, allowed_types: list[str], type_col: str = "message_type") -> DataFrame:
    if type_col not in df.columns:
        raise ValueError(f"缺少消息类型列：{type_col}")
    allowed = {str(item).strip().lower() for item in allowed_types}
    return df[df[type_col].isin(allowed)].copy().reset_index(drop=True)


def drop_empty_messages(
        df: DataFrame,
        message_type_col: str = "message_type",
        content_col: str = "message",
        sticker_path_col: str = "sticker_path"
) -> DataFrame:
    required_columns = {message_type_col, content_col, sticker_path_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要列：{sorted(missing_columns)}")
    text_valid = (
        df[message_type_col].eq("text")
        & df[content_col].notna()
        & df[content_col].astype(str).str.strip().ne("")
    )
    sticker_valid = (
        df[message_type_col].eq("sticker")
        & df[sticker_path_col].notna()
        & df[sticker_path_col].astype(str).str.strip().ne("")
    )
    return df[text_valid | sticker_valid].copy().reset_index(drop=True)


def add_message_id(df: DataFrame, id_col: str = "message_id", prefix: str = "msg") -> DataFrame:
    result = df.copy().reset_index(drop=True)
    result[id_col] = [f"{prefix}_{index:08d}" for index in range(len(result))]
    return result
