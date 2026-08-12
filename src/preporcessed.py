import pandas as pd
from src.config import load_config
from pandas import DataFrame
from pathlib import Path
"""
主要工作：
读取 CSV；
检查必要列是否存在；
解析 CreateTime；
生成 Unix 时间戳;
计算相邻消息时间差；
生成 message_id
过滤无关信息只保留文本和表情包
"""


#导入配置参数
config = load_config("configs/pipeline.yaml")

SAVE_COLUMNS = ["raw_index","message_type","is_sender","message","sticker_path","CreateTime","unix_seconds","time_gap"]

def process_chat_csv(df: DataFrame) -> DataFrame:
    #处理时间
    time_str = df['CreateTime']
    #转为时间格式
    dt = pd.to_datetime(time_str)
    #秒级时间戳
    df["unix_seconds"] = (
        dt.astype("int64") // 10**9
    )
    #返回两条消息之间的时间差
    df["time_gap"] = df["unix_seconds"].diff()
    
    #修改列名
    df["raw_index"] = df["id"]
    df['message_type'] = df['type_name']
    df['message'] = df['msg']
    df['sticker_path'] = df['src']

    return df[SAVE_COLUMNS]

#正常不返回，错误时抛异常。
def validate_dataframe_columns(path:str) -> None:
    df = pd.read_csv(path)
    for _ in config['columns'].values():
        if _ not in df.columns.tolist():
            print(f"缺少{_}列，请检查后重试")
    pass


def add_message_id(df:DataFrame,id_col:str="message_id",prefix:str="msg") -> DataFrame:
    """
    为 DataFrame 中的每条消息添加唯一 ID。
    需要在后续过滤后使用
    Parameters
    ----------
    df:
        已完成时间排序的聊天 DataFrame。
    id_col:
        新增 ID 列的名称。
    prefix:
        ID 前缀。
    Returns
    -------
    pd.DataFrame:
        包含 message_id 列的新 DataFrame。
    """
    result = df[SAVE_COLUMNS].copy()
    
    result[id_col] = [
        f"{prefix}_{index:08d}"
        for index in range(len(result))
    ]
    
    return result

def filter_message_types(
    df: DataFrame,
    type_col: str,
    allowed_types: list[str]
) -> DataFrame:
    
    df = df[df[type_col].isin(allowed_types)]
    return df

#返回删除空内容后的 DataFrame
def drop_empty_messages(
    df: DataFrame,
    content_col: str
) -> pd.DataFrame:
    df = df[df[content_col] != None]
    return df

