import re
import pandas as pd
from pandas import DataFrame
from typing import Optional

"""
规范化文本内容，便于后续转成训练格式
"""


def _normalize_text(text: obiect) -> str:
    """
    对普通文本进行轻量标准化。

    不修改错别字、方言、口头禅和重复字，
    避免破坏目标人物的聊天风格。
    """
    if text is None or pd.isna(text):
        return ""
    
    res = str(text)
    
    #统一换行符
    res = res.replace("\r\n", "\n")
    res = res.replace("\r","\n")
    
    # 删除不可见控制字符，但保留换行和制表符
    res = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        "",
        res
    )
    
    return res.strip()

def _normalize_message_types(
    df: DataFrame,
    raw_type_col: str = "type_name",
    output_col: str = "message_type"
) -> DataFrame:
    """
    将原始中文消息类型转换为项目内部统一类型。

    文本   → text
    表情包 → sticker
    """
    if raw_type_col not in df.columns:
        raise ValueError(
            f"缺少消息类型列：{raw_type_col}"
        )

    result = df.copy()

    result =  result.rename(columns={raw_type_col:output_col})

    return result
    
def _build_normalized_content(
    message_type: str,
    raw_content: object,
    sticker_caption: Optional[str] = None
) -> tuple[str, bool]:
    """
    为文本和表情包生成统一的 normalized_text。

    Returns
    -------
    tuple[str, bool]
        normalized_text:
            最终统一文本。
        sticker_resolved:
            表情包是否已经完成描述。
    """
    if message_type == "text":
        return _normalize_text(raw_content) , True
    
    if message_type == "sticeker":
        caption = _normalize_text(sticker_caption)
        
        if caption:
            return caption , True
        
        return "[表情包：待处理]" , False
    
    raise ValueError(
        f"不支持的内部消息类型：{message_type}"
    )
        
def normalize_message(
    df: DataFrame,
    raw_type_col: str = "type_name",
    content_col: str = "msg",
    sticker_caption_col: str = "sticker_caption"
) -> DataFrame:
    """
    批量标准化文本和表情包消息。
    返回的 DataFrame 会新增：
    - normalized_text
    - sticker_resolved
    """
    if content_col not in df.columns:
        raise ValueError(
            f"缺少消息内容列：{content_col}"
        )
        
    res = _normalize_message_types(
        df=df,
        raw_type_col=raw_type_col,
        output_col="message_type"
    )
    
     # 初次运行时可能还没有 sticker_caption 列
    if sticker_capation_col not in df.columns:
        res[sticker_caption_col] = None
        
    normalized_values = res.apply(
        lambda row:_build_normalized_content(
                message_type=row["message_type"],
                raw_content=row[contentcol],
                sticker_caption=row[sticker_caption_col]
        ),
        axis=1
    )
    
    res["normalized_text"] = [
        item[0]
        for item in normalized_values
    ]
    
    res["sticker_resolved"] = [
        item[1]
        for item in normalized_values
    ]
    
    return res