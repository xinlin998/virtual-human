import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas import DataFrame


PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ID_CARD_PATTERN = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}"
    r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
    r"\d{3}[\dXx](?!\d)"
)
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")
IP_PATTERN = re.compile(
    r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"
)

VERIFICATION_CODE_PATTERN = re.compile(
    r"(?P<label>验证码|校验码|动态码)"
    r"\s*(?:是|为|[:：])?\s*"
    r"(?P<value>[A-Za-z0-9]{4,10})",
    re.IGNORECASE
)

PASSWORD_PATTERN = re.compile(
    r"(?P<label>门锁密码|支付密码|登录密码|密码)"
    r"\s*(?:是|为|[:：])?\s*"
    r"(?P<value>[A-Za-z0-9@#_.\-]{4,32})",
    re.IGNORECASE
)

BANK_CARD_PATTERN = re.compile(
    r"(?P<label>银行卡号|银行卡|卡号)"
    r"\s*(?:是|为|[:：])?\s*"
    r"(?P<value>(?:\d[\s-]?){15,18}\d)"
)

ACCOUNT_PATTERN = re.compile(
    r"(?P<label>微信号|微信ID|QQ号|QQ|账号)"
    r"\s*(?:是|为|[:：])?\s*"
    r"(?P<value>[A-Za-z0-9_\-]{5,32})",
    re.IGNORECASE
)

ADDRESS_PATTERN = re.compile(
    r"(?P<label>收货地址|家庭地址|住址|地址)"
    r"\s*(?:是|为|[:：])?\s*"
    r"(?P<value>[^\n，,。；;]{4,80})"
)

COORDINATE_PATTERN = re.compile(
    r"(?P<label>经纬度|坐标|定位)"
    r"\s*(?:是|为|[:：])?\s*"
    r"(?P<value>-?\d{1,3}(?:\.\d+)?\s*[,，]\s*-?\d{1,3}(?:\.\d+)?)"
)

ROOM_PATTERN = re.compile(
    r"(?P<label>房间号|房号)"
    r"[ \t]*(?:是|为|[:：])?[ \t]*"
    r"(?P<value>[A-Za-z0-9\-]{2,12})",
    re.IGNORECASE
)

XML_TAG_PATTERN = re.compile(r"</?\w+[^>]*>")
HTML_ENTITY_PATTERN = re.compile(r"&(?:#x?[0-9A-Fa-f]+|amp|lt|gt|quot);")
EXPORT_NOISE_PATTERN = re.compile(r"Orthogonal:|<!\[CDATA\[|<msg>|<appmsg")

def load_privacy_aliases(path: str | Path | None) -> dict[str,str]:
    if path is None:
        return {}

    alias_path = Path(path)

    if not alias_path.exists():
        return {}

    with alias_path.open('r',encoding='utf-8') as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data,dict):
        raise ValueError('隐私别名文件必须是YAML对象')

    replacements = data.get('replacements',{})

    if not isinstance(replacements,dict):
        raise ValueError('replacements必须是YAML对象')

    result = {}

    for source,replacement in replacements.items():
        source_text = str(source).strip()
        replacement_text = str(replacement).strip()

        if source_text and replacement_text:
            result[source_text] = replacement_text

    return result

def _replace_context_value(
        text: str,
        pattern: re.Pattern,
        placeholder: str
) -> tuple[str,bool]:
    changed = False

    def replacement(match: re.Match) -> str:
        nonlocal changed
        changed = True
        label = match.group("label")
        return f"{label}：{placeholder}"

    return pattern.sub(replacement,text),changed


def mask_private_text(
        text: object,
        aliases: dict[str,str] | None = None
) -> tuple[str,list[str]]:
    if text is None or pd.isna(text):
        return "",[]

    result = str(text)
    privacy_types = []

    #人工指定的姓名、昵称、地址、单位等
    if aliases:
        for source,replacement in sorted(
            aliases.items(),
            key=lambda item:len(item[0]),
            reverse=True
        ):
            if source in result:
                result = result.replace(source,replacement)
                privacy_types.append("custom_alias")

    replacements = [
        ("url",URL_PATTERN,"[链接]"),
        ("email",EMAIL_PATTERN,"[邮箱]"),
        ("id_card",ID_CARD_PATTERN,"[身份证号]"),
        ("phone",PHONE_PATTERN,"[手机号]"),
        ("ip",IP_PATTERN,"[IP地址]")
    ]

    for privacy_type,pattern,replacement in replacements:
        result,count = pattern.subn(replacement,result)

        if count > 0:
            privacy_types.append(privacy_type)

    contextual_patterns = [
        ("verification_code",VERIFICATION_CODE_PATTERN,"[验证码]"),
        ("password",PASSWORD_PATTERN,"[密码]"),
        ("bank_card",BANK_CARD_PATTERN,"[银行卡号]"),
        ("account",ACCOUNT_PATTERN,"[账号]"),
        ("coordinate",COORDINATE_PATTERN,"[坐标]"),
        ("address",ADDRESS_PATTERN,"[地址]"),
        ("room", ROOM_PATTERN, "[房间号]"),
    ]

    for privacy_type,pattern,placeholder in contextual_patterns:
        result,changed = _replace_context_value(
            result,
            pattern,
            placeholder
        )

        if changed:
            privacy_types.append(privacy_type)

    privacy_types = list(dict.fromkeys(privacy_types))

    return result,privacy_types


def anonymize_dataframe(
        df: DataFrame,
        text_col: str = "normalized_text",
        aliases: dict[str,str] | None = None
) -> tuple[DataFrame,dict[str,Any]]:
    if text_col not in df.columns:
        raise ValueError(f"缺少待脱敏列：{text_col}")

    result = df.copy()

    masked_texts = []
    changed_flags = []
    privacy_type_values = []
    type_counts = {}

    for value in result[text_col]:
        original = "" if value is None or pd.isna(value) else str(value)

        masked,privacy_types = mask_private_text(
            original,
            aliases=aliases
        )

        masked_texts.append(masked)
        changed_flags.append(masked != original)
        privacy_type_values.append(
            json.dumps(privacy_types,ensure_ascii=False)
        )

        for privacy_type in privacy_types:
            type_counts[privacy_type] = (
                type_counts.get(privacy_type,0) + 1
            )

    result[text_col] = masked_texts
    result["privacy_changed"] = changed_flags
    result["privacy_types"] = privacy_type_values

    changed_count = int(sum(changed_flags))

    stats = {
        "total_messages":int(len(result)),
        "changed_messages":changed_count,
        "unchanged_messages":int(len(result) - changed_count),
        "type_counts":dict(sorted(type_counts.items()))
    }

    return result,stats


def build_anonymized_message_table(df: DataFrame) -> DataFrame:
    result = df.copy()

    #这些列仍然含原始信息，不能进入后续脱敏数据
    drop_columns = [
        "message",
        "sticker_path",
        "sticker_caption"
    ]

    existing_columns = [
        column
        for column in drop_columns
        if column in result.columns
    ]

    if existing_columns:
        result = result.drop(columns=existing_columns)

    return result

def is_export_noise(text: object) -> bool:
    if text is None:
        return False

    value = str(text).strip()

    if not value:
        return False

    if EXPORT_NOISE_PATTERN.search(value):
        return True

    if len(value) > 1000 and XML_TAG_PATTERN.search(value):
        return True

    if len(value) > 1000 and HTML_ENTITY_PATTERN.search(value):
        return True

    return False