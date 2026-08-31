import json
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


def ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    module_name = type(value).__module__
    if module_name.startswith("numpy"):
        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return to_jsonable(item_method())
            except ValueError:
                pass
        tolist_method = getattr(value, "tolist", None)
        if callable(tolist_method):
            return to_jsonable(tolist_method())
    raise TypeError(f"对象类型 {type(value).__name__} 不能转换为 JSON")


def save_jsonl(records: Iterable[Any], output_path: str | Path, ensure_ascii: bool = False) -> Path:
    output_path = ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(to_jsonable(record), ensure_ascii=ensure_ascii) + "\n")
    return output_path


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型输出中没有找到 JSON 对象")
    candidate = text[start:end + 1].replace("“", '"').replace("”", '"')
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    return result

def load_jsonl(path: str | Path) -> list[dict[str,Any]]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"JSONL文件不存在：{input_path}")

    records: list[dict[str,Any]] = []

    with input_path.open("r",encoding="utf-8") as file:
        for line_number,line in enumerate(file,start=1):
            text = line.strip()

            if not text:
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path} 第{line_number}行不是合法JSON"
                ) from exc

            if not isinstance(record,dict):
                raise ValueError(
                    f"{input_path} 第{line_number}行必须是JSON对象"
                )

            records.append(record)

    return records