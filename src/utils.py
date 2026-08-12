import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)
_MISSING = object()

def ensure_parent_dir(path: str | Path) -> Path:
    """创建目标文件的父目录，并返回 Path 对象。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    return output_path

def to_jsonable(value: Any) -> Any:
    """
    将常见 Python 对象递归转换为可被 json 序列化的对象。
    支持：
    - dataclass
    - Pydantic BaseModel
    - Path
    - datetime/date/time
    - Enum
    - set/tuple/list
    - dict/Mapping
    - NumPy 标量与数组（安装 NumPy 时）
    """
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    
    if is_dataclass(value) and not isinstance(value,type):
        return to_jsonable(asdict(value))
    
     # 避免在 utils.py 中强依赖 Pydantic。
    model_dump = getattr(value,"model_dump",None)
    if callable(model_dump):
        return to_isonable(model_dump())
    
    if isinstance(value,Path):
        return str(value)
    
    if isinstance(value,(datatime,data,time)):
        return value.isoformate()
    
    if isinstance(value,Enum):
        return to_jsonable(value.value)
    
    if isinstance(value,Mapping):
        return {
            str(key):to_jsonable(item)
            for key,item in value.items()
        }
    
    if isinstance(value,(list,tuple)):
        return [to_jsonable(item) for item in value]
    
    if isinstance(value,set):
        converted = [to_jsonable(item) for item in value]
        try:
            return sorted(converted)
        except TypeError:
            return converted
        
    module_name = type(value).__module__
    if module_name.startswith("numpy"):
        item_method = getattr(value,"item",None)
        if callable(item_method):
            try:
                return to_jsonable(item_method())
            except ValueError:
                pass
        
        tolist_method = getattr(value,"tolist",None)
        if callable(tolist_method):
            return to_jsonable(tolist_method())
        
    raise TypeError(
        f"对象类型{type(value).__name__}不能转换为JSON"
    )
    
def save_json(
    data: Any,
    output_path: str | Path,
    *,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    atomic: bool = True,) -> Path :

    output_path = ensure_parent_dir(output_path)
    serializable = to_jsonable(data)
    
    if not atomic:
        with output_path.open("w",encoding="utf-8") as file:
            json.dump(
                serializable,
                file,
                ensure_ascii=ensure_ascii,
                indent=indent
            )
            return output_path
        
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    
    try:
        with os.fdopen(file_desciptor,"w",encoding="utf-8") as file:
            json.dump(
                serializable,
                file,
                ensure_ascii=ensure_ascii,
                indent=indent,
            )
            file.flush()
            os.fsync(file.fileno())
            
        os.fsync(file.fileno())
        
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    
    return output_path

def load_json(
    input_path: str | Path,
    *,
    default: Any = _MISSING,
) -> Any:
    """读取 JSON 文件。文件不存在且提供 default 时返回 default。"""
    input_path = Path(input_path)
    
    if not input_path.exists():
        if default is not _MISSING:
            return default
        raise FileNotFoundError(f"JSON文件不存在：{input_path}")
    
    with input_path.open("r",encoding="utf-8") as file:
        return json.load(file)
    
def save_jsonl(
    records: Iterable[Any],
    output_path: str | Path,
    *,
    ensure_ascii: bool=False,
) -> Path:
    """覆盖写入 JSONL；每一行保存一个 JSON 对象。"""
    output_path = ensure_parent_dir(output_path)
    
    with output_path.open("w",encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    to_jsonable(record),
                    ensure_ascii=ensure_ascii,
                )
                +"\n"
            )
    
    return output_path

def append_jsonl(
    record: Any,
    output_path: str | Path,
    *,
    ensure_ascii: bool = False,
) -> Path:
    """向 JSONL 文件末尾追加一条记录。"""
    output_path = ensure_parent_dir(output_path)
    
    with output_path.open("w",encoding="utf-8") as file:
        file.write(
            json.dumps(
                to_jsonable(record),
                ensure_ascii=ensure_ascii,
            )
            +"\n"
        )
        
    return output_path

def iter_jsonl(
    input_path: str | Path,
    *,
    skip_invalid: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Iterator[dict[str, Any]]:
    """
    逐行迭代 JSONL，适合大型文件，避免一次加载到内存。
    """     
    input_path = Path(input_path)
    active_logger = logger or LOGGER
    
    if not input_path.exists():
        return
    
    with input_path.open("r",encoding="utf-8") as file:
        for line_number , line in enumerate(file,start=1):
            line = line.strip()
            
            if not line:
                continue
                
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    active_logger.warning(
                        "忽略损坏的 JSONL 行 %s:%d：%s",
                        input_path,
                        line_number,
                        exc,
                    )
                    continue
                    
                raise ValueError(
                    f"JSONL 解析失败：{input_path}:{line_number}，{exc}"
                ) from exc
            
            if not instance(record,dict):
                if skip_invalid:
                    active_logger.warning(
                        "忽略非 JSON 对象行 %s:%d",
                        input_path,
                        line_number,
                    )
                    continue

                raise ValueError(
                    f"JSONL 每一行必须是对象：{input_path}:{line_number}"
                )
                
            yield record
            
def load_jsonl(
    input_path: str | Path,
    *,
    skip_invalid: bool = False,
    logger: Optional[logging.Logger] = None,
) -> list[dict[str,Any]]:
    """将 JSONL 全部读取为字典列表。大文件优先使用 iter_jsonl。"""
    return list(
        iter_jsonl(
            input_path,
            skip_invalid=skip_invalid,
            logger=logger,
        )
    )

def compute_file_sha256(
    file_path: str | Path,
    *,
    chunk_size: int = 1024*1024
) -> str:
    """分块计算文件 SHA-256，适合大文件。"""
    file_path = Path(file_path)
    
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    
    if not file_path.is_file():
        raise FileNotError(f"文件不存在:{file_path}")
        
    digest = hashlib.sha256()
    
    with file_path.open("rb") as file:
        for chunk in iter(lambda:file.read(chunk_size),b""):
            digest.update(chunk)
            
    return digest.hexdigest()

def extract_json_object(
    raw_text: str,
    *,
    repair_chinese_quotes: bool = True,
) -> dict[str,Any]:
    """
    从 LLM 输出中提取第一个完整 JSON 对象。

    可处理常见的 ```json ... ``` 包裹和中文双引号。
    不使用 eval，避免执行不可信文本。
    """
    text = str(raw_text).strip()
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)
    
    start = text.find("{")
    end = text.rfind("}")
    
    if start < 0 or end < start:
        raise ValueError("文本中没有找到 JSON 对象")
        
    candidate = text[start : end + 1]
    
    if repair_chinese_quotes:
        candidate = (
            candidate
            .replace("“", '"')
            .replace("”", '"')
        )
        
    try:
        result = json,loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc

    if not isinstance(result, dict):
        raise ValueError("提取结果不是 JSON 对象")

    return result
        
        

    