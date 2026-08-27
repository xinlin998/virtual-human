"""
表情包处理流水线

功能：
1. 从 DataFrame 中提取所有表情包消息。
2. 根据 CSV 中的原始路径提取表情包文件名。
3. 将表情包定位到项目中的 data/stickers 目录。
4. 使用 pHash 对相似表情包进行去重。
5. 为唯一表情包建立 sticker_id。
6. 使用 Qwen2.5-VL 批量分析唯一表情包。
7. 生成统一 caption，并回填到 DataFrame。
8. 保存 sticker_metadata.jsonl。
"""

import json
from pathlib import Path
from typing import Any, Optional

import imagehash
import pandas as pd
import torch
from pandas import DataFrame
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.utils import save_jsonl


STICKER_PROMPT = """
你正在分析一张私人聊天中的表情包。

请综合分析图片本身的视觉内容和图片中的文字。

要求：

1. 描述图片中的主要人物、动物或物体；
2. 描述它们的动作、表情和状态；
3. 如果图片中存在文字，请结合文字理解表情包含义；
4. 判断表情包表达的主要情绪；
5. 判断它在私人聊天中的常见使用意图；
6. 判断整体语气；
7. 生成一条简短自然的表情包描述，用于聊天模型训练。

注意：

- 不要解释分析过程；
- 不要输出 Markdown；
- 不要使用代码块；
- 只返回合法 JSON；
- emotion 和 intent 必须返回字符串列表；
- caption 中不要加入 “[表情包:]” 前缀。

返回格式：

{
  "visual_description": "图片主要视觉内容以及必要的图片文字信息",
  "emotion": ["情绪1", "情绪2"],
  "intent": ["意图1", "意图2"],
  "tone": "整体语气",
  "caption": "适合放入训练数据中的自然语言表情包描述"
}
""".strip()


# Qwen2.5-VL表情包分析器
class QwenStickerAnalyzer:
    def __init__(
        self,
        model_name: str,
        model_dir: str | Path,
        max_new_tokens: int = 256
    ) -> None:
        self.model_name = model_name
        self.model_dir = str(model_dir)
        self.max_new_tokens = max_new_tokens

        print(f"正在加载视觉模型：{model_name}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=self.model_dir
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=self.model_dir
        )

        self.model.eval()

        print("视觉模型加载完成")

    def analyze(self, image_path: str | Path) -> dict[str, Any]:
        results = self.analyze_batch([image_path])
        return results[0]

    def analyze_batch(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        """
        批量分析表情包。

        每张图片对应一个独立 conversation，
        返回结果顺序与 image_paths 保持一致。
        """
        if not image_paths:
            return []

        batch_messages = []

        for image_path in image_paths:
            path = Path(image_path).resolve()

            if not path.exists():
                raise FileNotFoundError(f"表情包不存在：{path}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": path.as_uri()
                        },
                        {
                            "type": "text",
                            "text": STICKER_PROMPT
                        }
                    ]
                }
            ]

            batch_messages.append(messages)

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        image_inputs, video_inputs = process_vision_info(batch_messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return [
            _parse_model_output(output_text)
            for output_text in output_texts
        ]


# 解析Qwen输出
def _parse_model_output(output_text: str) -> dict[str, Any]:
    text = output_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end < start:
            raise ValueError(f"视觉模型没有返回合法JSON：\n{text}")

        json_text = text[start:end + 1]

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析视觉模型JSON：\n{text}") from exc

    if not isinstance(data, dict):
        raise TypeError("视觉模型输出必须是JSON对象")

    return _normalize_analysis(data)


# 标准化Qwen输出
def _normalize_analysis(data: dict[str, Any]) -> dict[str, Any]:
    visual_description = str(data.get("visual_description", "")).strip()
    tone = str(data.get("tone", "")).strip()
    caption = str(data.get("caption", "")).strip()
    emotion = data.get("emotion", [])
    intent = data.get("intent", [])

    if not isinstance(emotion, list):
        emotion = [str(emotion)]

    if not isinstance(intent, list):
        intent = [str(intent)]

    emotion = [
        str(item).strip()
        for item in emotion
        if str(item).strip()
    ]

    intent = [
        str(item).strip()
        for item in intent
        if str(item).strip()
    ]

    return {
        "visual_description": visual_description,
        "emotion": emotion,
        "intent": intent,
        "tone": tone,
        "caption": caption
    }


# 解析表情包真实路径
def _resolve_sticker_path(
        raw_path: str | Path,
        image_root: str | Path = "data/stickers"
) -> Path:
    """
    将聊天CSV中的表情包路径映射到项目实际存储目录。
    """
    normalized_path = str(raw_path).strip().replace("\\", "/")
    file_name = Path(normalized_path).name

    if not file_name:
        raise ValueError(f"无法从路径中提取表情包文件名：{raw_path}")

    return (Path(image_root) / file_name).resolve()


# 计算pHash
def compute_phash(image_path: str | Path) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"表情包不存在：{path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        phash = imagehash.phash(image)

    return str(phash)


# 计算pHash距离
def phash_distance(hash_a: str, hash_b: str) -> int:
    phash_a = imagehash.hex_to_hash(hash_a)
    phash_b = imagehash.hex_to_hash(hash_b)

    return phash_a - phash_b


# 查找重复表情包
def _find_duplicate_sticker(
        current_phash: str,
        unique_stickers: list[dict[str, Any]],
        threshold: int
) -> Optional[dict[str, Any]]:
    best_match = None
    best_distance = None

    for sticker in unique_stickers:
        distance = phash_distance(current_phash, sticker["phash"])

        if distance > threshold:
            continue

        if best_distance is None or distance < best_distance:
            best_match = sticker
            best_distance = distance

    return best_match


# 构造训练caption
def _build_sticker_caption(analysis: dict[str, Any]) -> str:
    caption = str(analysis.get("caption", "")).strip()

    if caption:
        if caption.startswith("[表情包"):
            return caption

        return f"[表情包:{caption}]"

    visual_description = str(
        analysis.get("visual_description", "")
    ).strip()

    emotion = analysis.get("emotion", [])
    tone = str(analysis.get("tone", "")).strip()

    parts: list[str] = []

    if visual_description:
        parts.append(visual_description)

    if emotion:
        emotion_text = "、".join(
            str(item)
            for item in emotion
        )
        parts.append(f"表达{emotion_text}")

    if tone:
        parts.append(f"带有{tone}语气")

    if not parts:
        return "[表情包:待处理]"

    content = "，".join(parts)

    return f"[表情包:{content}]"


# 将列表切分成batch
def _iter_batches(
        items: list,
        batch_size: int
):
    if batch_size <= 0:
        raise ValueError("batch_size必须大于0")

    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


# 第一阶段：pHash去重并建立sticker_id
def build_sticker_index(
        df: DataFrame,
        message_type_col: str = "message_type",
        sticker_path_col: str = "sticker_path",
        phash_threshold: int = 5,
        image_root: str | Path = "data/stickers"
) -> tuple[DataFrame, list[dict[str, Any]]]:
    required_columns = {
        message_type_col,
        sticker_path_col
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"缺少必要列：{sorted(missing_columns)}")

    result = df.copy()

    if "sticker_id" not in result.columns:
        result["sticker_id"] = None

    sticker_rows = result[
        result[message_type_col] == "sticker"
    ]

    print(f"表情包消息总数：{len(sticker_rows)}")

    unique_stickers: list[dict[str, Any]] = []
    sticker_counter = 1

    for index, row in sticker_rows.iterrows():
        raw_path = row[sticker_path_col]

        if pd.isna(raw_path):
            print(f"[WARNING] index={index} 表情包路径为空")
            continue

        image_path = str(raw_path).strip()

        if not image_path:
            print(f"[WARNING] index={index} 表情包路径为空字符串")
            continue

        try:
            absolute_path = _resolve_sticker_path(
                raw_path=image_path,
                image_root=image_root
            )
        except ValueError as exc:
            print(f"[WARNING] index={index} 路径解析失败：{exc}")
            continue

        if not absolute_path.exists():
            print(f"[WARNING] 文件不存在：{absolute_path}")
            continue

        try:
            current_phash = compute_phash(absolute_path)
        except Exception as exc:
            print(f"[WARNING] pHash计算失败：{absolute_path}，原因：{exc}")
            continue

        duplicate = _find_duplicate_sticker(
            current_phash=current_phash,
            unique_stickers=unique_stickers,
            threshold=phash_threshold
        )

        if duplicate is not None:
            duplicate["usage_count"] += 1
            result.at[index, "sticker_id"] = duplicate["sticker_id"]
            continue

        sticker_id = f"sticker_{sticker_counter:06d}"

        stored_path = (
            Path(image_root)
            / absolute_path.name
        ).as_posix()

        sticker = {
            "sticker_id": sticker_id,
            "file_path": stored_path,
            "phash": current_phash,
            "usage_count": 1
        }

        unique_stickers.append(sticker)

        result.at[index, "sticker_id"] = sticker_id

        sticker_counter += 1

    print(f"表情包去重后数量：{len(unique_stickers)}")

    return result, unique_stickers


# 第二阶段：Qwen批量分析并回填caption
def process_stickers(
        df: DataFrame,
        analyzer: QwenStickerAnalyzer,
        message_type_col: str = "message_type",
        sticker_path_col: str = "sticker_path",
        phash_threshold: int = 5,
        batch_size: int = 4,
        image_root: str | Path = "data/stickers"
) -> tuple[DataFrame, list[dict[str, Any]]]:

    result, known_stickers = build_sticker_index(
        df=df,
        message_type_col=message_type_col,
        sticker_path_col=sticker_path_col,
        phash_threshold=phash_threshold,
        image_root=image_root
    )

    if "sticker_caption" not in result.columns:
        result["sticker_caption"] = None

    total = len(known_stickers)

    if total == 0:
        print("没有需要分析的表情包")
        return result, known_stickers

    print(
        f"开始批量分析表情包，共{total}个唯一表情包，"
        f"batch_size={batch_size}"
    )

    for batch_number, batch_stickers in enumerate(
        _iter_batches(known_stickers, batch_size),
        start=1
    ):
        batch_paths = [
            Path(sticker["file_path"]).resolve()
            for sticker in batch_stickers
        ]

        print(
            f"正在处理第{batch_number}批，"
            f"本批{len(batch_paths)}个表情包"
        )

        try:
            batch_analyses = analyzer.analyze_batch(batch_paths)

            if len(batch_analyses) != len(batch_stickers):
                raise ValueError(
                    f"批量推理结果数量不一致："
                    f"输入{len(batch_stickers)}张图片，"
                    f"返回{len(batch_analyses)}个结果"
                )

        except Exception as exc:
            print(
                f"[WARNING] 第{batch_number}批分析失败：{exc}"
            )

            batch_analyses = [
                {
                    "visual_description": "",
                    "emotion": [],
                    "intent": [],
                    "tone": "",
                    "caption": ""
                }
                for _ in batch_stickers
            ]

        for sticker, analysis in zip(
            batch_stickers,
            batch_analyses
        ):
            caption = _build_sticker_caption(analysis)

            sticker["visual_description"] = analysis.get(
                "visual_description",
                ""
            )

            sticker["emotion"] = analysis.get(
                "emotion",
                []
            )

            sticker["intent"] = analysis.get(
                "intent",
                []
            )

            sticker["tone"] = analysis.get(
                "tone",
                ""
            )

            sticker["caption"] = caption

            sticker["resolved"] = (
                caption != "[表情包:待处理]"
            )

    sticker_caption_map = {
        sticker["sticker_id"]: sticker["caption"]
        for sticker in known_stickers
    }

    sticker_mask = result["sticker_id"].notna()

    result.loc[
        sticker_mask,
        "sticker_caption"
    ] = (
        result.loc[
            sticker_mask,
            "sticker_id"
        ].map(sticker_caption_map)
    )

    print(
        f"表情包批量分析完成，"
        f"共处理{len(known_stickers)}个唯一表情包"
    )

    return result, known_stickers


# 保存表情包metadata
def save_sticker_metadata(
        metadata: list[dict[str, Any]],
        output_path: str | Path
) -> None:
    save_jsonl(metadata, output_path)