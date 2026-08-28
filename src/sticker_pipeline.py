"""表情包去重、视觉分析、质量校验与结果回填。"""
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import imagehash
import pandas as pd
import torch
from pandas import DataFrame
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.utils import save_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PENDING_STICKER_CAPTION = "[表情包:待处理]"
EMOTION_LABELS = [
    "开心","兴奋","惊讶","困惑","无奈","尴尬","生气","不满","悲伤","委屈",
    "害羞","期待","疲惫","冷漠","紧张","害怕","得意","嫌弃","平静","亲昵"
]
INTENT_LABELS = [
    "回应","赞同","拒绝","吐槽","调侃","安慰","鼓励","感谢","道歉","撒娇",
    "表达喜欢","表达不满","表达惊讶","表达疑惑","引起注意","结束话题","缓和气氛","催促","打招呼","告别"
]
TONE_LABELS = ["轻松","幽默","调侃","讽刺","亲昵","温柔","冷淡","认真","生气","委屈"]
EMOTION_ALIASES = {
    "快乐":"开心","高兴":"开心","喜悦":"开心","愉快":"开心","激动":"兴奋","惊喜":"惊讶",
    "疑惑":"困惑","疑问":"困惑","无语":"无奈","愤怒":"生气","恼火":"生气","伤心":"悲伤",
    "难过":"悲伤","恐惧":"害怕","鄙视":"嫌弃","嫌恶":"嫌弃","淡定":"平静"
}
INTENT_ALIASES = {
    "回复":"回应","回应对方":"回应","肯定":"赞同","表示赞同":"赞同","开玩笑":"调侃","玩笑":"调侃",
    "抱怨":"吐槽","表达抱怨":"吐槽","表达惊喜":"表达惊讶","表达困惑":"表达疑惑","询问":"表达疑惑",
    "吸引注意":"引起注意","提醒":"引起注意","活跃气氛":"缓和气氛","结束聊天":"结束话题","再见":"告别"
}
TONE_ALIASES = {
    "轻松愉快":"轻松","诙谐":"幽默","搞笑":"幽默","开玩笑":"调侃","反讽":"讽刺",
    "温馨":"温柔","柔和":"温柔","严肃":"认真","正式":"认真"
}
PLACEHOLDER_TERMS = {
    "情绪1","情绪2","意图1","意图2","em1","em2","emotion1","emotion2","emotions1",
    "简洁自然","简洁的","图片细节描述","待处理"
}


def _build_sticker_prompt(is_gif: bool = False, retry: bool = False) -> str:
    gif_instruction = "\n输入的多张图片来自同一个GIF表情包的不同时间帧，请结合动作变化理解完整含义，不要逐帧分别描述。" if is_gif else ""
    retry_instruction = "\n这是一次质量重试。请重新检查图片，严格使用给定标签，不要返回占位词、空字段或格式之外的内容。" if retry else ""
    return f"""
你正在分析一张私人聊天中的表情包。{gif_instruction}
请结合画面主体、动作、表情以及图片中可见文字理解表情包语义。
要求：
1. visual_description：客观描述主体、动作、表情和必要的图片文字，不要自由发挥网络梗；
2. emotion：只能从以下标签中选择1～2个，不得创建新标签：{'、'.join(EMOTION_LABELS)}；
3. intent：只能从以下标签中选择1～2个，不得创建新标签：{'、'.join(INTENT_LABELS)}；
4. tone：只能从以下标签中选择1个：{'、'.join(TONE_LABELS)}；
5. caption：15～40个汉字，采用“主体/动作 + 核心情绪或聊天含义”的客观描述；
6. caption不要使用“这张图片”“这个表情包”等开头，不要使用疑问句、emoji、无必要的“可能/似乎”，不要加入“[表情包:]”前缀。
注意：
- 不要解释分析过程；
- 不要输出Markdown或代码块；
- 只返回合法JSON；
- emotion和intent必须返回字符串列表。{retry_instruction}
返回格式：
{{
  "visual_description":"图片主要视觉内容以及必要的图片文字信息",
  "emotion":["标签1","标签2"],
  "intent":["标签1","标签2"],
  "tone":"标签",
  "caption":"适合训练和检索的简洁描述"
}}
""".strip()


class QwenStickerAnalyzer:
    def __init__(
        self,
        model_name: str,
        model_dir: str | Path,
        max_new_tokens: int = 256,
        gif_max_frames: int = 4
    ) -> None:
        self.model_name = model_name
        self.model_dir = str(model_dir)
        self.max_new_tokens = max_new_tokens
        self.gif_max_frames = gif_max_frames
        print(f"正在加载视觉模型：{model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=self.model_dir
        )
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=self.model_dir)
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model.eval()
        print("视觉模型加载完成")

    def analyze(self, image_path: str | Path, retry: bool = False) -> dict[str, Any]:
        return self.analyze_batch([image_path], retry=retry)[0]

    def analyze_batch(self, image_paths: list[str | Path], retry: bool = False) -> list[dict[str, Any]]:
        if not image_paths:
            return []
        batch_messages = []
        frame_counts = []
        media_types = []
        with TemporaryDirectory(prefix="sticker_frames_") as temp_dir:
            for item_index,image_path in enumerate(image_paths):
                path = Path(image_path).resolve()
                if not path.exists():
                    raise FileNotFoundError(f"表情包不存在：{path}")
                messages,frame_count,media_type = _build_sticker_messages(
                    path=path,
                    temp_dir=Path(temp_dir),
                    gif_max_frames=self.gif_max_frames,
                    retry=retry,
                    item_index=item_index
                )
                batch_messages.append(messages)
                frame_counts.append(frame_count)
                media_types.append(media_type)
            texts = [
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in batch_messages
            ]
            image_inputs,video_inputs = process_vision_info(batch_messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            generated_ids_trimmed = [
                output_ids[len(input_ids):]
                for input_ids,output_ids in zip(inputs.input_ids,generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        results = []
        for index,output_text in enumerate(output_texts):
            try:
                analysis = _parse_model_output(output_text)
                analysis["_parse_error"] = ""
            except Exception as exc:
                analysis = _empty_analysis(str(exc))
            analysis["_frame_count"] = frame_counts[index]
            analysis["_media_type"] = media_types[index]
            results.append(analysis)
        return results


def _build_sticker_messages(
        path: Path,
        temp_dir: Path,
        gif_max_frames: int,
        retry: bool,
        item_index: int
) -> tuple[list[dict[str, Any]],int,str]:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        frame_paths = _extract_gif_frames(path, temp_dir, gif_max_frames, item_index)
        image_items = [{"type":"image","image":frame_path.as_uri()} for frame_path in frame_paths]
        prompt = _build_sticker_prompt(is_gif=True, retry=retry)
        frame_count = len(frame_paths)
        media_type = "gif"
    else:
        image_items = [{"type":"image","image":path.as_uri()}]
        prompt = _build_sticker_prompt(is_gif=False, retry=retry)
        frame_count = 1
        media_type = suffix.lstrip(".") or "image"
    return [{"role":"user","content":[*image_items,{"type":"text","text":prompt}]}],frame_count,media_type


def _extract_gif_frames(
        path: Path,
        output_dir: Path,
        max_frames: int,
        item_index: int
) -> list[Path]:
    if max_frames <= 0:
        raise ValueError("gif_max_frames必须大于0")
    frame_paths = []
    with Image.open(path) as image:
        total_frames = max(1, int(getattr(image, "n_frames", 1)))
        frame_count = min(max_frames, total_frames)
        if frame_count == 1:
            indices = [0]
        else:
            indices = sorted({round(i * (total_frames - 1) / (frame_count - 1)) for i in range(frame_count)})
        for output_index,frame_index in enumerate(indices):
            image.seek(frame_index)
            frame = image.convert("RGB")
            frame_path = output_dir / f"item_{item_index:03d}_frame_{output_index:02d}.png"
            frame.save(frame_path, format="PNG")
            frame_paths.append(frame_path.resolve())
    return frame_paths


def _parse_model_output(output_text: str) -> dict[str, Any]:
    text = output_text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"视觉模型没有返回合法JSON：{text}")
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析视觉模型JSON：{text}") from exc
    if not isinstance(data, dict):
        raise TypeError("视觉模型输出必须是JSON对象")
    return _normalize_analysis(data)


def _normalize_analysis(data: dict[str, Any]) -> dict[str, Any]:
    visual_description = _clean_text(data.get("visual_description", ""))
    caption = _clean_caption(data.get("caption", ""))
    emotion = _canonicalize_labels(data.get("emotion", []), EMOTION_LABELS, EMOTION_ALIASES, max_items=2)
    intent = _canonicalize_labels(data.get("intent", []), INTENT_LABELS, INTENT_ALIASES, max_items=2)
    tone = _canonicalize_tone(data.get("tone", ""))
    return {
        "visual_description":visual_description,
        "emotion":emotion,
        "intent":intent,
        "tone":tone,
        "caption":caption
    }


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_caption(value: object) -> str:
    caption = _clean_text(value)
    caption = re.sub(r"^\[表情包[:：]?\s*", "", caption)
    caption = re.sub(r"\]$", "", caption).strip()
    caption = re.sub(r"^(这张图片|这个表情包|该表情包)[，,:：\s]*", "", caption)
    return caption


def _canonicalize_labels(
        value: object,
        allowed: list[str],
        aliases: dict[str, str],
        max_items: int
) -> list[str]:
    items = value if isinstance(value, list) else [value]
    result = []
    allowed_set = set(allowed)
    for item in items:
        label = _clean_text(item)
        label = aliases.get(label, label)
        if label in allowed_set and label not in result:
            result.append(label)
        if len(result) >= max_items:
            break
    return result


def _canonicalize_tone(value: object) -> str:
    tone = _clean_text(value)
    tone = TONE_ALIASES.get(tone, tone)
    return tone if tone in TONE_LABELS else ""


def _empty_analysis(error: str = "") -> dict[str, Any]:
    return {
        "visual_description":"",
        "emotion":[],
        "intent":[],
        "tone":"",
        "caption":"",
        "_parse_error":error,
        "_frame_count":1,
        "_media_type":""
    }


def _evaluate_analysis(
        analysis: dict[str, Any],
        min_quality_score: float
) -> tuple[bool,float,list[str]]:
    visual_description = _clean_text(analysis.get("visual_description", ""))
    caption = _clean_caption(analysis.get("caption", ""))
    emotion = analysis.get("emotion", [])
    intent = analysis.get("intent", [])
    tone = _clean_text(analysis.get("tone", ""))
    errors = []
    score = 0.0
    if visual_description:
        score += 0.25
    else:
        errors.append("visual_description为空")
    if caption:
        score += 0.30
    else:
        errors.append("caption为空")
    if emotion:
        score += 0.15
    else:
        errors.append("emotion无有效标签")
    if intent:
        score += 0.15
    else:
        errors.append("intent无有效标签")
    if tone:
        score += 0.10
    placeholder_found = any(term.lower() in caption.lower() for term in PLACEHOLDER_TERMS)
    if placeholder_found:
        errors.append("caption包含占位内容")
    else:
        score += 0.05
    if caption and not 6 <= len(caption) <= 60:
        errors.append("caption长度不合理")
    parse_error = _clean_text(analysis.get("_parse_error", ""))
    if parse_error:
        errors.append(f"模型输出解析失败：{parse_error}")
    score = round(score, 2)
    resolved = not errors and score >= min_quality_score
    return resolved,score,errors


def _build_sticker_caption(analysis: dict[str, Any]) -> str:
    caption = _clean_caption(analysis.get("caption", ""))
    return f"[表情包:{caption}]" if caption else PENDING_STICKER_CAPTION


def _build_analysis_candidate(
        analysis: dict[str, Any],
        min_quality_score: float,
        analysis_mode: str,
        analysis_attempts: int
) -> dict[str, Any]:
    resolved,quality_score,validation_errors = _evaluate_analysis(analysis, min_quality_score)
    return {
        "visual_description":analysis.get("visual_description", ""),
        "emotion":analysis.get("emotion", []),
        "intent":analysis.get("intent", []),
        "tone":analysis.get("tone", ""),
        "caption":_build_sticker_caption(analysis) if resolved else PENDING_STICKER_CAPTION,
        "resolved":resolved,
        "quality_score":quality_score,
        "analysis_attempts":analysis_attempts,
        "analysis_mode":analysis_mode,
        "analysis_frame_count":int(analysis.get("_frame_count", 1)),
        "media_type":str(analysis.get("_media_type", "")),
        "analysis_error":str(analysis.get("_parse_error", "")),
        "validation_errors":validation_errors
    }


def _apply_candidate(sticker: dict[str, Any], candidate: dict[str, Any]) -> None:
    current_score = float(sticker.get("quality_score", -1.0))
    current_resolved = bool(sticker.get("resolved", False))
    should_replace = (
        candidate["resolved"] and not current_resolved
        or candidate["quality_score"] > current_score
        or "quality_score" not in sticker
    )
    attempts = max(int(sticker.get("analysis_attempts", 0)), int(candidate["analysis_attempts"]))
    if should_replace:
        sticker.update(candidate)
    sticker["analysis_attempts"] = attempts


def _resolve_sticker_path(
        raw_path: str | Path,
        image_root: str | Path = "data/stickers"
) -> Path | None:
    path_text = str(raw_path).strip()
    if not path_text or path_text.startswith(("http://", "https://")):
        return None
    file_name = Path(path_text.replace("\\", "/")).name
    if not file_name:
        return None
    root = Path(image_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return (root / file_name).resolve()


def _stored_sticker_path(absolute_path: Path, image_root: str | Path) -> str:
    root = Path(image_root)
    if not root.is_absolute():
        return (root / absolute_path.name).as_posix()
    try:
        return absolute_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return absolute_path.as_posix()


def compute_phash(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"表情包不存在：{path}")
    with Image.open(path) as image:
        phash = imagehash.phash(image.convert("RGB"))
    return str(phash)


def phash_distance(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)


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


def _iter_batches(items: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size必须大于0")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_sticker_index(
        df: DataFrame,
        message_type_col: str = "message_type",
        sticker_path_col: str = "sticker_path",
        phash_threshold: int = 5,
        image_root: str | Path = "data/stickers"
) -> tuple[DataFrame,list[dict[str, Any]]]:
    required_columns = {message_type_col, sticker_path_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要列：{sorted(missing_columns)}")
    result = df.copy()
    if "sticker_id" not in result.columns:
        result["sticker_id"] = None
    sticker_rows = result[result[message_type_col] == "sticker"]
    unique_stickers: list[dict[str, Any]] = []
    sticker_counter = 1
    valid_count = 0
    missing_path_count = 0
    remote_url_count = 0
    missing_file_count = 0
    phash_failed_count = 0
    print(f"表情包消息总数：{len(sticker_rows)}")
    for index,row in sticker_rows.iterrows():
        raw_path = row[sticker_path_col]
        if pd.isna(raw_path) or not str(raw_path).strip():
            missing_path_count += 1
            continue
        image_path = str(raw_path).strip()
        if image_path.startswith(("http://", "https://")):
            remote_url_count += 1
            continue
        absolute_path = _resolve_sticker_path(image_path, image_root)
        if absolute_path is None:
            missing_path_count += 1
            continue
        if not absolute_path.exists():
            missing_file_count += 1
            print(f"[WARNING] 文件不存在：{absolute_path}")
            continue
        try:
            current_phash = compute_phash(absolute_path)
        except Exception as exc:
            phash_failed_count += 1
            print(f"[WARNING] pHash计算失败：{absolute_path}，原因：{exc}")
            continue
        valid_count += 1
        stored_path = _stored_sticker_path(absolute_path, image_root)
        duplicate = _find_duplicate_sticker(current_phash, unique_stickers, phash_threshold)
        if duplicate is not None:
            duplicate["usage_count"] += 1
            if stored_path not in duplicate["aliases"]:
                duplicate["aliases"].append(stored_path)
            result.at[index, "sticker_id"] = duplicate["sticker_id"]
            continue
        sticker_id = f"sticker_{sticker_counter:06d}"
        unique_stickers.append({
            "sticker_id":sticker_id,
            "file_path":stored_path,
            "aliases":[stored_path],
            "phash":current_phash,
            "usage_count":1
        })
        result.at[index, "sticker_id"] = sticker_id
        sticker_counter += 1
    print(f"有效本地表情包消息：{valid_count}")
    print(f"空路径表情包消息：{missing_path_count}")
    print(f"远程URL表情包消息：{remote_url_count}")
    print(f"本地文件不存在：{missing_file_count}")
    print(f"pHash计算失败：{phash_failed_count}")
    print(f"表情包去重后数量：{len(unique_stickers)}")
    return result,unique_stickers


def process_stickers(
        df: DataFrame,
        analyzer: QwenStickerAnalyzer,
        message_type_col: str = "message_type",
        sticker_path_col: str = "sticker_path",
        phash_threshold: int = 5,
        batch_size: int = 4,
        gif_batch_size: int = 1,
        retry_failed: bool = True,
        min_quality_score: float = 0.85,
        image_root: str | Path = "data/stickers"
) -> tuple[DataFrame,list[dict[str, Any]]]:
    result,known_stickers = build_sticker_index(
        df=df,
        message_type_col=message_type_col,
        sticker_path_col=sticker_path_col,
        phash_threshold=phash_threshold,
        image_root=image_root
    )
    for column in ["sticker_caption", "sticker_quality_score", "sticker_needs_review"]:
        if column not in result.columns:
            result[column] = None
    if not known_stickers:
        print("没有需要分析的表情包")
        return result,known_stickers
    analysis_queue = sorted(known_stickers, key=lambda item: item["usage_count"], reverse=True)
    static_queue = [item for item in analysis_queue if Path(item["file_path"]).suffix.lower() != ".gif"]
    gif_queue = [item for item in analysis_queue if Path(item["file_path"]).suffix.lower() == ".gif"]

    def run_batches(queue: list[dict[str, Any]], current_batch_size: int, label: str) -> None:
        total_batches = (len(queue) + current_batch_size - 1) // current_batch_size if queue else 0
        for batch_number,batch_stickers in enumerate(_iter_batches(queue, current_batch_size), start=1):
            batch_paths = [_resolve_sticker_path(sticker["file_path"], image_root) for sticker in batch_stickers]
            if any(path is None for path in batch_paths):
                batch_analyses = [_empty_analysis("无法解析本地路径") for _ in batch_stickers]
            else:
                try:
                    batch_analyses = analyzer.analyze_batch([path for path in batch_paths if path is not None])
                except Exception as exc:
                    batch_analyses = [_empty_analysis(str(exc)) for _ in batch_stickers]
            print(f"{label}第{batch_number}/{total_batches}批，本批{len(batch_stickers)}个表情包")
            for sticker,analysis in zip(batch_stickers,batch_analyses):
                candidate = _build_analysis_candidate(
                    analysis=analysis,
                    min_quality_score=min_quality_score,
                    analysis_mode="batch",
                    analysis_attempts=1
                )
                _apply_candidate(sticker, candidate)

    run_batches(static_queue, batch_size, "静态图片")
    run_batches(gif_queue, gif_batch_size, "GIF")

    if retry_failed:
        retry_queue = sorted(
            [sticker for sticker in known_stickers if not sticker.get("resolved", False)],
            key=lambda item: item["usage_count"],
            reverse=True
        )
        print(f"首次分析未通过质量门控：{len(retry_queue)}个，开始单张严格重试")
        for retry_index,sticker in enumerate(retry_queue, start=1):
            path = _resolve_sticker_path(sticker["file_path"], image_root)
            if path is None:
                continue
            try:
                analysis = analyzer.analyze(path, retry=True)
            except Exception as exc:
                analysis = _empty_analysis(str(exc))
            candidate = _build_analysis_candidate(
                analysis=analysis,
                min_quality_score=min_quality_score,
                analysis_mode="single_retry",
                analysis_attempts=2
            )
            _apply_candidate(sticker, candidate)
            if retry_index % 20 == 0 or retry_index == len(retry_queue):
                print(f"严格重试进度：{retry_index}/{len(retry_queue)}")

    for sticker in known_stickers:
        if "quality_score" not in sticker:
            sticker.update(_build_analysis_candidate(_empty_analysis("未执行分析"), min_quality_score, "failed", 0))
        sticker["needs_review"] = not sticker["resolved"] or sticker["quality_score"] < 0.95
        risk = 2.0 if not sticker["resolved"] else max(0.0, 1.0 - sticker["quality_score"])
        sticker["review_priority"] = round(float(sticker["usage_count"]) * risk, 2)

    caption_map = {item["sticker_id"]:item["caption"] for item in known_stickers}
    quality_map = {item["sticker_id"]:item["quality_score"] for item in known_stickers}
    review_map = {item["sticker_id"]:item["needs_review"] for item in known_stickers}
    sticker_mask = result["sticker_id"].notna()
    result.loc[sticker_mask, "sticker_caption"] = result.loc[sticker_mask, "sticker_id"].map(caption_map)
    result.loc[sticker_mask, "sticker_quality_score"] = result.loc[sticker_mask, "sticker_id"].map(quality_map)
    result.loc[sticker_mask, "sticker_needs_review"] = result.loc[sticker_mask, "sticker_id"].map(review_map)
    resolved_count = sum(bool(item["resolved"]) for item in known_stickers)
    weighted_total = sum(int(item["usage_count"]) for item in known_stickers)
    weighted_resolved = sum(int(item["usage_count"]) for item in known_stickers if item["resolved"])
    weighted_rate = weighted_resolved / weighted_total if weighted_total else 0.0
    print(f"唯一表情包解析成功：{resolved_count}/{len(known_stickers)}")
    print(f"按使用次数加权的语义覆盖率：{weighted_rate:.2%}")
    return result,known_stickers


def save_sticker_metadata(metadata: list[dict[str, Any]], output_path: str | Path) -> None:
    save_jsonl(metadata, output_path)


def save_sticker_review_queue(metadata: list[dict[str, Any]], output_path: str | Path) -> None:
    review_items = sorted(
        [item for item in metadata if item.get("needs_review", False)],
        key=lambda item: float(item.get("review_priority", 0.0)),
        reverse=True
    )
    save_jsonl(review_items, output_path)
