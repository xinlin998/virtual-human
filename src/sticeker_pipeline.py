"""
表情包处理流水线
功能：
1. 从 DataFrame 中提取所有表情包消息。
2. 使用感知哈希去重相似图片。
3. 对每个唯一表情包执行 OCR 和视觉大模型描述。
4. 生成统一文本描述，并填充回 DataFrame。
5. 保存 sticker_metadata.jsonl 用于后续检索和部署。
"""
import json
from pathlib import Path
from typing import Any, Optional
from src.utils import save_jsonl
import imagehash
import pandas as pd
import torch
from pandas import DataFrame
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

STICKER_PROMPT = """
你正在分析一张私人聊天中的表情包。

请综合分析图片本身的视觉内容和图片中的文字。

要求：

1. 描述图片中的主要人物、动物或物体；
2. 描述它们的动作、表情和状态；
3. 判断表情包表达的主要情绪；
4. 判断它在私人聊天中的常见使用意图；
5. 判断整体语气；
6. 生成一条简短自然的表情包描述，用于聊天模型训练。

注意：

- 不要解释你的分析过程；
- 不要输出 Markdown；
- 不要使用代码块；
- 只返回合法 JSON；
- emotion 和 intent 必须返回字符串列表；
- 如果图片没有文字，visible_text 返回空字符串；
- caption 中不要自己加入 “[表情包:]” 前缀。

返回格式：

{
  "visual_description": "图片主要视觉内容",
  "emotion": ["情绪1", "情绪2"],
  "intent": ["意图1", "意图2"],
  "tone": "整体语气",
  "caption": "适合放入训练数据中的自然语言表情包描述"
}
""".strip()

#1、Qwen2.5-VL 表情包分析器
class QwenStickerAnalyzer:
     """
    使用 Qwen2.5-VL 分析聊天表情包。

    一个 Analyzer 对象只加载一次模型，
    后续可以连续分析多张表情包。
    """
    def __init__(self,model_name: str,model_dir: str,max_new_tokens: int = 512) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        print(f"正在加载视觉模型:{model_name}")

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                cache_dir=model_dir
            )
        )
        self.processor =  (
            AutoProcessor.from_pretrained(model_name)
        )   

        self.model.eval()

        print("视觉模型加载完成")

    def analyze(self,image_path: str | Path) -> dict[str,Any]:
        """
        分析一张表情包。

        返回字段：
        - visual_description
        - emotion
        - intent
        - tone
        - caption
        """
        path = Path(image_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"表情包不存在:{path}")

        messages = [
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                        "image":path.as_uri(),
                    },
                    {
                        "type":"text",
                        "text":STICKER_PROMPT
                    }
                ]
            }
        ]

        prompt_text = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

        image_inputs,video_inputs = process_vision_info(messages)

        inputs = self.processor(text=[prompt_text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensores="pt")

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs,
                                                max_new_tokens=self.max_new_tokens,
                                                do_sample=False)

        generated_ids = generated_ids[:,inputs.input_ids.shape[1]:]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return _parse_model_output(output_text)

     #2、解析Qwen输出
    def _parse_model_output(output_text: str) -> dict[str,Any]:
        """
        将视觉模型返回文本解析成字典。
        """
        text = output_text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise ValueError(f"视觉模型没有返回合法JSON:\n{text}")

            json_text = text[start:end + 1]

            try:
                data = json.loads(json_text)

            except json.JSONDecodeError as exc:
                raise ValueError(f"无法解析视觉模型JSON:\n{text}")

        if not isinstance(data,dict):
            raise TypeError("视觉模型输出必须是JSON对象")

        return _normalize_analysis(data)

     #标准化视觉模型输出
    def _normalize_analysis(data: dict[str,Any]) -> dict[str,Any]:
        visual_description = str(data.get("visual_description","")).strip()
        tone = str(data.get("tone","")).strip()
        caption = str(data.get("caption","")).strip()
        emotion = data.get("visual_description",[])
        intent = data.get("intent",[])

        if not isinstance(emotion,list):
            emotion = [str(emotion)]

        if not isinstance(intent,list):
            intent = [str(intent)]

        emotion = [str(item).strip() for item in emotion if str(item).strip()]

        intent = [str(item).strip() for item in intent if str(item).strip()]

        return {
            "visual_description":visual_description,
            "emotion":emotion,
            "intent":intent,
            "tone":tone,
            "caption":caption
        }

#计算phash
def compute_phash(image_path: str | Path) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"表情包不存在:{path}")

    with Image.open(path) as iamge:
        image = image.convert("RGB")
        phash = imagehash.phash(image)

    return str(phash)

#计算phash距离
def phash_distance(hash_a: str,hash_b: str) -> int:

    phash_a = imagehash.hex_to_hash(hash_a)
    phash_b = imagehash.hex_to_hash(hash_b)

    return phash_a - phash_b

#查找重复表情包
def _find_duplicate_sticker(current_phash: str,
                            known_stickers: list[dict[str,Any]],
                            threshold: int) -> Optional[dict[str,Any]]:

    best_match = None
    best_distance = None

    for sticker in known_stickers:
        distance = phash_distance(current_phash,sticker['phash'])

        if distance > threshold:
            continue

        if best_distance is None or distance < best_distance:
            best_match = sticker
            best_distance = distance

    return best_match

#构造训练caption
def _build_sticker_caption(analysis: dict[str,Any]) -> str:

    caption = str(analysis.get("caption","")).strip()

    if caption:
        if caption.startswith("[表情包"):
            return caption

        return f"[表情包:{caption}]"

    visual_description = str(analysis.get("visual_descriptin","")).strip()

    emotion = analysis.get("emotion",[])

    tone = str(analysis.get("tone","")).strip()

    parts: list[str] = []

    if visual_description:
        parts.append(visual_description)

    if emotion:
        emotion_text = '、'.join(str(item) for item in emotion)

        parts.append(f"表达{emotion_text}")

    if tone:
        parts.append(f"带有{tone}语气")

    if not parts:
        return "[表情包:待处理]"

    content = ','.join(parts)

    return f"[表情包：{content}]"

#构造metadata
def _build_sticker_metadata(
        sticker_id: str,
        image_path: str,
        phash: str,
        analyzer: QwenStickerAnalyzer
) -> dict[str,Any]:
    
    analysis = analyzer.analyze(image_path)

    caption = _build_sticker_caption(analysis)

    resolved = caption != "[表情包：待处理]"

    return {
        "sticker_id":sticker_id,
        "file_path":image_path,
        "phash":phash,
        "visual_description":analysis["visual_description"],
        "emotion":analysis["emotion"],
        "intent":analysis["intent"],
        "tone":analysis["tone"],
        "caption":caption,
        "resolved":resolved,
        "usage_count":1
    }

#批量处理所有表情包
def process_stickers(
        df: DataFrame,
        analyzer: QwenStickerAnalyzer,
        message_type_col: str = "message_type",
        sticker_path_col: str = "sticker_path",
        phash_threshold: int = 5
) -> tuple[DataFrame,list[dict[str,Any]]]:

    required_columns = {message_type_col,sticker_path_col}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"缺少必要列：\n{sorted(missing_columns)}")

    result = df.copy()

    if "sticker_id" not in result.columns:
        result["sticker_id"] = None

    if "sticker_caption" not in result.columns:
        result["sticker_caption"] = None

    known_stickers: list[dict[str,Any]] = []

    sticker_counter = 1

    sticker_rows = result[result[message_type_col] == "sticker"]

    total = len(sticker_rows)
    print(f"待处理表情包消息：{total}")

    for number , (index,row) in enumerate(sticker_rows.iterrows(),start=1):
        raw_path = row[sticker_path_col]

        if pd.isna(raw_path):
            continue

        image_path = str(raw_path).strip()

        if not image_path:
            continue

        path = Path(image_path)

        path = path.parent.parent/"stickers"/path.name

        if not path.exists():
            print(f"[WARNIGNG]文件不存在：{path}")
            continue

        current_phash = compute_phash(path)

        duplicate = _find_duplicate_sticker(current_phash=current_phash,known_stickers=known_stickers,threshold=phash_threshold)

        if duplicate is not None:
            duplicate["usage_count"] += 1

            result.at[index,"sticker_id"] = duplicate['sticker_id']

            result.at[index,'sticker_caption'] = duplicate['caption']

            continue

        sticker_id = f"sticker_{sticker_counter:06d}"

        print(f"[{number}/{total}]分析新表情包：{sticker_id}")

        try:
            metadata = _build_sticker_metadata(
                sticker_id=sticker_id,
                image_path=path,
                phash=current_phash,
                analyzer=analyzer
            )
        except Exception as exc:
            print(f"[WARNING]{sticker_id}分析失败：{exc}")

            metadata = {
                "sticker_id":sticker_id,
                'file_path':image_path,
                'phash':current_phash,
                'visual_description':"",
                'emotion':[],
                'intent':[],
                'tone':"",
                'caption':'[表情包：待处理]',
                'resolved':False,
                'usage_count':1
            }

        known_stickers.append(metadata)

        result.at[index,'sticker_id'] = sticker_id
        result.at[index,'sticker_caption'] = metadata['caption']

        sticker_counter += 1

    print(f"表情包去重后数量：{len(known_stickers)}")

    return result , known_stickers

def save_sticker_metadata(metadata:list[dict[str,Any]],
                          output_path: str | Path) -> None:
    """
    保存成 JSONL。
    """

    path = Path(output_path)

    save_jsonl(metadata,path)


    







