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

    def analyze_batch(self,image_paths: list[str | Path]) -> list[dict[str,Any]]:
        """
        批量分析多张表情包。
        一张图片对应一个独立 conversation，
        最终返回一个和 image_paths 等长的结果列表。
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
                        "image": path.as_uri(),
                    },
                    {
                        "type": "text",
                        "text": STICKER_PROMPT,
                    },
                ],
                }
            ]
            batch_messages.append(messages)
        texts = [
            self.processor.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
                                               for messages in batch_messages
        ]    
        image_inputs,video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

        inputs = inputs.to(
            self.model.device
        )

        # --------------------------------
        # 5. GPU 一次生成整个 batch
        # --------------------------------

        with torch.inference_mode():

            generated_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=(
                        self.max_new_tokens
                    ),
                    do_sample=False,
                )
            )

        # --------------------------------
        # 6. 去掉输入 prompt 的 token
        # --------------------------------

        generated_ids_trimmed = [
            output_ids[
                len(input_ids):
            ]
            for input_ids, output_ids
            in zip(
                inputs.input_ids,
                generated_ids,
            )
        ]

        # --------------------------------
        # 7. 一次 decode 整个 batch
        # --------------------------------

        output_texts = (
            self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        # --------------------------------
        # 8. 每个输出分别解析 JSON
        # --------------------------------

        results = [
            _parse_model_output(
                output_text
            )
            for output_text
            in output_texts
        ]

        return results

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
                            unique_stickers: list[dict[str,Any]],
                            threshold: int) -> Optional[dict[str,Any]]:

    best_match = None
    best_distance = None

    for sticker in unique_stickers:
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
#切分列表
def _iter_batches(
    items: list,
    batch_size: int,
):
    """
    将列表按 batch_size 切成小批次。
    """

    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start:
            start + batch_size
        ]

def build_sticker_index(
    df: DataFrame,
    message_type_col: str = "message_type",
    sticker_path_col: str = "sticker_path",
    phash_threshold: int = 5,
    project_root: str | Path | None = None,
) -> tuple[
    DataFrame,
    list[dict[str, Any]],
]:
    """
    第一阶段：建立表情包索引。

    完成：
    1. 筛选 sticker 消息；
    2. 检查表情包路径；
    3. 计算 pHash；
    4. 根据 pHash 去重；
    5. 为唯一表情包分配 sticker_id；
    6. 将 sticker_id 回填到每一条表情包消息；
    7. 统计每个表情包出现次数。

    Returns
    -------
    result:
        原聊天 DataFrame 的副本，
        新增 sticker_id 列。

    unique_stickers:
        去重后的唯一表情包列表。
    """

    # --------------------------------------------------
    # 1. 检查必要字段
    # --------------------------------------------------

    required_columns = {
        message_type_col,
        sticker_path_col,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "缺少必要列："
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------
    # 2. 不直接修改原始 df
    # --------------------------------------------------

    result = df.copy()

    # --------------------------------------------------
    # 3. 创建 sticker_id 列
    # --------------------------------------------------

    if "sticker_id" not in result.columns:
        result["sticker_id"] = None

    # --------------------------------------------------
    # 4. 确定项目根目录
    # --------------------------------------------------

    if project_root is None:
        root = Path.cwd()
    else:
        root = Path(project_root).resolve()

    # --------------------------------------------------
    # 5. 只筛选表情包消息
    # --------------------------------------------------

    sticker_rows = result[
        result[message_type_col]
        == "sticker"
    ]

    print(
        f"表情包消息总数：{len(sticker_rows)}"
    )

    # --------------------------------------------------
    # 6. 保存唯一表情包
    # --------------------------------------------------

    unique_stickers: list[
        dict[str, Any]
    ] = []

    sticker_counter = 1

    # --------------------------------------------------
    # 7. 遍历所有表情包消息
    # --------------------------------------------------

    for index, row in sticker_rows.iterrows():

        raw_path = row[
            sticker_path_col
        ]

        # 路径为空
        if pd.isna(raw_path):
            print(
                f"[WARNING] index={index} "
                "表情包路径为空"
            )
            continue

        relative_path = str(
            raw_path
        ).strip()

        if not relative_path:
            print(
                f"[WARNING] index={index} "
                "表情包路径为空字符串"
            )
            continue

        # --------------------------------------------------
        # 8. 构造真正用于访问图片的绝对路径
        # --------------------------------------------------

        path = Path(relative_path)

        if not path.is_absolute():
            absolute_path = (
                root / path
            ).resolve()
        else:
            absolute_path = (
                path.resolve()
            )

        if not absolute_path.exists():
            print(
                f"[WARNING] 文件不存在："
                f"{absolute_path}"
            )
            continue

        # --------------------------------------------------
        # 9. 计算当前图片 pHash
        # --------------------------------------------------

        current_phash = compute_phash(
            absolute_path
        )

        # --------------------------------------------------
        # 10. 查找是否已经存在相同/相似表情包
        # --------------------------------------------------

        duplicate = (
            _find_duplicate_sticker(
                current_phash=(
                    current_phash
                ),
                unique_stickers=(
                    unique_stickers
                ),
                threshold=(
                    phash_threshold
                ),
            )
        )

        # --------------------------------------------------
        # 11. 如果是重复表情包
        # --------------------------------------------------

        if duplicate is not None:

            duplicate[
                "usage_count"
            ] += 1

            result.at[
                index,
                "sticker_id",
            ] = duplicate[
                "sticker_id"
            ]

            continue

        # --------------------------------------------------
        # 12. 如果是全新的表情包
        # --------------------------------------------------

        sticker_id = (
            f"sticker_"
            f"{sticker_counter:06d}"
        )

        sticker = {
            "sticker_id": sticker_id,

            # 保存原始/相对路径，
            # 不保存机器绑定的绝对路径
            "file_path": relative_path,

            "phash": current_phash,

            "usage_count": 1,
        }

        unique_stickers.append(
            sticker
        )

        # 当前聊天消息和 sticker_id 建立映射
        result.at[
            index,
            "sticker_id",
        ] = sticker_id

        sticker_counter += 1

    # --------------------------------------------------
    # 13. 输出统计信息
    # --------------------------------------------------

    print(
        "表情包去重后数量："
        f"{len(unique_stickers)}"
    )

    return result,unique_stickers,

#批量处理所有表情包
def process_stickers(
        df: DataFrame,
        analyzer: QwenStickerAnalyzer,
        message_type_col: str = "message_type",
        sticker_path_col: str = "sticker_path",
        phash_threshold: int = 5,
        batch_size: int = 4,
        project_root: str | Path | None = None
) -> tuple[DataFrame,list[dict[str,Any]]]:

    #第一阶段：pHash去重并建立sticker_id映射
    result,known_stickers = build_sticker_index(
        df=df,
        message_type_col=message_type_col,
        sticker_path_col=sticker_path_col,
        phash_threshold=phash_threshold,
        project_root=project_root
    )

    if "sticker_caption" not in result.columns:
        result["sticker_caption"] = None

    total = len(known_stickers)

    if total == 0:
        print("没有需要分析的表情包")
        return result,known_stickers

    print(f"开始批量分析表情包，共{total}个唯一表情包，batch_size={batch_size}")

    if project_root is None:
        root = Path.cwd()
    else:
        root = Path(project_root).resolve()

    #第二阶段：按照batch_size批量调用Qwen2.5-VL
    for batch_number,batch_stickers in enumerate(_iter_batches(known_stickers,batch_size),start=1):

        batch_paths = []

        for sticker in batch_stickers:
            path = Path(sticker["file_path"])

            if not path.is_absolute():
                path = (root/path).resolve()
            else:
                path = path.resolve()

            batch_paths.append(path)

        print(f"正在处理第{batch_number}批，本批{len(batch_paths)}个表情包")

        try:
            batch_analyses = analyzer.analyze_batch(batch_paths)

            if len(batch_analyses) != len(batch_stickers):
                raise ValueError(
                    f"批量推理结果数量不一致：输入{len(batch_stickers)}张图片，返回{len(batch_analyses)}个结果"
                )

        except Exception as exc:
            print(f"[WARNING]第{batch_number}批分析失败：{exc}")

            batch_analyses = [
                {
                    "visual_description":"",
                    "emotion":[],
                    "intent":[],
                    "tone":"",
                    "caption":""
                }
                for _ in batch_stickers
            ]

        #将每张图片的分析结果写回对应metadata
        for sticker,analysis in zip(batch_stickers,batch_analyses):

            caption = _build_sticker_caption(analysis)

            sticker["visual_description"] = analysis.get("visual_description","")
            sticker["emotion"] = analysis.get("emotion",[])
            sticker["intent"] = analysis.get("intent",[])
            sticker["tone"] = analysis.get("tone","")
            sticker["caption"] = caption
            sticker["resolved"] = caption not in {"[表情包:待处理]","[表情包：待处理]"}

    #第三阶段：根据sticker_id将caption回填到每一条聊天消息
    sticker_caption_map = {
        sticker["sticker_id"]:sticker["caption"]
        for sticker in known_stickers
    }

    sticker_mask = result["sticker_id"].notna()

    result.loc[sticker_mask,"sticker_caption"] = (
        result.loc[sticker_mask,"sticker_id"].map(sticker_caption_map)
    )

    print(f"表情包批量分析完成，共处理{len(known_stickers)}个唯一表情包")

    return result,known_stickers

def save_sticker_metadata(metadata:list[dict[str,Any]],
                          output_path: str | Path) -> None:
    """
    保存成 JSONL。
    """

    path = Path(output_path)

    save_jsonl(metadata,path)


    







