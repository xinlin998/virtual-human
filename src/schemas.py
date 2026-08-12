from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class StickerMetadata:
    # —— 来自字典的字段（匹配 _build_sticker_metadata 返回的键）——
    sticker_id: str
    file_path: str
    phash: str
    visual_description: str
    visible_text: str               
    emotion: List[str] = field(default_factory=list)
    intent: List[str] = field(default_factory=list)
    tone: str = ""
    caption: str = ""
    resolved: bool = False          
    usage_count: int = 0           


@dataclass
class StickerMessageResult:
    message_id: str
    sticker_id: Optional[str]
    sticker_path: str
    sticker_caption: str
    status: str

    is_duplicate: bool = False
    duplicate_method: Optional[str] = None
    error: Optional[str] = None
    
@dataclass
class Message:
    message_id: str
    timestamp: str
    datatime: str
    speaker: str
    message_type: str
    text: str
    raw_index: str
    sticker_id: Optional[str] = None
    sticker_path: Optional[str] = None
    emotion: Optional[List[str]] = None
    intent: Optional[List[str]] = None

@dataclass
class Turn:
    turn_id: str
    session_id: Optional[str]
    speaker: str
    start_timestamp: int
    end_timestamp: int
    message_ids: List[str]
    messages: List[str]
    merged_text: str
    
@dataclass
class AtomicUnit:
    unit_id: str
    turn_id: str
    message_id: str
    speaker: str
    timestamp: int
    text: str

    dialogue_act: Optional[str]
    implicit_question: bool
    answer_type: Optional[str]
    entities: dict = field(default_factory=dict)
    
@dataclass
class ReplyEdge:
    source_unit_id: str
    target_unit_id: Optional[str]
    relation: str
    score: float
    confidence: str