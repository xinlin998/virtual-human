from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StickerMetadata:
    sticker_id: str
    file_path: str
    phash: str
    visual_description: str = ""
    aliases: list[str] = field(default_factory=list)
    emotion: list[str] = field(default_factory=list)
    intent: list[str] = field(default_factory=list)
    tone: str = ""
    caption: str = ""
    resolved: bool = False
    usage_count: int = 0
    quality_score: float = 0.0
    analysis_attempts: int = 0
    analysis_mode: str = ""
    analysis_frame_count: int = 1
    media_type: str = ""
    analysis_error: str = ""
    validation_errors: list[str] = field(default_factory=list)
    needs_review: bool = True
    review_priority: float = 0.0


@dataclass
class Message:
    message_id: str
    timestamp: int
    datetime: str
    speaker: str
    message_type: str
    text: str
    raw_index: str
    sticker_id: Optional[str] = None
    sticker_path: Optional[str] = None
    emotion: Optional[list[str]] = None
    intent: Optional[list[str]] = None


@dataclass
class Turn:
    turn_id: str
    session_id: Optional[str]
    speaker: str
    start_timestamp: int
    end_timestamp: int
    message_ids: list[str]
    messages: list[str]
    merged_text: str


@dataclass
class AtomicUnit:
    unit_id: str
    turn_id: str
    message_id: str
    speaker: str
    timestamp: int
    text: str
    dialogue_act: Optional[str] = None
    implicit_question: bool = False
    answer_type: Optional[str] = None
    entities: dict = field(default_factory=dict)


@dataclass
class ReplyEdge:
    source_unit_id: str
    target_unit_id: Optional[str]
    relation: str
    score: float
    confidence: str
