"""反推数据契约：从视频到 IR 的中间表示。

放在 ``core`` 而不是 ``app``：领域层（反推、交叉校验、置信度）要用这些类型，
不能让领域依赖具体实现。``app/providers`` 只负责生产它们。

分层对应 PRD §6.1.3：

    L0 信号层   VideoInfo / ShotBoundary / MotionEstimate / FramingEstimate …
    L1 视觉理解 ShotUnderstanding
    L2 语义归纳 SemanticSummary
    （L3 结构化输出 = 直接产出 Framework IR，见 framework.py）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .types import (
    BlinkBeat,
    Genre,
    Provenance,
    QualityPreset,
    RefType,
    ShotMode,
    Span,
)


# ==========================================================================
# L0 信号层
# ==========================================================================
# ==========================================================================
# 非位置字段
# ==========================================================================
NON_POSITION_KEYS: frozenset[str] = frozenset({
    "framework_version",
    "transcript",
})
"""IR 顶层里**不是可编辑位置**的键。

- ``framework_version``：契约版本标记，只读元数据；
- ``transcript``：ASR 原始产物，属于证据而非提示词内容。

位置表、字数统计、闸门都要跳过它们 —— 否则用户会在"位置标注"里
看到一个既不可编辑、也不是内容版本的字段，纯噪声。
"""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """ffprobe 的产物。**真实时长唯一可信来源**（PRD §6.1.6 致命级维度）。"""

    duration_sec: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str = ""
    audio_codec: str = ""
    path: str = ""

    @property
    def aspect_ratio(self) -> str:
        if self.height == 0:
            return "9:16"
        ratio = self.width / self.height
        if ratio < 0.7:
            return "9:16"
        if ratio > 1.4:
            return "16:9"
        return "1:1"

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_sec": round(self.duration_sec, 3),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "has_audio": self.has_audio,
            "codec": self.codec,
            "audio_codec": self.audio_codec,
            "aspect_ratio": self.aspect_ratio,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class ShotBoundary:
    """一个镜头的边界。切点误差目标 ≤±2 帧（PRD §6.1.2）。"""
    index: int
    start_ms: int
    end_ms: int

    @property
    def span(self) -> Span:
        return Span(self.start_ms, self.end_ms)

    @property
    def duration_sec(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_sec": round(self.duration_sec, 3),
        }


@dataclass(frozen=True, slots=True)
class MotionEstimate:
    """运镜判断（光流分析）。目标准确率 ≥90%。"""

    kind: str  # fixed / push_in / pan / track / handheld / unknown
    confidence: float
    magnitude: float = 0.0
    divergence: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "magnitude": round(self.magnitude, 4),
            "divergence": round(self.divergence, 4),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class FramingEstimate:
    """景别判断（人体框占画面比例）。目标准确率 ≥85%。"""

    kind: str  # 全景 / 中全景 / 中景 / 中近景 / 近景 / 脸部特写 / unknown
    confidence: float
    person_ratio: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "person_ratio": round(self.person_ratio, 4),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class BlockingEstimate:
    """站位判断（检测框中心 x vs 画面中线）。目标准确率 ≥95%。"""

    subject: str
    side: str  # left / right / center
    x_ratio: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "side": self.side,
            "x_ratio": round(self.x_ratio, 4),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True, slots=True)
class ShotSignals:
    """单个镜头的全部 L0 信号。"""

    boundary: ShotBoundary
    motion: MotionEstimate | None = None
    framing: FramingEstimate | None = None
    people_count: int = 0
    people_confidence: float = 0.0
    blocking: tuple[BlockingEstimate, ...] = ()
    keyframe: str = ""
    """该镜头的代表帧图片路径（base64 或本地路径），供 L1 视觉理解使用。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary.to_dict(),
            "motion": self.motion.to_dict() if self.motion else None,
            "framing": self.framing.to_dict() if self.framing else None,
            "people_count": self.people_count,
            "people_confidence": round(self.people_confidence, 3),
            "blocking": [b.to_dict() for b in self.blocking],
            "keyframe": self.keyframe,
        }


# ==========================================================================
# L1 视觉理解
# ==========================================================================
@dataclass(frozen=True, slots=True)
class ShotUnderstanding:
    """多模态模型对单个镜头的理解。**必须与 CV 交叉校验**（PRD F1.6）。"""

    shot_index: int
    description: str = ""
    framing: str = ""
    camera: str = ""
    location: str = ""
    expression: str = ""
    emotion: str = ""
    blocking: str = ""
    people: tuple[str, ...] = ()
    confidence: float = 0.55
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_index": self.shot_index,
            "description": self.description,
            "framing": self.framing,
            "camera": self.camera,
            "location": self.location,
            "expression": self.expression,
            "emotion": self.emotion,
            "blocking": self.blocking,
            "people": list(self.people),
            "confidence": round(self.confidence, 3),
        }


# ==========================================================================
# L2 语义归纳
# ==========================================================================
@dataclass(frozen=True, slots=True)
class SemanticSummary:
    """文本/多模态模型对全片的归纳。"""

    logline: str = ""
    scene_type: str = ""
    genre: Genre = Genre.DIALOGUE
    shot_mode: ShotMode = ShotMode.ONE_TAKE
    hook_index: int = 0
    """钩子节拍所在的镜头号 —— LOCK 位，必须保留（G11）。"""
    quality_preset: QualityPreset = QualityPreset.P1_PHONE
    blink_beat: BlinkBeat = BlinkBeat.FIVE
    confidence: float = 0.5
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "logline": self.logline,
            "scene_type": self.scene_type,
            "genre": self.genre.value,
            "shot_mode": self.shot_mode.value,
            "hook_index": self.hook_index,
            "quality_preset": self.quality_preset.value,
            "blink_beat": self.blink_beat.value,
            "confidence": round(self.confidence, 3),
            "note": self.note,
        }


# ==========================================================================
# L3 输入：参考物料与台词
# ==========================================================================
@dataclass(frozen=True, slots=True)
class PlaceholderRef:
    """从样本提示词里抽出的占位符。"""

    placeholder: str
    ref_type: RefType
    slots: tuple[str, ...] = ()
    prep_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "placeholder": self.placeholder,
            "type": self.ref_type.value,
            "slots": list(self.slots),
            "prep_hint": self.prep_hint,
        }


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """ASR 词级时间戳片段。

    ``has_timestamps`` 必须如实标注，这是**区分"知道"与"估的"** 的开关：

    - ``True``：时间戳来自 ASR 的词/句对齐，可以按重叠把台词分配到镜头；
    - ``False``：区间只是"音频总时长"推导出来的，只能说明"这段时间在说话"，
      不能拿来当对齐依据。

    少了这个标记，整段文本会因为"有区间"而被当成精确时间戳，
    结果同一句台词被复制到所有重叠镜头 —— 实测踩过这个坑。
    """

    text: str
    span: Span
    speaker: str = ""
    confidence: float = 0.85
    has_timestamps: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "span": self.span.to_dict(),
            "speaker": self.speaker,
            "confidence": round(self.confidence, 3),
            "has_timestamps": self.has_timestamps,
        }


# ==========================================================================
# 反推结果
# ==========================================================================
@dataclass(slots=True)
class InferenceReport:
    """一次反推的完整产物。UI 与回归测试都读它。"""

    video: VideoInfo
    shots: tuple[ShotSignals, ...] = ()
    understandings: tuple[ShotUnderstanding, ...] = ()
    semantic: SemanticSummary | None = None
    transcript: tuple[TranscriptSegment, ...] = ()
    provenance: dict[str, Provenance] = field(default_factory=dict)
    conflicts: tuple["Conflict", ...] = ()
    degradations: tuple[str, ...] = ()
    """反推失败时的降级说明。**绝不输出假数据**（PRD F1.12）。"""
    model_id: str = ""
    """模型版本记录（PRD F1.11），可锁定与回滚。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "shots": [s.to_dict() for s in self.shots],
            "understandings": [u.to_dict() for u in self.understandings],
            "semantic": self.semantic.to_dict() if self.semantic else None,
            "transcript": [t.to_dict() for t in self.transcript],
            "provenance": {k: v.to_dict() for k, v in self.provenance.items()},
            "conflicts": [c.to_dict() for c in self.conflicts],
            "degradations": list(self.degradations),
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class Conflict:
    """模型判断与 CV 计算的冲突。**冲突时以 CV 为准并降低置信度**（PRD §6.1.6）。"""

    field: str
    shot_index: int
    cv_value: str
    model_value: str
    resolution: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "shot_index": self.shot_index,
            "cv_value": self.cv_value,
            "model_value": self.model_value,
            "resolution": self.resolution,
            "confidence": round(self.confidence, 3),
        }
