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
    mean_volume_db: float | None = None
    """音轨的平均响度（dBFS，ffmpeg ``volumedetect`` 量测）。

    **``None`` 表示没量到**（没有音轨，或量测失败），不表示静音。
    静音是 −91 这种具体数值 —— 二者必须分清，否则"量不到"会被当成
    "没声音"，把一条真有台词的片子判成哑片。
    """

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

    @property
    def has_speech(self) -> bool:
        """音轨里**是否可能有说话声** —— 决定要不要跑 ASR。

        为什么不能只看 ``has_audio``（实测踩过）：容器里挂一条纯静音的音轨，
        ffprobe 报 ``has_audio=True``，于是照常调 ASR；而 ASR 网关在静音上
        会凭空气编出一整段「0, 1, 2, …, 100.」当成转写结果，
        最后被当作真实台词写进交付物，还派生出音色参考句。
        那条片子的实测平均响度是 **−91.0 dBFS**（数字静音），
        而正常人声在 −30 ~ −12 dBFS 量级。
        """
        if not self.has_audio:
            return False
        if self.mean_volume_db is None:
            return True          # 量不到就不擅自判哑，交给 ASR 与输出兜底
        return self.mean_volume_db > SILENCE_DBFS

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_sec": round(self.duration_sec, 3),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "has_audio": self.has_audio,
            "has_speech": self.has_speech,
            "mean_volume_db": self.mean_volume_db,
            "codec": self.codec,
            "audio_codec": self.audio_codec,
            "aspect_ratio": self.aspect_ratio,
            "path": self.path,
        }


SILENCE_DBFS = -50.0
"""低于这个平均响度就按"没有声音"处理。

取值依据：数字静音实测 −91 dBFS，环境底噪通常 −60 以下，
正常人声 −30 ~ −12 dBFS。−50 落在底噪与人声之间，两边都留了余量。
"""


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
    composition: str = ""
    """整个画面的构图偏侧（left/center/right）。

    与 ``blocking``（逐个人的左右）**不是一回事**：双人镜头里两人一左一右时，
    逐人看是 left+right，构图看是 center。用户要的"站位居中不居中"是后者。
    """
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
            "composition": self.composition,
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
    camera_angle: str = ""
    """拍摄角度（平视/俯视/仰视/过肩…）。与 ``camera``（运镜）是两件事：
    角度回答"从哪个方向拍"，运镜回答"镜头怎么动"。"""
    camera: str = ""
    location: str = ""
    expression: str = ""
    emotion: str = ""
    blocking: str = ""
    people: tuple[str, ...] = ()
    subject: str = ""
    """**这个镜头的主体人物**（用户 2026-10-07 要求）。

    多人带练类视频里，主体就是**带练的老师**：位于画面最中间、最显眼、
    正在做示范的那位。它和其他人是**两种东西** ——
    老师要给完整的人物一致性约束（多视图、五官锁定），
    学员只是"跟着做"的群体，由生成模型自己生成即可。

    ``people`` 是"画面里都有谁"，``subject`` 是"**谁是这个镜头的主角**"。
    主体为空表示这个镜头没有明确主角（空镜、群像）。
    """
    group: str = ""
    """**非主体的人作为一个群体**的描述：年龄段 + 穿着概况。

    用户原话：「其他学员直接由 Seedance 自己生成，只需要给她们定位成学员们，
    然后穿着、年龄定位好，动作跟着老师做就行了。**这样就不需要那么多的提示词**。」
    """
    confidence: float = 0.55
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_index": self.shot_index,
            "description": self.description,
            "framing": self.framing,
            "camera_angle": self.camera_angle,
            "camera": self.camera,
            "location": self.location,
            "expression": self.expression,
            "emotion": self.emotion,
            "blocking": self.blocking,
            "people": list(self.people),
            "subject": self.subject,
            "group": self.group,
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
    content_kind: str = ""
    """**有没有需要复刻的身体动作**（``action``/``speech``；空=模型没judge出来）。

    决定交付物里出不出现【动作与物理】。空值不是默认成某一类，而是
    **交给兜底判据**（``to_ir`` 按画面里的动作词判）—— 用户 2026-10-07 反馈：
    口播片里出现了"保持动作轨迹与幅度"这种无对象约束，而动作片缺 1:1 复刻句。
    """
    people_structure: Mapping[str, Any] = field(default_factory=dict)
    """**模型对"这条片子里的人怎么分工"的判断**（用户 2026-10-08 要求）。

    为什么要它：三次"一堆人物参考"都是**我在代码里用关键词/硬规则猜结构**猜错的
    （模型写「其余数人」就不命中我的"人群词"）。既然已经用了多模态理解，
    结构判断就该由模型给出：

        {"lead": 主角, "second": 第二位要锁定的人, "distinct_others": [独立配角],
         "crowd_label": 群像称呼, "crowd_look": 年龄+穿着, "crowd_behavior": 跟随/旁观/各做各的}

    空字典 = 模型没给（老数据或它没答），此时**退回代码兜底**并如实说明是推断的。
    """
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
            "content_kind": self.content_kind,
            "people_structure": dict(self.people_structure),
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
# L2.5 台词归属
# ==========================================================================
@dataclass(frozen=True, slots=True)
class ShotDialogueContext:
    """判"这句谁说的"时，这个镜头能提供的证据。

    只给**画面里能看见的人**（``people``）与镜头时长 ——
    不给整段转写，避免模型忽略画面证据、纯按语感瞎分。

    ``action`` / ``expression`` 是**"谁在说话"最直接的画面证据**：
    视觉理解经常直接写着「紫衣女子张口说话」。实测踩过（留出集那条框架外片子）：
    这两个字段一开始没进上下文，于是 4 句台词只判出第 1 句的说话人 ——
    画面明明把答案写出来了，归属环节却看不到。
    """

    shot_index: int
    duration_sec: float
    framing: str = ""
    camera_angle: str = ""
    location: str = ""
    people: tuple[str, ...] = ()
    action: str = ""
    expression: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_index": self.shot_index,
            "duration_sec": round(self.duration_sec, 2),
            "framing": self.framing,
            "camera_angle": self.camera_angle,
            "location": self.location,
            "people": list(self.people),
            "action": self.action,
            "expression": self.expression,
        }


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """一句台词，带**它属于哪个镜头**与**是谁说的**。

    ``speaker`` 用画面里的稳定特征（"紫色背心女子"），
    这样用户在前面的「人物」段里能直接对上号。
    """

    shot_index: int
    speaker: str
    text: str
    tone: str = ""
    confidence: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_index": self.shot_index,
            "speaker": self.speaker,
            "text": self.text,
            "tone": self.tone,
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True, slots=True)
class CritiqueEdit:
    """自检回路提出的一处修改。

    **模型只能提"字段级修改"，不能重写整篇**：路径由代码按契约校验
    （``review.apply_edits``），越界的一律拒绝并记账。
    这样既保住"IR 是唯一事实来源"，也不给模型破坏永不可裁内容的机会。
    """

    path: str
    action: str = "set"     # set / clear
    value: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "action": self.action, "value": self.value,
                "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """自检回路里，模型要求"**再看一眼**"的一个具体请求。

    这是把管线变成"会调查的 agent"的关键一步：以前它看不懂就只能留空或降级，
    现在它可以点名——"这个时间点、这个区域，我要再看一次"。

    ``region`` 是裁剪放大用的预设区域（``full`` / ``left`` / ``right`` /
    ``center`` / ``top`` / ``bottom``）—— 小字、细节、手势都得放大才看得清。
    """

    question: str
    time_start: float
    time_end: float
    region: str = "full"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "time_start": round(self.time_start, 3),
            "time_end": round(self.time_end, 3),
            "region": self.region,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProbeRequest":
        def _seconds(key: str) -> float:
            try:
                return max(0.0, float(raw.get(key) or 0.0))
            except (TypeError, ValueError):
                return 0.0

        start = _seconds("time_start")
        end = _seconds("time_end")
        return cls(
            question=str(raw.get("question") or "").strip(),
            time_start=start,
            time_end=end if end >= start else start,
            region=str(raw.get("region") or "full").strip() or "full",
            reason=str(raw.get("reason") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class ProbeFinding:
    """取证结果：视觉模型看着抽出来的那几帧给出的回答。"""

    request: ProbeRequest
    answer: str
    frames: tuple[str, ...] = ()
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "answer": self.answer,
            "frames": list(self.frames),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True, slots=True)
class Extension:
    """**扩展段**：框架 13 个段落都没覆盖的信息（屏上文案、分屏、变装点…）。

    由自检回路提出（它在看交付物时最容易发现"这段信息没地方放"），
    由 ``core/review.apply_extensions`` 按四条纪律把关（证据/不重名/有上限/不写台词）。

    **写给用户看，不是写给我自己看**（用户 2026-10-08）：
    「一般都是描述补充场景后的信息，目前你给出的信息太复杂，看不懂，
     直接**大白话描述出 AI 读取的画面信息**，例如文字、画面排版、有什么东西，直接写出来；
     有什么没读取到的画面，**直接说什么原因**，然后给出你能读取到的，
     再**提示人工自己确认修改**。目前你给出的括号里的内容就不要了。」
    """

    title: str
    lines: tuple[str, ...] = ()
    evidence: str = ""
    unreadable: str = ""
    """没看清的部分 + **为什么**（如「屏幕下方还有一行小字，画面分辨率不够看不清」）。

    渲染成一句大白话，并在后面提示人工确认 —— 以前这种情况我们什么都不写，
    用户以为画面里就这些字。
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "lines": list(self.lines),
            "evidence": self.evidence,
            "unreadable": self.unreadable,
        }


@dataclass(frozen=True, slots=True)
class Question:
    """**结构化提问**：Agent 真的两难时，把选择题交给人。

    为什么要有它（用户确认的第 ⑤ 批）：现在低置信度只是进「待确认」列表 ——
    **被动**，而且是一句自由文本，用户看完还得自己想怎么改。
    改成选择题之后：① 选项就是能直接写进字段的值；② 用户点一下，
    答案**直接落进 IR**，交付物当场更新。

    纪律：**只有真的两难才问**（最多 2 条）。能取证解决的就去取证，
    问人不该是第一选择。
    """

    id: str
    path: str
    text: str
    options: tuple[str, ...] = ()
    reason: str = ""
    answer: str = ""

    @property
    def resolved(self) -> bool:
        return bool(str(self.answer).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "text": self.text,
            "options": list(self.options),
            "reason": self.reason,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Question":
        options = raw.get("options")
        return cls(
            id=str(raw.get("id") or ""),
            path=str(raw.get("path") or ""),
            text=str(raw.get("text") or ""),
            options=tuple(str(o) for o in (options if isinstance(options, list) else [])),
            reason=str(raw.get("reason") or ""),
            answer=str(raw.get("answer") or ""),
        )


@dataclass(frozen=True, slots=True)
class CritiqueOutcome:
    """一次自检的完整产出：**改什么** + **还要看什么** + **哪里有新东西** + **哪里两难**。

    四者放在一起是有意的，这正是"会调查的 Agent"的四种动作：
    能改就改、没把握就取证、框架装不下就扩展、真的两难就**问人**。
    """

    edits: tuple[CritiqueEdit, ...] = ()
    probes: tuple[ProbeRequest, ...] = ()
    extensions: tuple[Extension, ...] = ()
    questions: tuple[Question, ...] = ()

    @classmethod
    def of(
        cls,
        edits: Sequence[CritiqueEdit] = (),
        probes: Sequence[ProbeRequest] = (),
        extensions: Sequence[Extension] = (),
        questions: Sequence[Question] = (),
    ) -> "CritiqueOutcome":
        return cls(
            edits=tuple(edits), probes=tuple(probes),
            extensions=tuple(extensions), questions=tuple(questions),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edits": [edit.to_dict() for edit in self.edits],
            "probes": [probe.to_dict() for probe in self.probes],
            "extensions": [item.to_dict() for item in self.extensions],
            "questions": [item.to_dict() for item in self.questions],
        }


@dataclass(frozen=True, slots=True)
class ReviewNote:
    """自检回路的一条记账：改了什么 / 拒了什么 / 看了什么。**用户要看得见**。"""

    path: str
    action: str
    reason: str
    applied: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "action": self.action, "reason": self.reason,
                "applied": self.applied, "detail": self.detail}


# ==========================================================================
# 用户意见改写（PRD F8.4「按建议重写」）
# ==========================================================================
@dataclass(frozen=True, slots=True)
class RevisionOp:
    """改写助手提出的一步**字段级**修改。

    为什么与自检回路的 :class:`CritiqueEdit` 分开：**授权来源不同**。
    自检只能改"描述性文字"（它写错了），而改写的授权来自**用户本人** ——
    他可以说「换个场景」「加个人」「加一个镜头」，这些是**结构变动**，
    不是文本修补。所以这里多了 ``add`` / ``remove``，值也可以是对象
    （往数组里插一条人物 / 一条节拍）。

    但**门禁的形态不变**：路径仍由代码按契约校验，
    只是校验清单比自检宽（见 ``engine/core/revision.py`` 的 ``REVISABLE_ROOTS``），
    而真正的红线（编造素材绑定、改交付形态）照样拒绝并记账。
    """

    op: str = "set"          # set / clear / add / remove
    path: str = ""
    value: Any = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "path": self.path, "value": self.value,
                "reason": self.reason}


@dataclass(frozen=True, slots=True)
class RevisionRefusal:
    """一条**做不到**的用户意见，以及为什么。

    「做不到」必须说出来。这是本功能最容易变成欺骗的地方：
    模型改不了却返回空数组，界面上看着像"改好了"。
    """

    instruction: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"instruction": self.instruction, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class RevisionChange:
    """一处**已生效**的改动，给人看的形态（原句 → 新句）。"""

    path: str
    label: str = ""
    before: str = ""
    after: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "label": self.label, "before": self.before,
                "after": self.after, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class RevisionOutcome:
    """改写助手一次调用的完整产出：**改什么** + **新段落** + **哪条做不到**。"""

    ops: tuple[RevisionOp, ...] = ()
    extensions: tuple[Extension, ...] = ()
    refusals: tuple[RevisionRefusal, ...] = ()
    note: str = ""
    """模型对自己这次改写的一句话总结（直接给用户看）。"""

    @classmethod
    def of(
        cls,
        ops: Sequence[RevisionOp] = (),
        extensions: Sequence[Extension] = (),
        refusals: Sequence[RevisionRefusal] = (),
        note: str = "",
    ) -> "RevisionOutcome":
        return cls(ops=tuple(ops), extensions=tuple(extensions),
                   refusals=tuple(refusals), note=str(note or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ops": [item.to_dict() for item in self.ops],
            "extensions": [item.to_dict() for item in self.extensions],
            "refusals": [item.to_dict() for item in self.refusals],
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    """一轮改写的账（落盘 + 结果页展示）。

    **为什么必须落盘**：改完之后用户要能看到"它到底动了什么"、
    "我说的话哪条它没做到"，并且能撤销。第七轮的教训在这里同样适用 ——
    ``InferenceReport.to_dict()`` 曾经漏字段，导致记账在落盘时静默消失，
    看起来像"回路没发现问题"。所以这里的 ``to_dict`` / ``from_dict``
    **成对写**，并有回归测试守着。
    """

    id: str
    instruction: str
    note: str = ""
    model: str = ""
    created_at: float = 0.0
    seconds: float = 0.0
    changes: tuple[RevisionChange, ...] = ()
    notes: tuple[ReviewNote, ...] = ()
    """随改动一起要露给用户的记账：补齐了哪些素材位、
    这次改动是否让它偏离了原片的 1:1 复刻、交付判定有没有变差。"""
    rejected: tuple[ReviewNote, ...] = ()
    refusals: tuple[RevisionRefusal, ...] = ()
    verdict_before: str = ""
    verdict_after: str = ""
    ok: bool = True
    """本次是否真的写进了 IR。全被拒绝 / 模型没给出可用的改动时为 False。"""
    model_calls: tuple[Mapping[str, Any], ...] = ()
    """这一轮改写**实际调用了哪些模型**（用途/型号/耗时/token）。

    用户 2026-10-08 的要求是"只让 DeepSeek 改、不要再送去 MiMo 自检"——
    这条记账就是**证据**：界面上会写「共 1 次模型调用：按意见改写 · deepseek-flash」。
    与 ``InferenceReport.model_calls`` 同一套格式（`providers/mimo.record_call`）。
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "note": self.note,
            "model": self.model,
            "created_at": round(self.created_at, 3),
            "seconds": round(self.seconds, 1),
            "changes": [item.to_dict() for item in self.changes],
            "notes": [item.to_dict() for item in self.notes],
            "rejected": [item.to_dict() for item in self.rejected],
            "refusals": [item.to_dict() for item in self.refusals],
            "verdict_before": self.verdict_before,
            "verdict_after": self.verdict_after,
            "ok": self.ok,
            "model_calls": [dict(item) for item in self.model_calls],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RevisionRecord":
        def _review_notes(items: Any) -> tuple[ReviewNote, ...]:
            out: list[ReviewNote] = []
            for item in items or []:
                if not isinstance(item, Mapping):
                    continue
                out.append(ReviewNote(
                    path=str(item.get("path") or ""),
                    action=str(item.get("action") or ""),
                    reason=str(item.get("reason") or ""),
                    applied=bool(item.get("applied")),
                    detail=str(item.get("detail") or ""),
                ))
            return tuple(out)

        return cls(
            id=str(raw.get("id") or ""),
            instruction=str(raw.get("instruction") or ""),
            note=str(raw.get("note") or ""),
            model=str(raw.get("model") or ""),
            created_at=float(raw.get("created_at") or 0.0),
            seconds=float(raw.get("seconds") or 0.0),
            changes=tuple(
                RevisionChange(
                    path=str(item.get("path") or ""),
                    label=str(item.get("label") or ""),
                    before=str(item.get("before") or ""),
                    after=str(item.get("after") or ""),
                    reason=str(item.get("reason") or ""),
                )
                for item in (raw.get("changes") or [])
                if isinstance(item, Mapping)
            ),
            rejected=_review_notes(raw.get("rejected")),
            notes=_review_notes(raw.get("notes")),
            refusals=tuple(
                RevisionRefusal(
                    instruction=str(item.get("instruction") or ""),
                    reason=str(item.get("reason") or ""),
                )
                for item in (raw.get("refusals") or [])
                if isinstance(item, Mapping)
            ),
            verdict_before=str(raw.get("verdict_before") or ""),
            verdict_after=str(raw.get("verdict_after") or ""),
            ok=bool(raw.get("ok", True)),
            model_calls=tuple(
                dict(item) for item in (raw.get("model_calls") or [])
                if isinstance(item, Mapping)
            ),
        )


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
    dialogue_turns: tuple[DialogueTurn, ...] = ()
    """切分并归属好的逐句台词。**优先于 transcript 用于填节拍**。

    ASR 给不出说话人与时间戳时，``transcript`` 只有一整段；
    这一项是文本模型按画面证据切分后的结果（见 ``DialogueAttributor``）。
    """
    provenance: dict[str, Provenance] = field(default_factory=dict)
    conflicts: tuple["Conflict", ...] = ()
    questions: tuple["Question", ...] = ()
    """Agent 交给人裁决的选择题（最多 2 条）。答案由用户在结果页点选，
    经 ``PUT /field`` 的同一道门禁落进 IR。"""
    reviews: tuple[ReviewNote, ...] = ()
    """自检-重写回路的记账（改了哪些、拒了哪些、为什么）。

    这是"agent 看过自己写的东西"的证据 —— 没有它，用户无法判断
    交付物是被校验过的，还是流水线吐出来就直接给了。
    """
    degradations: tuple[str, ...] = ()
    """反推失败时的降级说明。**绝不输出假数据**（PRD F1.12）。"""
    timings: Mapping[str, float] = field(default_factory=dict)
    """逐阶段耗时（秒）—— 用户 2026-10-07 报"自检与重写卡很久"之后加的。

    没有它就只能靠猜"时间花在哪"（一次反推要调十几次模型，某一次调用最坏 9 分半）。
    现在最慢的阶段一眼可见。
    """
    model_calls: tuple[Mapping[str, Any], ...] = ()
    """**每一次**模型调用的记账（用途/型号/耗时/成败）。

    阶段计时只能到"阶段"，而"自检"这一段本身要走 2~4 次调用 ——
    实测它花了 563.7 秒，但看不出是"一次调用想了 4 分钟"还是"两次重试各 4 分钟"。
    有了这个，下次就能直接指着某一次调用说"就是它慢"。
    """
    model_id: str = ""
    """模型版本记录（PRD F1.11），可锁定与回滚。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "shots": [s.to_dict() for s in self.shots],
            "understandings": [u.to_dict() for u in self.understandings],
            "semantic": self.semantic.to_dict() if self.semantic else None,
            "transcript": [t.to_dict() for t in self.transcript],
            "dialogue_turns": [t.to_dict() for t in self.dialogue_turns],
            "provenance": {k: v.to_dict() for k, v in self.provenance.items()},
            "conflicts": [c.to_dict() for c in self.conflicts],
            "questions": [q.to_dict() for q in self.questions],
            "reviews": [note.to_dict() for note in self.reviews],
            "degradations": list(self.degradations),
            "timings": {k: round(v, 1) for k, v in self.timings.items()},
            "model_calls": [dict(item) for item in self.model_calls],
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
