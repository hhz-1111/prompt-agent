"""领域基础类型：枚举、字面量、坐标与置信度。

本模块是整个 engine 的最底层，**不依赖任何第三方库**。
所有上层模块（契约、短语、闸门、组装）都只从这里取类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

FRAMEWORK_VERSION = "mod-framework-v1"


class StrEnum(str, Enum):
    """字符串枚举：可直接与 JSON 互转，不引入额外依赖。"""

    def __str__(self) -> str:  # pragma: no cover - 便于日志
        return self.value


class Genre(StrEnum):
    """场景类型（框架 §0.2）。"""

    DIALOGUE = "dialogue"           # 对话类
    TEACHING = "teaching"           # 教学类
    STUDIO = "studio"               # 演播厅类
    STREET_INTERVIEW = "street_interview"  # 街采访谈类
    REPLACEMENT = "replacement"     # 替换复刻类


class ShotMode(StrEnum):
    """镜头模式（框架 §0.2）。决定拆条策略与提示词条数。"""

    ONE_TAKE = "one_take"                       # 一镜到底（固定机位）
    MULTI_SHOT = "multi_shot"                   # 多镜头切分
    SAME_ANGLE_SEGMENTS = "same_angle_segments"  # 分镜但同机位分段


class ContentKind(StrEnum):
    """**这条片子有没有需要复刻的身体动作**（用户 2026-10-07 提出）。

    为什么要单独判一次，而不是看 ``genre``（实测证据）：用户给的一条
    瑜伽带练视频（老师带学员做侧伸展）被模型归成了 ``dialogue``，
    而一条舞台口播被归成了 ``teaching`` —— **genre 是从"场景/题材"角度分的，
    和"有没有动作要复刻"根本不是一回事**。

    这一项直接决定交付物里出不出现【动作与物理】：

        ACTION  动作/体式/教学动作 —— 动作是观看价值的一部分，必须 1:1 复刻
        SPEECH  口播/访谈/剧情对话 —— 以说话为主，没有要复刻的身体动作

    用户原话：「不涉及做动作的视频，就不用出现了」。
    """

    ACTION = "action"   # 动作 / 体式 / 教学动作
    SPEECH = "speech"   # 口播 / 访谈 / 剧情对话


class ReplacementLevel(StrEnum):
    """替换级别（框架 §0.2）。由参考物料的种类推导，不作为独立输入。"""

    L0 = "L0"  # 全原创
    L1 = "L1"  # 只换台词
    L2 = "L2"  # 换人 + 换场景
    L3 = "L3"  # 换场景 + 动作复刻


class RefType(StrEnum):
    """参考物料类型（框架 §2.1）。"""

    MULTI_VIEW = "multi_view"       # 人物多视图 → 锁五官/脸型/发型/体态/服装
    SCENE = "scene"                 # 场景图 → 锁环境与光线
    BLOCKING = "blocking"           # 人物位置图 / 站位图 → 锁画面位置
    POSE_SHEET = "pose_sheet"       # 体式图 / 动作拆解图 → 锁动作序列
    VIDEO = "video"                 # 参考视频 → 1:1 复刻动作轨迹 / 运镜
    AUDIO = "audio"                 # 音频 → 音色克隆 + 全片声音一致


class SeedanceRole(StrEnum):
    """参考物料在 Seedance 侧的归属字段。"""

    IMAGE = "reference_image"
    VIDEO = "reference_video"
    AUDIO = "reference_audio"


class PositionCategory(StrEnum):
    """四类位置（框架 §10.1）。"""

    LOCK = "LOCK"    # 锁死位：改了复刻就失败
    PARAM = "PARAM"  # 档位位：固定白名单，切换即改风格档
    FREE = "FREE"    # 创作位：用户自由改写，产品价值所在
    AUTO = "AUTO"    # 自动位：系统从视频分析得出，用户无需关心


class Color(StrEnum):
    """三种颜色（PRD §6.2.3）。用户不需要理解框架，只需要理解三种颜色。

    这是**给用户看的唯一分类**，由 PositionCategory 单向映射而来。
    """

    BLUE = "blue"    # 🔵 可替换素材位
    ORANGE = "orange"  # 🟠 可编辑文案位
    GRAY = "gray"    # ⚫ 锁定区，只读


class RiskLevel(StrEnum):
    """改动风险（框架 §10.2）。决定 UI 提示强度与 diff 严格度。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BeatRole(StrEnum):
    """节拍功能（框架 §4 C2）。LOCK —— 改了就不是这条爆款的复刻。"""

    HOOK = "hook"              # 钩子
    SETUP = "setup"            # 铺垫
    DEMO = "demo"              # 展示
    PROOF = "proof"            # 证明
    CTA = "cta"                # 行动召唤
    TRANSITION = "transition"  # 过渡


class LockStrength(StrEnum):
    """五官锁定强度（框架 §3 B2）。两档。"""

    STANDARD = "standard"  # 锁定人物五官脸型，全程同一人物，五官稳定不漂移
    EXTREME = "extreme"    # 面部特征100%相同…禁止美化，禁止改变年龄


class BlinkBeat(StrEnum):
    """眨眼节拍（框架 §3 B5 唯一参数化差异）。必写维度，覆盖 10/10。"""

    THREE_TO_FOUR = "3-4s"  # 口播 / 教学 / 自拍
    FIVE = "5s"             # 对话类
    THREE = "3s"            # 带练 / 体式类


class QualityPreset(StrEnum):
    """画质预设（框架 §3 B5 三套）。"""

    P1_PHONE = "P1"   # 手机原生
    P2_SELFIE = "P2"  # 手机前置自拍原生
    P3_ARRI = "P3"    # ARRI 电影质感


class SkinText(StrEnum):
    """皮肤质感句（框架 §3 B5 两版，跨文档几乎逐字复用）。"""

    FULL = "full"    # 完整版（教学/演播厅）
    BRIEF = "brief"  # 简版（对话类常用）


class VolumeMode(StrEnum):
    """音量/语气模式（框架 §3 B6）。悄悄话是特殊模式，需前后双向约束。"""

    NORMAL = "normal"
    WHISPER = "whisper"


class SplitStrategy(StrEnum):
    """拆条策略（框架 §5）。因 Seedance 单次生成上限 4~15 秒。"""

    S1_ONE_TAKE = "S1"          # 一镜到底，时间戳分段
    S2_PER_SHOT = "S2"          # 分镜逐条
    S3_FIRST_LAST_FRAME = "S3"  # 首尾帧衔接
    S4_SAME_ANGLE = "S4"        # 同机位分段


class DeliveryMode(StrEnum):
    """交付形态（PRD §6.4.5 两阶段）。

    决定了**参数声明是否写进正文** —— 同一个提示词，两种模式正文不同。
    """

    PHASE1_MANUAL = "phase1_manual"  # 测试期：参数声明保留在提示词开头
    PHASE2_API = "phase2_api"        # 落地期：参数声明移除，走 API 参数


class OutputForm(StrEnum):
    """三形态输出（PRD F4.5）。内容必须一致，只是承载格式不同。"""

    MARKDOWN = "markdown"
    PLAIN = "plain"
    JSON = "json"


class Grade(StrEnum):
    """约束表述有效性分级（PRD §6.3.4 / 框架 §11 G14）。

    A→B→C→D 优先，E→F 必须阻断。
    """

    A_QUANTIFIED = "A"   # 量化：带数字，最高
    B_GEOMETRIC = "B"    # 几何：具体部位/结构，高
    C_TARGET = "C"       # 目标状态：给正向目标，高
    D_ENUMERATED = "D"   # 逐条禁止：枚举，中
    E_GENERIC = "E"      # 泛化：无具体对象，低 —— 样本明确禁止写
    F_WEAK = "F"         # 弱化：愿望式，无效 —— 样本中不出现


CONSTRAINT_GRADE_RANK: Mapping[Grade, int] = {
    Grade.A_QUANTIFIED: 4,
    Grade.B_GEOMETRIC: 3,
    Grade.C_TARGET: 2,
    Grade.D_ENUMERATED: 1,
    Grade.E_GENERIC: 0,
    Grade.F_WEAK: -1,
}

WEAK_GRADES = frozenset({Grade.E_GENERIC, Grade.F_WEAK})


@dataclass(frozen=True, slots=True)
class Span:
    """毫秒级时间区间。证据链的最小单位（PRD F1.8 每个位置带证据）。"""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    def to_dict(self) -> dict[str, int]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Span":
        return cls(int(raw["start_ms"]), int(raw["end_ms"]))

    def format_timecode(self) -> str:
        """00:03.8 形式，用于三列对照表。"""
        total = self.start_ms / 1000.0
        minutes = int(total // 60)
        seconds = total - minutes * 60
        return f"{minutes:02d}:{seconds:04.1f}"


class ConfidenceBasis(StrEnum):
    """置信度基础分来源（MULTIMODAL §4.2 权威表）。"""

    FFMPEG = "ffmpeg"                  # 元数据 / ffprobe / 音频信号分析
    CV = "cv"                          # CV 计算（光流、检测框、占比）
    ASR_CLEAR = "asr_clear"            # ASR（清晰语音）
    ASR_NOISY = "asr_noisy"            # ASR（嘈杂/多人重叠）
    VLM_CROSSED = "vlm_crossed"        # 多模态模型 + CV 一致
    VLM_ALONE = "vlm_alone"            # 多模态模型单独判断
    VLM_CONFLICT = "vlm_conflict"      # 多模态模型 + CV 冲突
    INFERRED = "inferred"              # 结构化推断（由以上推导，再 ×0.85）


CONFIDENCE_BASE: Mapping[ConfidenceBasis, float] = {
    ConfidenceBasis.FFMPEG: 1.00,
    ConfidenceBasis.CV: 0.95,
    ConfidenceBasis.ASR_CLEAR: 0.90,
    ConfidenceBasis.ASR_NOISY: 0.65,
    ConfidenceBasis.VLM_CROSSED: 0.90,
    ConfidenceBasis.VLM_ALONE: 0.70,
    ConfidenceBasis.VLM_CONFLICT: 0.60,
    ConfidenceBasis.INFERRED: 0.85,  # 实际算法：由来源基础分再乘 0.85
}

STRUCTURED_INFERENCE_FACTOR = 0.85
"""结构化推断的额外折扣（MULTIMODAL §4.2 末行）。"""

CONSISTENCY_COEFFICIENT: Mapping[str, float] = {
    "agree": 1.00,     # 两个独立来源一致
    "single": 0.85,    # 只有一个来源
    "conflict": 0.70,  # 两个来源冲突
}

EVIDENCE_COEFFICIENT: Mapping[str, float] = {
    "timecode_and_frame": 1.00,  # 有明确时间码 + 画面区域证据
    "timecode_only": 0.90,       # 只有时间码
    "none": 0.70,                # 无证据（纯推断）
}

# 置信度分级处置（MULTIMODAL §4.3 —— 注意与 PRD §6.1.4 的分档口径不同，
# 这里以更细的 MULTIMODAL 表为准，PRD 的 0.70/0.50 是两个分界点）
CONFIDENCE_CERTAIN = 0.85   # ≥0.85 确定：直接采用，不打扰用户
CONFIDENCE_ACCEPT = 0.70    # 0.70~0.85 高：直接采用
CONFIDENCE_REVIEW = 0.50    # 0.50~0.70 中：标黄提示确认；<0.50 低：不填


def compute_confidence(
    basis: ConfidenceBasis,
    *,
    consistency: str = "single",
    evidence: str = "timecode_only",
    is_structured_inference: bool = False,
) -> float:
    """置信度 = 基础分（来源类型）× 一致性系数 × 证据完整度。

    ``is_structured_inference`` 为真时再乘 0.85（由其它维度推导出来的值）。
    """
    base = CONFIDENCE_BASE[basis]
    value = (
        base
        * CONSISTENCY_COEFFICIENT.get(consistency, 0.85)
        * EVIDENCE_COEFFICIENT.get(evidence, 0.90)
    )
    if is_structured_inference:
        value *= STRUCTURED_INFERENCE_FACTOR
    return round(min(1.0, max(0.0, value)), 4)


@dataclass(frozen=True, slots=True)
class Provenance:
    """反推溯源：这个值从哪来、有多可信、证据在片子的哪一段。"""

    confidence: float
    basis: ConfidenceBasis
    span: Span | None = None
    note: str = ""
    evidence: str = ""

    @property
    def disposition(self) -> str:
        """分级处置（MULTIMODAL §4.3）：certain / accept / review / blank。"""
        if self.confidence >= CONFIDENCE_CERTAIN:
            return "certain"
        if self.confidence >= CONFIDENCE_ACCEPT:
            return "accept"
        if self.confidence >= CONFIDENCE_REVIEW:
            return "review"
        return "blank"

    @property
    def must_confirm(self) -> bool:
        return self.confidence < CONFIDENCE_ACCEPT

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "basis": self.basis.value,
            "disposition": self.disposition,
            "span": self.span.to_dict() if self.span else None,
            "note": self.note,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Provenance":
        span = raw.get("span")
        return cls(
            confidence=float(raw["confidence"]),
            basis=ConfidenceBasis(raw["basis"]),
            span=Span.from_dict(span) if span else None,
            note=str(raw.get("note", "")),
            evidence=str(raw.get("evidence", "")),
        )


UNKNOWN = "unknown"
"""低置信度留空时的占位值。**不使用空字符串**，以便区分"没反推出来"与"本来就是空"。"""
