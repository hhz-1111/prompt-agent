"""MiMo 的三个 Provider：视觉理解（L1）、语义归纳（L2）、语音识别（L0）。

分工原则（PRD §6.1.3）：**能算的绝不让模型猜**。
所以这里只做"真需要理解"的部分；时长、切分、站位这类确定性强判断
走 ffmpeg 与 CV 层，这里不掺和。

每条 prompt 都限定**闭集词表**（景别、机位、运镜），
否则模型会自由发挥出 ``缓慢升降镜头`` 这种库里没有的档位，
下游的位置标注与词库校验就全废了。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from engine.core.contracts import (
    SemanticSummary,
    ShotBoundary,
    ShotUnderstanding,
    TranscriptSegment,
    VideoInfo,
)
from engine.core.types import (
    BlinkBeat,
    Genre,
    QualityPreset,
    ShotMode,
    compute_confidence,
    ConfidenceBasis,
)

from ..config import Settings
from .ffmpeg_media import MediaError
from .mimo import BASE_SYSTEM, MimoClient, MimoError

FRAMING_CHOICES = ["全景", "中全景", "中景", "中近景", "近景", "脸部特写", "unknown"]
CAMERA_CHOICES = ["平视", "正面", "过肩", "俯拍", "仰拍", "固定", "推近", "跟拍", "横摇", "手持", "unknown"]

VISUAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["shots"],
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "shot_index", "framing", "camera", "location",
                    "description", "people_count", "people_labels",
                    "blocking", "expression", "emotion",
                ],
                "properties": {
                    "shot_index": {"type": "integer"},
                    "framing": {"type": "string", "enum": FRAMING_CHOICES},
                    "camera": {"type": "string", "enum": CAMERA_CHOICES},
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                    "people_count": {"type": "integer"},
                    "people_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "blocking": {"type": "string"},
                    "expression": {"type": "string"},
                    "emotion": {"type": "string"},
                },
            },
        }
    },
}

SEMANTIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "logline", "scene_type", "genre", "shot_mode",
        "hook_index", "quality_preset", "blink_beat",
    ],
    "properties": {
        "logline": {"type": "string"},
        "scene_type": {"type": "string"},
        "genre": {
            "type": "string",
            "enum": ["dialogue", "teaching", "studio", "street_interview", "replacement"],
        },
        "shot_mode": {
            "type": "string",
            "enum": ["one_take", "multi_shot", "same_angle_segments"],
        },
        "hook_index": {"type": "integer"},
        "quality_preset": {"type": "string", "enum": ["P1", "P2", "P3"]},
        "blink_beat": {"type": "string", "enum": ["3-4s", "5s", "3s"]},
    },
}

VISUAL_SYSTEM = (
    BASE_SYSTEM
    + "\n你是短视频分镜分析师。你会收到同一条片子的多个镜头关键帧，"
    "按给定顺序逐个镜头描述：景别、机位运镜、场景地点、人物动作走位、表情、情绪。"
    "\n纪律：只描述你**确实看到**的内容。看不清就填 unknown，不要猜。"
    "禁止添加画面里没有的人物性别、年龄、职业、道具。"
    "\n景别与机位必须从给定闭集里选，不要自创说法。"
    "\npeople_labels 只填能区分的称呼（如「左侧女子」「右侧男子」），"
    "**不要猜姓名、年龄、职业**；看不清就留空数组。"
)

SEMANTIC_SYSTEM = (
    BASE_SYSTEM
    + "\n你是短视频结构分析师。基于逐镜头描述与台词，归纳这条片子的整体结构。"
    "\n纪律："
    "\n- logline 是**一句话**剧情主线，不超过 40 字，只写发生了什么；"
    "\n- hook_index 是最抓人的那个钩子镜头号（从 1 开始）；"
    "\n- shot_mode 依据镜头是否真正切换来判定；"
    "\n- 不要复述镜头细节，只做归纳。"
)


class MimoVisualDescriber:
    """L1：多模态视觉理解（逐镜头抽帧）。"""

    def __init__(self, settings: Settings) -> None:
        self.client = MimoClient(settings, model_id=settings.model_vision)

    def describe_shots(
        self,
        frames: Sequence[str],
        shots: Sequence[ShotBoundary],
        context: str = "",
    ) -> list[ShotUnderstanding]:
        if not frames:
            return []
        if len(frames) != len(shots):
            # 不静默对齐：帧数与镜头数不符说明抽帧阶段出了问题
            raise MimoError(
                f"帧数({len(frames)})与镜头数({len(shots)})不一致，无法逐镜头标注"
            )

        labels = "、".join(f"第{i}张=镜头{s.index}" for i, s in enumerate(shots, 1))
        user = (
            f"共 {len(shots)} 个镜头，图片顺序即镜头顺序：{labels}。\n"
            f"请逐个镜头输出结构化描述。\n{context}"
        )
        raw = self.client.complete_json(
            system=VISUAL_SYSTEM, user=user, schema=VISUAL_SCHEMA, images=frames
        )
        by_index = {
            int(item.get("shot_index", 0)): item
            for item in raw.get("shots", [])
            if isinstance(item, Mapping)
        }

        base = compute_confidence(
            ConfidenceBasis.VLM_ALONE, consistency="single", evidence="timecode_and_frame"
        )
        out: list[ShotUnderstanding] = []
        for shot in shots:
            item = by_index.get(shot.index, {})
            labels = item.get("people_labels")
            out.append(ShotUnderstanding(
                shot_index=shot.index,
                description=str(item.get("description") or ""),
                framing=_clean(item.get("framing"), FRAMING_CHOICES),
                camera=_clean(item.get("camera"), CAMERA_CHOICES),
                location=str(item.get("location") or ""),
                expression=str(item.get("expression") or ""),
                emotion=str(item.get("emotion") or ""),
                blocking=str(item.get("blocking") or ""),
                people=tuple(
                    str(name).strip() for name in labels
                    if isinstance(labels, list) and str(name).strip()
                ),
                confidence=base,
                raw=dict(item),
            ))
        return out


class MimoSemanticInducer:
    """L2：语义归纳。"""

    def __init__(self, settings: Settings) -> None:
        self.client = MimoClient(settings, model_id=settings.model_text)

    def induce(
        self,
        understandings: Sequence[ShotUnderstanding],
        transcript: Sequence[TranscriptSegment] = (),
        video: VideoInfo | None = None,
    ) -> SemanticSummary:
        if not understandings:
            raise MimoError("没有镜头理解结果，无法做语义归纳")

        lines = [
            f"镜头{u.shot_index}：景别={u.framing or '未知'}，机位={u.camera or '未知'}，"
            f"场景={u.location or '未知'}，动作={u.blocking or '无'}，"
            f"表情={u.expression or '无'}，情绪={u.emotion or '无'}"
            for u in understandings
        ]
        if transcript:
            lines.append("台词：")
            lines.extend(f"  [{t.span.format_timecode()}] {t.speaker}：{t.text}" for t in transcript)
        if video:
            lines.append(f"片长 {video.duration_sec:.1f} 秒，画面比例 {video.aspect_ratio}，"
                         f"共 {len(understandings)} 个镜头。")
        lines.append(
            "注意：单次生成上限 15 秒，"
            f"当前 {len(understandings)} 个镜头，请据此判断 shot_mode。"
        )

        raw = self.client.complete_json(
            system=SEMANTIC_SYSTEM, user="\n".join(lines), schema=SEMANTIC_SCHEMA
        )
        base = compute_confidence(
            ConfidenceBasis.INFERRED,
            consistency="single",
            evidence="timecode_only",
        )
        return SemanticSummary(
            logline=str(raw.get("logline") or ""),
            scene_type=str(raw.get("scene_type") or ""),
            genre=_enum(raw.get("genre"), Genre, Genre.DIALOGUE),
            shot_mode=_enum(raw.get("shot_mode"), ShotMode, ShotMode.ONE_TAKE),
            hook_index=int(raw.get("hook_index") or 0),
            quality_preset=_enum(raw.get("quality_preset"), QualityPreset, QualityPreset.P1_PHONE),
            blink_beat=_enum(raw.get("blink_beat"), BlinkBeat, BlinkBeat.FIVE),
            confidence=base,
        )


class MimoTranscriber:
    """L0：语音识别。"""

    def __init__(self, settings: Settings) -> None:
        self.client = MimoClient(settings, model_id=settings.model_asr)

    def transcribe(
        self,
        audio_path: str,
        model: str | None = None,
    ) -> list[TranscriptSegment]:
        return self.client.transcribe(audio_path, model=model)


# ----------------------------------------------------------------------
def _clean(value: Any, choices: Sequence[str]) -> str:
    """收敛到闭集词表。不在表里就返回 unknown —— 不猜、不自创。"""
    text = str(value or "").strip()
    return text if text in choices else "unknown"


def _enum(value: Any, enum_type: Any, default: Any) -> Any:
    try:
        return enum_type(value)
    except (ValueError, TypeError):
        return default
