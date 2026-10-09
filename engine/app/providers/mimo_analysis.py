"""MiMo 的三个 Provider：视觉理解（L1）、语义归纳（L2）、语音识别（L0）。

分工原则（PRD §6.1.3）：**能算的绝不让模型猜**。
所以这里只做"真需要理解"的部分；时长、切分、站位这类确定性强判断
走 ffmpeg 与 CV 层，这里不掺和。

每条 prompt 都限定**闭集词表**（景别、机位、运镜），
否则模型会自由发挥出 ``缓慢升降镜头`` 这种库里没有的档位，
下游的位置标注与词库校验就全废了。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from engine.core.contracts import (
    CritiqueEdit,
    CritiqueOutcome,
    DialogueTurn,
    Extension,
    ProbeFinding,
    ProbeRequest,
    Question,
    RevisionOutcome,
    SemanticSummary,
    ShotBoundary,
    ShotDialogueContext,
    ShotUnderstanding,
    TranscriptSegment,
    VideoInfo,
)
from engine.core.review import allowed_path_reference
from engine.core.types import (
    BlinkBeat,
    ContentKind,
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
CAMERA_ANGLE_CHOICES = ["平视", "俯视", "仰视", "正面", "过肩", "侧拍", "偷拍", "自拍", "unknown"]
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
                    "shot_index", "framing", "camera_angle", "camera", "location",
                    "description", "people_count", "people_labels",
                    "blocking", "expression", "emotion", "subject", "group",
                ],
                "properties": {
                    "shot_index": {"type": "integer"},
                    "framing": {"type": "string", "enum": FRAMING_CHOICES},
                    "camera_angle": {"type": "string", "enum": CAMERA_ANGLE_CHOICES},
                    "camera": {"type": "string", "enum": CAMERA_CHOICES},
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                    "people_count": {"type": "integer"},
                    "people_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "subject": {"type": "string"},
                    "group": {"type": "string"},
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
        "hook_index", "quality_preset", "blink_beat", "content_kind",
        "people_structure",
    ],
    "properties": {
        "logline": {"type": "string"},
        "scene_type": {"type": "string"},
        "content_kind": {"type": "string", "enum": ["action", "speech"]},
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
        # **人物结构由模型回答，不由我在代码里猜**（用户 2026-10-08：
        # 「我做的不是 agent 吗？画面理解呢？」——三次"一堆人物参考"都是
        # 我在代码里用关键词/硬规则猜结构猜错的）。
        "people_structure": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "lead", "second", "distinct_others", "crowd_label",
                "crowd_look", "crowd_behavior",
            ],
            "properties": {
                "lead": {"type": "string"},
                "second": {"type": "string"},
                "distinct_others": {"type": "array", "items": {"type": "string"}},
                "crowd_label": {"type": "string"},
                "crowd_look": {"type": "string"},
                "crowd_behavior": {
                    "type": "string", "enum": ["follow", "watch", "own_action", "none"],
                },
            },
        },
    },
}

VISUAL_SYSTEM = (
    BASE_SYSTEM
    + "\n你是短视频分镜分析师。你会收到同一条片子的多个镜头关键帧，"
    "按给定顺序逐个镜头描述：景别、机位运镜、场景地点、人物动作走位、表情、情绪。"
    "\n纪律：只描述你**确实看到**的内容。看不清就填 unknown，不要猜。"
    "禁止添加画面里没有的人物性别、年龄、职业、道具。"
    "\n景别与机位必须从给定闭集里选，不要自创说法。"
    "\npeople_labels 是**跨镜头认人**用的，必须用**不随镜头变化**的特征来描述："
    "服装颜色、发型、配饰（如「紫色上衣女子」「红衣男子」「扎马尾女子」）。"
    "\n**不要用「左侧/右侧/中间/前景」这类位置词当称呼** —— "
    "同一个人换机位就会被叫成另一个名字，下游会把 2 个人当成 4 个。"
    "如果一个人身上实在看不出任何稳定特征，宁可在 people_labels 里留空。"
    "\n**不要猜姓名、年龄、职业**；看不清就留空数组。"
    "\n**subject / group 是多人与单人视频都要填的两个字段**："
    "\n- `subject`：这个镜头的**主角**——画面最中间、最显眼、正在做示范或正在说话的那位。"
    "用与 people_labels 一样的稳定特征词（如「白色上衣女子」）。"
    "**这个人必须同时出现在 people_labels 里**。"
    "如果画面里没有明确主角（空镜、纯群像），subject 留空字符串。"
    "\n- `group`：**除主角以外的那些人，作为一个整体**描述 —— 年龄段 + 穿着概况，"
    "一句话（如「40~50 岁女性，穿各色运动上衣与长裤」）。"
    "没有其他人就留空字符串。**不要在这里逐个列举个人特征** —— "
    "他们是群体，由生成模型自行生成，给群体的年龄与穿着就够了。"
)

DIALOGUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["turns"],
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["shot_index", "speaker", "text"],
                "properties": {
                    "shot_index": {"type": "integer"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "tone": {"type": "string"},
                },
            },
        }
    },
}

DIALOGUE_SYSTEM = (
    BASE_SYSTEM
    + "\n你是短视频台词归属分析师。你会收到：一段**没有说话人、没有时间戳**的转写文本，"
    "以及这条片子的逐镜头画面信息（每个镜头里有谁、什么景别、什么机位、多长）。"
    "\n你要做两件事，都必须做对："
    "\n1. **按说话人变化把转写切成一句一句**（一问一答就是两句，不要合并）；"
    "\n2. **判定每句是谁说的、属于哪个镜头**。"
    "\n判据（按可信度从高到低）："
    "\n- 画面里只有一个人时，那句就是这个人说的；"
    "\n- 有两个人时，用**画面里的人的特征**当说话人（如「紫色背心女子」），"
    "**不要写「人物1」「说话人A」这类代号**，也不要写「左侧的人」这类位置词；"
    "\n- 一问一答的对话：提问方与回答方通常是不同的人，"
    "按谁在画面里、以及话轮交替来分；"
    "\n- 按镜头顺序分配：前面的台词给前面的镜头，不要把所有台词塞进最后一个镜头。"
    "\n纪律："
    "\n- **只切分与归属，绝不改写、增删、润色任何一个字**；"
    "\n- speaker 只能用画面信息里出现过的特征词，拼不出来就留空字符串；"
    "\n- 不确定某句属于哪个镜头时，按它在文本里的先后顺序归到相应镜头。"
)
SEMANTIC_SYSTEM = (
    BASE_SYSTEM
    + "\n你是短视频结构分析师。基于逐镜头描述与台词，归纳这条片子的整体结构。"
    "\n纪律："
    "\n- logline 是**一句话**剧情主线，不超过 40 字，只写发生了什么；"
    "\n- hook_index 是最抓人的那个钩子镜头号（从 1 开始）；"
    "\n- shot_mode 依据镜头是否真正切换来判定；"
    "\n- scene_type 是**给用户看的中文场景名**：4~8 个汉字，写清「在哪、什么场景」"
    "（如「菜市场采访」「演播厅教学」「家中客厅口播」「户外健身带练」）。"
    "**不要用英文、下划线、代号，也不要写 single / unknown 这类空值**；"
    "实在判断不出来就留空字符串。"
    "\n- **content_kind 判的是「这条片子有没有需要复刻的身体动作」**，"
    "不是题材、也不是 genre："
    "\n    action = 有人在**做身体动作**，而且动作本身是观看价值的一部分"
    "（体式/瑜伽/拉伸/健身/舞蹈/动作示范教学 —— 老师带学员一起练属于这一档）；"
    "\n    speech = 以**说话**为主（口播/访谈/剧情对话/带货讲解）——"
    "站着说话时抬手、做手势、比划、举话筒都**不算** action。"
    "\n  判据看**全身与四肢**在不在动（弓步、抬腿、弯腰、侧伸展、下蹲、跳跃、转身…），"
    "而不是手部小动作。拿不准就填 speech。"
    "\n"
    "\n- **people_structure：这条片子里的人该怎么分工，由你判断**，"
    "不要只罗列镜头里出现过的每个人。规则："
    "\n    * `lead` = **全片主角**：贯穿全片、正在讲话/示范、画面最中间最显眼的那位"
    "（用逐镜头事实里的稳定特征词，如「紫色短袖上衣女子」）。"
    "\n    * `second` = **第二位需要锁定形象的人**（如双人对话里的另一位说话人）。"
    "**没有就留空** —— 不要为了凑数硬填。"
    "\n    * `distinct_others` = 除上面两位之外，**各自有独立身份、需要单独呈现**的人"
    "（剧情片里有台词或有戏份的配角）。**没有就留空数组**。"
    "\n    * `crowd_label` / `crowd_look` / `crowd_behavior` = **其余那些人作为群像**："
    "称呼（观众们/学员们/背景人群…）、年龄+穿着概况一句话、以及他们的行为"
    "（follow=跟着主角做同一套动作；watch=在旁边看；own_action=各自在做别的事，"
    "例如背景里有人在练瑜伽；没有群像就填 none 并留空两个文本）。"
    "\n    * **判据是画面事实**：一群人穿着各异、各自忙自己的、没有单独戏份 → 群像；"
    "有台词、有特写、被单独介绍的 → 独立人物。"
    "\n    * **宁少勿多**：拿不准某个人该不该单独锁定，就把他放进群像。"
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
            system=VISUAL_SYSTEM, user=user, schema=VISUAL_SCHEMA, images=frames,
            purpose="视觉理解",
        )
        by_index = {
            int(item.get("shot_index", 0)): item
            for item in (raw.get("shots") or [])
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
                camera_angle=_clean(item.get("camera_angle"), CAMERA_ANGLE_CHOICES),
                camera=_clean(item.get("camera"), CAMERA_CHOICES),
                location=str(item.get("location") or ""),
                expression=str(item.get("expression") or ""),
                emotion=str(item.get("emotion") or ""),
                blocking=str(item.get("blocking") or ""),
                # 注意：类型判断必须在**迭代之前**。
                # 原来写成 `for name in labels if isinstance(labels, list) ...`，
                # 推导式先迭代再判断，`people_labels: null` 时直接 TypeError ——
                # 留出集里一条新套路片子就是这么把整段视觉理解弄丢的。
                people=tuple(
                    str(name).strip()
                    for name in (labels if isinstance(labels, list) else [])
                    if str(name).strip()
                ),
                subject=str(item.get("subject") or "").strip(),
                group=str(item.get("group") or "").strip(),
                confidence=base,
                raw=dict(item),
            ))
        return out


PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "certain"],
    "properties": {
        "answer": {"type": "string"},
        "certain": {"type": "boolean"},
    },
}

PROBE_SYSTEM = (
    BASE_SYSTEM
    + "\n你是**取证员**。你会收到同一条片子某个时间点的几帧画面（可能是裁剪放大的局部），"
    "以及一个具体问题。"
    "\n只回答**画面里确实能看到的东西**：看得到什么就说什么，看不清就在 answer 里写明"
    "「看不清」并把 certain 置为 false。"
    "\n禁止推测：不要说画面上看不出的身份、动机、年龄、职业。"
    "\nanswer 用一句话，直接回答问题本身，不要复述问题、不要给建议。"
)

PROBE_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # 名称 → (x, y, 宽, 高) 相对比例。全画幅用 full。
    "left": (0.0, 0.0, 0.5, 1.0),
    "right": (0.5, 0.0, 0.5, 1.0),
    "center": (0.2, 0.0, 0.6, 1.0),
    "top": (0.0, 0.0, 1.0, 0.5),
    "bottom": (0.0, 0.5, 1.0, 0.5),
}
"""裁剪区域预设。小字/细节/手势放大才看得清，所以取证允许指名区域。"""


SHOT_FILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["location", "blocking", "expression", "emotion"],
    "properties": {
        "location": {"type": "string"},
        "blocking": {"type": "string"},
        "expression": {"type": "string"},
        "emotion": {"type": "string"},
    },
}

SHOT_FILL_SYSTEM = (
    BASE_SYSTEM
    + "\n你在给**单个镜头**补描述。你会收到这个镜头的几帧画面。"
    "\n只描述画面里**确实能看到**的：场景地点、人物在做什么、表情、情绪。"
    "\n看不清的字段**留空字符串**，不要写「未知」「看不清」这类字，也不要猜。"    "\n每项一句话、不要复述画面里没有的东西 —— 这是「补」，不是「编」。"
)


class MimoVisionProbe:
    """按需取证：抽指定时间点的帧（可裁剪放大）→ 让视觉模型回答一个具体问题。"""

    def __init__(self, settings: Settings, media: Any) -> None:
        self.settings = settings
        self.media = media
        self.client = MimoClient(settings, model_id=settings.model_vision)
        self.model_id = settings.model_vision
        # 取证单次调用也**封顶**（用户 2026-10-07）：原来走全局默认 180 秒 × 3 次，
        # 实测把自检的实际耗时顶到了 563.7 秒、冲破 480 秒预算。
        self.timeout_sec = float(getattr(settings, "probe_call_timeout_sec", 120.0) or 120.0)
        self.retries = int(getattr(settings, "probe_call_retries", 0) or 0)

    def look(self, video_path: str, request: ProbeRequest) -> ProbeFinding:
        start = max(0.0, request.time_start)
        end = max(start, request.time_end)
        # 区间内取 3 帧：动作类问题单帧往往看不出"在做什么"
        span = end - start
        stamps = (
            [start, start + span / 2, end] if span > 0.15 else [start]
        )
        out_dir = str(
            self.settings.workspace_dir / "probe" / f"{abs(hash((video_path, start, end))):x}"
        )
        frames = self.media.extract_frames(
            video_path, out_dir, timestamps_sec=stamps, width=768,
            crop=_crop_filter(request.region),
        )
        if not frames:
            return ProbeFinding(request=request, answer="", confidence=0.0)
        raw = self.client.complete_json(
            system=PROBE_SYSTEM,
            user=f"问题：{request.question}",
            schema=PROBE_SCHEMA,
            images=frames,
            timeout_sec=self.timeout_sec, retries=self.retries, purpose="取证",
        )
        answer = str(raw.get("answer") or "").strip()
        return ProbeFinding(
            request=request,
            answer=answer,
            frames=tuple(frames),
            confidence=compute_confidence(
                ConfidenceBasis.VLM_ALONE, consistency="single",
                evidence="timecode_and_frame",
            ) if raw.get("certain") else compute_confidence(
                ConfidenceBasis.INFERRED, consistency="single", evidence="timecode_only",
            ),
        )

    def describe_shot(
        self,
        video_path: str,
        shot: ShotBoundary,
        missing: Sequence[str],
    ) -> dict[str, str]:
        """重新描述**单个镜头**，只回填缺的字段（看不清就留空）。"""
        start = shot.start_ms / 1000
        end = shot.end_ms / 1000
        span = max(0.0, end - start)
        stamps = [start + span * f for f in (0.35, 0.65)] if span > 0.3 else [start]
        out_dir = str(
            self.settings.workspace_dir / "probe" / f"shot{shot.index}-{int(start * 1000)}"
        )
        frames = self.media.extract_frames(
            video_path, out_dir, timestamps_sec=stamps, width=768
        )
        if not frames:
            return {}
        raw = self.client.complete_json(
            system=SHOT_FILL_SYSTEM,
            user=f"这个镜头需要补的信息：{'、'.join(missing) or '场景与动作'}。",
            schema=SHOT_FILL_SCHEMA,
            images=frames,
            timeout_sec=self.timeout_sec, retries=self.retries, purpose="感知补看",
        )
        allowed = {"location", "blocking", "expression", "emotion"}
        return {
            key: str(raw.get(key) or "").strip()
            for key in allowed
            if key in missing and str(raw.get(key) or "").strip()
        }


def _crop_filter(region: str) -> str:
    """区域名 → ffmpeg ``crop`` 滤镜参数（拿不到就返回空串＝不裁）。"""
    box = PROBE_REGIONS.get(str(region or "").strip().lower())
    if not box:
        return ""
    x, y, width, height = box
    return f"crop=iw*{width}:ih*{height}:iw*{x}:ih*{y}"


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
            system=SEMANTIC_SYSTEM, user="\n".join(lines), schema=SEMANTIC_SCHEMA,
            purpose="结构归纳",
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
            # 拿不准就空着，交给 to_ir 的兜底判据（按画面里的动作词）——
            # 默认填 speech 会把动作片误判成口播，默认 action 会给口播片加空约束，
            # **空着**才能让兜底生效。
            content_kind=_enum(raw.get("content_kind"), ContentKind, ""),
            people_structure=_clean_structure(raw.get("people_structure")),
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


class MimoDialogueAttributor:
    """台词归属：整段转写 → 逐句台词 + 说话人 + 所属镜头。

    **为什么必须有这一环**（实测，不是设计洁癖）：

        中台封装的 ASR 是纯转写网关 —— 加提示词要时间戳返回 400
        （"ASR request must not include text parts"），
        加 ``response_format`` 返回 500，加 ``diarization`` 开关没有任何效果。
        拿回来的就是一段没有说话人、没有时间戳的文本。

        于是原来的实现只能把整段丢给"重叠最多的镜头"，
        结果是：**双人对话被压成一句话，全片只有最后一个镜头有台词**，
        说话人也因为"多角色时不推导"而留空。

    这一环用文本模型补上切分与归属，依据是**画面证据**
    （每个镜头里实际有谁、什么特征），而不是纯语感。
    """

    def __init__(self, settings: Settings) -> None:
        self.client = MimoClient(settings, model_id=settings.model_text)
        # 归属**不能**用取证那种短超时（用户 2026-10-07 的实测教训）：
        # flash 单次调用实测能跑到 236 秒，卡 120 秒会**直接把台词判没**。
        # 它是"内容"而不是"可选证据"，所以给到和自检同档的 240 秒；
        # 超时也不再丢内容 —— 管线会兜底成"整段挂在第一个镜头、说话人留空"。
        self.timeout_sec = float(
            getattr(settings, "review_call_timeout_sec", 240.0) or 240.0
        )
        self.retries = int(getattr(settings, "review_call_retries", 0) or 0)

    def attribute(
        self,
        transcript: str,
        shots: Sequence[ShotDialogueContext],
    ) -> list[DialogueTurn]:
        text = str(transcript or "").strip()
        if not text or not shots:
            return []

        lines = ["转写文本（无说话人、无时间戳）：", text, "", "逐镜头画面信息："]
        for shot in shots:
            who = "、".join(shot.people) if shot.people else "（画面里没有识别到人）"
            # 动作与表情是**"谁在说话"最直接的画面证据**：视觉理解经常直接写着
            # 「紫衣女子张口说话」。实测踩过：不给这两项时，4 句台词只判出第 1 句的
            # 说话人 —— 画面明明写了答案，归属环节却看不到。
            evidence = []
            if shot.action:
                evidence.append(f"动作 {shot.action}")
            if shot.expression:
                evidence.append(f"表情 {shot.expression}")
            tail = ("；" + "；".join(evidence)) if evidence else ""
            lines.append(
                f"  镜头{shot.shot_index}：时长 {shot.duration_sec:.1f} 秒，"
                f"景别 {shot.framing or '未知'}，机位 {shot.camera_angle or '未知'}，"
                f"场景 {shot.location or '未知'}，画面里的人物：{who}{tail}"
            )
        lines.append("")
        lines.append(
            "请把上面的转写切成逐句台词，并给出每句的 shot_index 与 speaker。"
            "speaker 用画面里人物的特征词（如「紫色背心女子」），不要用代号或位置词。"
            "\n**动作/表情里写着「谁张口说话」时，说话人就是那个人** —— 这是最硬的证据。"
            "实在判不出的，speaker 留空，不要猜。"
        )

        raw = self.client.complete_json(
            system=DIALOGUE_SYSTEM, user="\n".join(lines), schema=DIALOGUE_SCHEMA,
            timeout_sec=self.timeout_sec, retries=self.retries, purpose="台词归属",
        )
        base = compute_confidence(
            ConfidenceBasis.INFERRED, consistency="single", evidence="timecode_only"
        )
        valid_shots = {shot.shot_index for shot in shots}
        turns: list[DialogueTurn] = []
        for item in raw.get("turns") or []:
            if not isinstance(item, Mapping):
                continue
            sentence = str(item.get("text") or "").strip()
            if not sentence:
                continue
            try:
                shot_index = int(item.get("shot_index") or 0)
            except (TypeError, ValueError):
                shot_index = 0
            if shot_index not in valid_shots:
                # 模型给了不存在的镜头号：归到第一个镜头而不是丢掉这一句
                shot_index = shots[0].shot_index
            turns.append(DialogueTurn(
                shot_index=shot_index,
                speaker=str(item.get("speaker") or "").strip(),
                text=sentence,
                tone=str(item.get("tone") or "").strip(),
                confidence=base,
            ))
        return turns


# ----------------------------------------------------------------------
# 自检-重写回路：让写提示词的模型回头看自己写的东西
# ----------------------------------------------------------------------
CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["edits"],
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "action", "value", "reason"],
                "properties": {
                    "path": {"type": "string"},
                    "action": {"type": "string", "enum": ["set", "clear"]},
                    "value": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "probes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "time_start", "time_end", "region", "reason"],
                "properties": {
                    "question": {"type": "string"},
                    "time_start": {"type": "number"},
                    "time_end": {"type": "number"},
                    "region": {
                        "type": "string",
                        "enum": ["full", "left", "right", "center", "top", "bottom"],
                    },
                    "reason": {"type": "string"},
                },
            },
        },
        "extensions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "lines", "evidence", "unreadable"],
                "properties": {
                    "title": {"type": "string"},
                    # 大白话描述画面信息（有什么字、在哪儿、旁边有什么东西）
                    "lines": {"type": "array", "items": {"type": "string"}},
                    # 没看清的部分 + 为什么（如「分辨率不够、字太小」）；没有就留空
                    "unreadable": {"type": "string"},
                    # 依据哪一帧/哪次取证（只给我看，不进提示词）
                    "evidence": {"type": "string"},
                },
            },
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "text", "options", "reason"],
                "properties": {
                    "path": {"type": "string"},
                    "text": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}

CRITIQUE_SYSTEM = (
    BASE_SYSTEM
    + "\n你是提示词交付前的**自检员**。你会收到：① 一份已经写好的提示词交付物；"
    "② 这条片子的逐镜头画面事实（场景/人物/动作/表情，都是视觉模型实际看到的内容）。"
    "\n你的任务**不是重写**，而是找出下面这几类**语义问题**，然后只对出问题的字段提修改："
    "\n1. **人物结构**（用户 2026-10-07 反复强调、最容易出错的一类）："
    "需要「人物参考」的人**不该超过 2 个**；画面里成群的、各自忙自己的人"
    "（背景学员、台下观众）应当是**群像**（`locked.B2b_ensemble`：称呼 + 年龄 + 穿着），"
    "不该被当成一个个独立人物去锁形象；反过来，**有台词或有单独戏份的人不能漏**。"
    "发现这类问题就改 `locked.B2b_ensemble` 的字段（或如实写进问题里）。"
    "\n2. **没对象的约束**：一句约束没说清对谁/对哪个镜头（如「保持一致性」）。"
    "\n3. **与画面事实不符**：提示词里写的人数、位置、人物标号、场景与画面事实矛盾。"
    "\n4. **重复冗余**：同一信息在多个地方重复（如每个镜头都把同一个场景抄一遍）。"
    "\n5. **自相矛盾**：正文里两个声明互斥（如「固定 1 位人物出镜」却有「人物2」）。"
    "\n6. **未定义的引用**：用到了没定义过的人物标号。"
    "\n7. **空话**：写进去但没有任何约束力的句子。"
    "\n"
    "\n纪律（违反会被代码直接拒绝并记账）："
    "\n- **只能改下面列出的字段**，路径必须逐字照抄模板（下标写具体数字，如 creative.C2_beats[1].location）："
    f"\n{allowed_path_reference()}"
    "\n- **只改你自己写的那些描述性字段**。站位、景别、运镜、音色、台词原文、禁止项、"
    "角色名、素材绑定、遮挡/视线/画外**一个字都不许动** —— "
    "它们是量测结果或用户提供的原文，不是你能用判断推翻的东西。"
    "\n- 指出问题时**只提对应字段的修改**，不要把自己的分析写进任何字段。"
    "\n- **宁少勿多**：只改确实有问题的字段，没把握就不要提。改错一条比不改更糟。"
    "\n- **没把握就去查，不要猜**：如果某个判断需要看清某个时间点的画面"
    "（谁在说话、手势是什么、屏上写了什么字、人物关系如何），"
    "在 `probes` 里提出来 —— 你会拿到那个时间点的几帧（可指定裁剪区域放大），"
    "然后再决定改不改。**probes 与 edits 可以同时给，也可以在没把握时只给 probes。**"
    "\n- probes 每条要写：question（具体问什么，一句话）、time_start/time_end（秒）、"
    "region（full/left/right/center/top/bottom，看小字或细节时选局部放大）、reason。"
    "\n- **框架装不下的信息放进 `extensions`，不要硬塞进别的段落、也不要丢掉**："
    "例如屏上大字/字幕文案、分屏或画中画布局、变装或换背景的切换点、多人对话的轮次表 —— "
    "框架的 13 个段落一个都不覆盖这些。"
    "\n  写 extensions 的唯一标准是**让用户一眼看懂画面里有什么**（用户 2026-10-08 的原话："
    "「目前你给出的信息太复杂，看不懂，直接大白话描述出 AI 读取的画面信息」）："
    "\n   * `lines` 用**大白话直接说画面信息**：有什么字（原文照抄进「」里）、字在哪儿、"
    "什么颜色、旁边有什么东西、版面怎么排的。"
    "例如「背景屏上写着黄橙色大字『5天普拉提训练营』，右侧有两张人物对比照片」。"
    "\n   * **不要写我们自己的判断过程**（取证过、是否分屏、算不算字幕、属不属于场景元素…）——"
    "用户看不懂，对生成也没有用；那些写进 `evidence` 就行。"
    "\n   * **有没看清的就如实写进 `unreadable`**：是什么没看清 + **为什么**"
    "（如「屏幕右下角还有一行小字，画面分辨率不够、字太小，辨认不出内容」）。"
    "**不要装作看清了**，也不要因为看不清就整条不提。"
    "\n   * `title` 是段落名（**不得与既有段落重名**，4~10 个字，如「背景屏文字」）；"
    "`lines` 最多 3 行、每行一句话、80 字以内；`evidence` 写依据哪一帧/哪次取证看到的"
    "（**只给我看，不会进提示词**）。"
    "\n- **交付前必须专项确认一次这几件事**（它们是「框架外」信息的典型来源，"
    "而且你手上的证据里看不到）：① 画面上有没有文字（字幕/大字/贴纸/包装字样）；"
    "② 有没有分屏/画中画；③ 有没有换装或换背景的切换点。"
    "**不确定就用 probes 去看一眼**（例如 question=「画面上有没有文字？写的什么？」、"
    "region 用 top/center 放大），看到了就放进 extensions。"
    "\n- extensions **必须有画面证据**：没证据的段落会被代码直接拒绝。"
    "**不许在 extensions 里写台词**（台词只能来自用户的转写）。"
    "\n- **真的两难才在 `questions` 里问人**（最多 2 条）。问之前先想：这件事能不能**取证**看清？"
    "能取证就去查，别拿问题烦人。只有「两种都说得通、画面也定不了」时才问 —— 例如"
    "「这一镜是展示动作还是展示产品？」。每条写 path（**答案要写进哪个字段**，"
    "必须取自上面列出的可改路径）、text（一句话的问题）、"
    "options（2~4 个**能直接写进字段的值**，不要写「其他」「说不清」）、reason（为什么定不了）。"
    "\n- 有问题就查、查完再改；没问题就返回空的 edits 与 probes —— 这是正常的、也是我们希望的答案。"
)


class MimoPromptCritic:
    """自检回路：把交付物 + 画面证据交给文本模型，收回**字段级**修改。"""

    # 类级默认值：即使不走 ``__init__``（测试里常用 ``__new__`` 构造）也不会崩
    timeout_sec: float = 90.0
    retries: int = 0
    evidence_char_cap: int = 1200

    def __init__(self, settings: Settings) -> None:
        # **自检专用模型**（用户 2026-10-07 批准）：实测同一个自检任务
        # pro-ultraspeed 41.5 秒挑出 7 条，flash 60.9 秒挑出 1 条。
        model = str(getattr(settings, "model_review", "") or settings.model_text)
        self.client = MimoClient(settings, model_id=model)
        self.model_id = model
        # **单次调用的超时与重试单独收紧**（用户 2026-10-07 报"自检与重写卡很久"）：
        # 这是推理型长调用，默认的 180 秒 × 3 次尝试最坏能耗掉 9 分半，
        # 用户在界面上看到的就是"卡死"。超时算失败，由回路的时间预算兜底。
        self.timeout_sec = float(getattr(settings, "review_call_timeout_sec", 90.0) or 90.0)
        self.retries = int(getattr(settings, "review_call_retries", 0) or 0)
        self.evidence_char_cap = int(getattr(settings, "review_evidence_char_cap", 1200) or 0)

    def critique(
        self,
        *,
        prompt: str,
        evidence: Mapping[str, Any],
    ) -> CritiqueOutcome:
        """自检一次。**推理预算不够时自动退到最小证据重试**。

        实测踩过：完整证据（2511 字符）会让这个推理模型把 8192 个输出 token
        全花在推理上，``finish_reason=length``、正文返回空 —— 表面看像"模型没意见"，
        其实它根本没来得及说话。所以失败后必须**换小输入重试**，
        不能当成"没有问题"糊过去。
        """
        from engine.core.review import minimal_evidence, shrink_evidence

        # 裁到体积上限（防爆护栏）：长视频的逐镜头证据会膨胀到几千字
        evidence = shrink_evidence(evidence, int(getattr(self, "evidence_char_cap", 0) or 0))

        attempts = [dict(evidence)]
        reduced = minimal_evidence(evidence)
        if reduced != dict(evidence):
            attempts.append(reduced)

        last_error: Exception | None = None
        for payload in attempts:
            user = (
                "【交付物】\n" + prompt
                + "\n\n【逐镜头画面事实】\n"
                + json.dumps(payload, ensure_ascii=False)
            )
            try:
                raw = self.client.complete_json(
                    system=CRITIQUE_SYSTEM, user=user, schema=CRITIQUE_SCHEMA,
                    timeout_sec=self.timeout_sec, retries=self.retries, purpose="自检",
                )
            except MimoError as exc:
                last_error = exc
                continue
            return self._to_outcome(raw)
        if last_error is not None:
            raise last_error
        return CritiqueOutcome()

    @staticmethod
    def _to_outcome(raw: Mapping[str, Any]) -> CritiqueOutcome:
        """解析自检输出。

        **一律用 ``or []`` 兜底**：结构化输出里字段可能是 ``null``
        （实测踩过：某条片子上模型返回 ``{"shots": null}``，
        ``raw.get("shots", [])`` 拿到的是 ``None``，迭代直接 TypeError，
        整段视觉理解被降级丢掉 —— 一条新套路的片子才暴露出来）。
        """
        edits: list[CritiqueEdit] = []
        for item in raw.get("edits") or []:
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            action = str(item.get("action") or "set").strip()
            if action not in ("set", "clear"):
                action = "set"
            edits.append(CritiqueEdit(
                path=path,
                action=action,
                value=str(item.get("value") or "").strip(),
                reason=str(item.get("reason") or "").strip(),
            ))
        probes: list[ProbeRequest] = []
        for item in raw.get("probes") or []:
            if not isinstance(item, Mapping):
                continue
            request = ProbeRequest.from_dict(item)
            if request.question:
                probes.append(request)
        extensions: list[Extension] = []
        for item in raw.get("extensions") or []:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or "").strip()
            raw_lines = item.get("lines")
            lines = tuple(
                str(line).strip() for line in (raw_lines if isinstance(raw_lines, list) else [])
                if str(line).strip()
            )
            unreadable = str(item.get("unreadable") or "").strip()
            if title and (lines or unreadable):
                extensions.append(Extension(
                    title=title,
                    lines=lines,
                    evidence=str(item.get("evidence") or "").strip(),
                    unreadable=unreadable,
                ))
        questions: list[Question] = []
        for item in raw.get("questions") or []:
            if not isinstance(item, Mapping):
                continue
            question = Question.from_dict(item)
            if question.text and question.options:
                questions.append(question)
        return CritiqueOutcome.of(edits, probes, extensions, questions)


# ----------------------------------------------------------------------
# 用户意见改写（PRD F8.4「按建议重写」）
# ----------------------------------------------------------------------
REVISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ops", "note"],
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "path", "reason"],
                "properties": {
                    "op": {"type": "string", "enum": ["set", "clear", "add", "remove"]},
                    "path": {"type": "string"},
                    # 值可以是字符串（改一处文字），也可以是对象（新增一条人物/镜头），
                    # 所以这里**不限定类型** —— 具体形状由 core/revision 按契约校验。
                    "value": {},
                    "reason": {"type": "string"},
                },
            },
        },
        "extensions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "lines", "evidence", "unreadable"],
                "properties": {
                    "title": {"type": "string"},
                    # 大白话描述画面信息（有什么字、在哪儿、旁边有什么东西）
                    "lines": {"type": "array", "items": {"type": "string"}},
                    # 没看清的部分 + 为什么（如「分辨率不够、字太小」）；没有就留空
                    "unreadable": {"type": "string"},
                    # 依据哪一帧/哪次取证（只给我看，不进提示词）
                    "evidence": {"type": "string"},
                },
            },
        },
        "refusals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["instruction", "reason"],
                "properties": {
                    "instruction": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "note": {"type": "string"},
    },
}


class MimoPromptReviser:
    """用户意见改写：把「用户想要什么」翻成**字段级修改**的 Adapter。

    与 :class:`MimoPromptCritic` 的三点不同（都是用户 2026-10-08 的要求）：

    1. **可换端点**：``REVISE_BASE_URL`` / ``REVISE_API_KEY`` 填了就用它
       （DeepSeek 官方端点支持 ``max_completion_tokens``，但不支持
       ``json_schema`` → 走 ``json_object`` + schema 写进提示词）；
    2. **模型独立**：``MODEL_REVISE``（默认跟自检那个"极速"模型）；
    3. **是同步接口**：用户在对话框前等着，所以超时要明确、重试设 0，
       失败就把原因带回去，不让界面一直转圈。
    """

    # 类级默认值：测试里常用 ``__new__`` 构造，不能崩
    timeout_sec: float = 240.0
    retries: int = 0
    model_id: str = ""

    def __init__(self, settings: Settings) -> None:
        from .mimo import MimoClient

        base_url, api_key = settings.revise_endpoint
        model = settings.resolved_model_revise
        # **端点能力探测**：DeepSeek 官方对 json_schema 直接 400，
        # 用域名判断不够稳，所以按"端点不是 MiMo 就退回 json_object"处理，
        # 并把这条写在 capabilities 里（换端点后能一眼看出走的是哪条路）。
        native_schema = "xiaomimimo.com" in base_url
        self.client = MimoClient(
            settings, model_id=model, base_url=base_url, api_key=api_key,
            json_schema=native_schema,
        )
        self.model_id = model
        self.base_url = base_url
        self.timeout_sec = float(getattr(settings, "revise_call_timeout_sec", 240.0) or 240.0)
        self.retries = int(getattr(settings, "revise_call_retries", 0) or 0)

    def revise(
        self,
        *,
        instruction: str,
        ir: Mapping[str, Any],
        prompt: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> RevisionOutcome:
        """改写一次。失败就抛错（由编排层记成"这次没改成 + 原因"）。"""
        from engine.core.revision import parse_outcome, system_prompt, user_prompt

        user = user_prompt(instruction=instruction, ir=ir, prompt_text=prompt,
                           evidence=evidence)
        raw = self.client.complete_json(
            system=system_prompt(),
            user=user,
            schema=REVISION_SCHEMA,
            timeout_sec=self.timeout_sec,
            retries=self.retries,
            purpose="按意见改写",
        )
        return parse_outcome(raw)


# ----------------------------------------------------------------------
def _clean_structure(raw: Any) -> dict[str, Any]:
    """收敛模型给的"人物结构"。

    **这是模型对画面的理解，不是我在代码里猜的规则。** 只做三件事：
    去掉空值、把行为归到闭集、字段类型不对就整块丢掉（宁缺勿滥 ——
    空着会退回代码兜底，写错则会把人认错）。
    """
    if not isinstance(raw, Mapping):
        return {}
    behavior = str(raw.get("crowd_behavior") or "").strip()
    if behavior not in ("follow", "watch", "own_action", "none"):
        behavior = ""
    structure = {
        "lead": str(raw.get("lead") or "").strip(),
        "second": str(raw.get("second") or "").strip(),
        "distinct_others": [
            str(item).strip()
            for item in (raw.get("distinct_others") or [])
            if isinstance(item, str) and str(item).strip()
        ],
        "crowd_label": str(raw.get("crowd_label") or "").strip(),
        "crowd_look": str(raw.get("crowd_look") or "").strip(),
        "crowd_behavior": behavior,
    }
    if not any((structure["lead"], structure["second"], structure["crowd_label"])):
        return {}
    return structure


def _clean(value: Any, choices: Sequence[str]) -> str:
    """收敛到闭集词表。不在表里就返回 unknown —— 不猜、不自创。"""
    text = str(value or "").strip()
    return text if text in choices else "unknown"


def _enum(value: Any, enum_type: Any, default: Any) -> Any:
    try:
        return enum_type(value)
    except (ValueError, TypeError):
        return default
