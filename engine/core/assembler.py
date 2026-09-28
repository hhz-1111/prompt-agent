"""组装与交付（框架 §9、PRD §6.4）。

职责边界很清楚：

    组装器只做「把 IR 变成提示词文本」这一件事。
    它不判断能不能交付 —— 那是闸门的事。

三件容易被做错、这里刻意做对的事：

1. **压缩不是可选项。** 500 字上限 vs 正文常超 800 字是 PRD 明列的风险，
   所以 ``compress`` 不只给建议，而是**真的执行** PRD §9.2 的丢弃优先级，
   并把每一步记为 ``CompressionEntry`` —— 红线 4 禁止静默裁剪。
2. **永不可裁的内容有硬保护。** 站位、音色、台词原文、D0、D7
   在压缩阶段被显式跳过，任何改写都不会碰到它们。
3. **每段带 sources。** ``PromptSection.sources`` 记录该段由哪些位置决定，
   这是框架 §10.3「应用位置」的落地：改了某个槽位，
   系统能准确回答"输出提示词的第几段会变"。
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import framework as fw
from . import phrases as ph
from .types import (
    UNKNOWN,
    BlinkBeat,
    DeliveryMode,
    Grade,
    OutputForm,
    QualityPreset,
    RefType,
    SeedanceRole,
    SkinText,
    SplitStrategy,
    VolumeMode,
)

PARAM_HEADER_PREFIX = "参数声明 · 仅手动复制模式携带，直调 API 时请移除"


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # 基本汉字
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
        or 0xF900 <= code <= 0xFAFF   # 兼容汉字
    )


# 参数声明（Phase 1 手动复制模式保留；Phase 2 移除）
PARAM_HEADER_TEMPLATE = "{ratio} 竖屏，{duration} 秒，{resolution}，生成同步音频。"

# 音色/口型固定句
LIPSYNC_SENTENCE = "所有对白保持自然停顿和情绪起伏，口型与台词同步。"
DIALOGUE_PREFIX = "口型要求：中文普通话，语速中等，吐字清晰。"

# 人物一致性的长尾展开段。
# 框架 §9.2 明确它可在超预算时丢弃，只保留
# 「{角色}形象参考[图N]」+「锁定人物五官脸型」两句 —— 这两句才是硬约束。
PERSON_LONGTAIL_RE = _re.compile(
    r"，用于[^。]{0,120}?(?:气质参考|视觉辨识度|自然真实质感)[^。]{0,40}?。"
)
PERSON_LONGTAIL_TEXT = "用于…的五官细节、面部轮廓、外貌、发型、体态和气质参考"

ROLE_OF_REF: Mapping[RefType, SeedanceRole] = {
    RefType.MULTI_VIEW: SeedanceRole.IMAGE,
    RefType.SCENE: SeedanceRole.IMAGE,
    RefType.BLOCKING: SeedanceRole.IMAGE,
    RefType.POSE_SHEET: SeedanceRole.IMAGE,
    RefType.AUDIO: SeedanceRole.AUDIO,
    RefType.VIDEO: SeedanceRole.VIDEO,
}

SEEDANCE_ROLE_LABEL: Mapping[SeedanceRole, str] = {
    SeedanceRole.IMAGE: "图",
    SeedanceRole.VIDEO: "视频",
    SeedanceRole.AUDIO: "音频",
}

# 各类素材的默认用途声明。G12 判定为"自动修复"，这里就是那个修复动作的内容。
DEFAULT_PURPOSE: Mapping[RefType, str] = {
    RefType.MULTI_VIEW: "人物形象参考{token}，用于五官细节、面部轮廓、外貌、发型、体态和气质参考",
    RefType.SCENE: "场景环境参考{token}，用于锁定环境与光线氛围",
    RefType.BLOCKING: "人物位置参考{token}，用于锁定人物在画面中的位置",
    RefType.POSE_SHEET: "动作序列参考{token}，用于锁定动作拆解顺序",
    RefType.VIDEO: "动作轨迹参考{token}，1:1复刻其动作与运镜",
    RefType.AUDIO: "音色参考{token}，百分百还原其声音音色",
}


# --------------------------------------------------------------------------
# 输出结构
# --------------------------------------------------------------------------
@dataclass(slots=True)
class PromptSection:
    """提示词的一个段落。

    ``lines`` 是**纯正文**，不含段落标题 —— 因此 ``【题目】`` 不可能被
    占位符残留检查（G1）误判。标题只在渲染时加上去。
    """

    title: str
    lines: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines)

    @property
    def text(self) -> str:
        return f"【{self.title}】\n{self.body}" if self.lines else f"【{self.title}】"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "lines": list(self.lines),
            "text": self.text,
            "sources": list(self.sources),
        }


@dataclass(frozen=True, slots=True)
class CompressionEntry:
    """一条压缩记录。**不允许静默裁剪**（红线 4）。"""

    section: str
    what: str
    reason: str
    saved_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "what": self.what,
            "reason": self.reason,
            "saved_chars": self.saved_chars,
        }


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """一条拆条（一次生成请求）。"""

    index: int
    strategy: SplitStrategy
    beat_ids: tuple[int, ...]
    duration_sec: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "strategy": self.strategy.value,
            "beat_ids": list(self.beat_ids),
            "duration_sec": round(self.duration_sec, 2),
            "note": self.note,
        }


@dataclass(slots=True)
class PromptDocument:
    """交付物。三形态输出的共同来源，保证内容一致。"""

    sections: list[PromptSection] = field(default_factory=list)
    param_header: str = ""
    plan: list[SegmentPlan] = field(default_factory=list)
    compression: list[CompressionEntry] = field(default_factory=list)
    strategy: SplitStrategy = SplitStrategy.S1_ONE_TAKE
    warnings: list[str] = field(default_factory=list)
    budget: int = 500

    # ---- 文本视图 ----
    def full_text(self, *, include_header: bool = True) -> str:
        parts: list[str] = []
        if include_header and self.param_header:
            parts.append(self.param_header)
        parts.extend(section.text for section in self.sections)
        return "\n\n".join(parts)

    @property
    def body_text(self) -> str:
        """不含参数声明、也不含段落标题的正文。

        字数预算与占位符残留检查都读它 —— 标题不是提示词内容。
        """
        return "\n\n".join(section.body for section in self.sections)

    @property
    def char_count(self) -> int:
        """**中文**字数 —— 只数中文字符，不含空白、标点与英文。

        为什么这样算：Seedance 的 ≤500 字建议值针对的是中文表述，
        而画质预设（``auto white balance slightly cool…``）是约 200 字符的
        英文技术短语，从不计入这个预算。PRD §5 示例里的 "486/500"
        就是这个口径 —— 若把英文也算进去，任何合规提示词都会"超标"。
        """
        return sum(1 for ch in self.body_text if _is_cjk(ch))

    @property
    def total_char_count(self) -> int:
        """含英文与标点的总字符数（去掉空白）。用于体检与展示。"""
        return sum(1 for ch in self.body_text if not ch.isspace())

    @property
    def over_budget(self) -> int:
        """超出预算的中文字数；未超为 0。"""
        return max(0, self.char_count - self.budget)

    def section(self, title: str) -> PromptSection | None:
        for item in self.sections:
            if item.title == title:
                return item
        return None

    def text_lines(self) -> list[tuple[str, str]]:
        """``(段落标题, 正文行)`` 序列。短语体检需要段落语境。

        为什么要带段落：创作区（剧情/节拍/对白）是**用户的表达**，
        在里面出现"参考一下"是正常说话，不是把强约束降级。
        只有约束文本才该过 G14。
        """
        return [
            (section.title, line)
            for section in self.sections
            for line in section.lines
        ]

    def all_lines(self) -> list[str]:
        return [line for _, line in self.text_lines()]

    # ---- 三形态输出（PRD F4.5：内容必须一致）----
    def render(self, form: OutputForm) -> str:
        if form is OutputForm.PLAIN:
            return self.render_plain()
        if form is OutputForm.JSON:
            return self.render_json()
        return self.render_markdown()

    def render_markdown(self) -> str:
        parts: list[str] = []
        if self.param_header:
            parts.append(f"> {self.param_header}")
        parts.extend(section.text for section in self.sections)
        return "\n\n".join(parts) + "\n"

    def render_plain(self) -> str:
        """纯文本：零格式污染（PRD F4.7），可直接粘进中台。"""
        return self.full_text(include_header=True) + "\n"

    def render_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "param_header": self.param_header,
            "char_count": self.char_count,
            "total_char_count": self.total_char_count,
            "budget": self.budget,
            "over_budget": self.over_budget,
            "sections": [section.to_dict() for section in self.sections],
            "plan": [seg.to_dict() for seg in self.plan],
            "compression": [entry.to_dict() for entry in self.compression],
            "warnings": list(self.warnings),
            "text": self.full_text(),
        }


@dataclass(frozen=True, slots=True)
class SplitIssue:
    """拆条边界问题（PRD F4.2 的 6 种情况）。"""

    code: str
    severity: str  # block / warn
    title: str
    detail: str = ""
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "suggestion": self.suggestion,
        }


# --------------------------------------------------------------------------
# 素材句柄
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Handle:
    """``@人物A多视图`` → ``[图1]人物A多视图``"""

    name: str
    token: str
    role: SeedanceRole
    index: int

    def __str__(self) -> str:
        return self.token


class HandleResolver:
    """把占位符名解析成 Seedance 能识别的 ``[图N]`` / ``[视频N]`` / ``[音频N]``。

    **硬性验收：交付物中 ``【】`` 残留 = 0**（PRD 红线 6）。
    没绑定的素材会被登记为 "unbound"，由闸门 G1/G2 阻断，绝不静默通过。
    """

    def __init__(self, ir: Mapping[str, Any]) -> None:
        self._handles: dict[str, Handle] = {}
        self.unbound: list[str] = []
        self._defaults: dict[str, str] = {}
        counters: dict[SeedanceRole, int] = {
            SeedanceRole.IMAGE: 0,
            SeedanceRole.VIDEO: 0,
            SeedanceRole.AUDIO: 0,
        }
        for ref in ir.get("references", []):
            ref_type = _safe_ref_type(ref.get("type"))
            role = ROLE_OF_REF.get(ref_type, SeedanceRole.IMAGE)
            counters[role] += 1
            label = SEEDANCE_ROLE_LABEL[role]
            token = f"[{label}{counters[role]}]"
            name = _ref_display_name(ref)
            handle = Handle(name=name, token=token, role=role, index=counters[role])
            for key in self._alias_keys(ref, name):
                self._handles[key] = handle
                self._defaults[key] = DEFAULT_PURPOSE[ref_type].format(token=token)
            if not ref.get("asset_id"):
                self.unbound.append(name)

    @staticmethod
    def _alias_keys(ref: Mapping[str, Any], name: str) -> list[str]:
        keys = {name, str(ref.get("placeholder", "")).strip("【】")}
        keys.update(str(slot) for slot in ref.get("slots", []))
        for key in list(keys):
            if "-" in key:
                keys.add(key.split("-", 1)[1])
        return [k for k in keys if k]

    def resolve(self, name: str) -> Handle | None:
        if name in self._handles:
            return self._handles[name]
        stripped = name.strip().lstrip("@").strip("【】")
        return self._handles.get(stripped)

    def token_for(self, name: str, *, fallback: str = "") -> str:
        handle = self.resolve(name)
        if handle:
            return handle.token
        return fallback or f"@{name.lstrip('@').strip('【】')}"

    def default_purpose(self, name: str) -> str:
        """素材的默认用途声明 —— G12 自动修复用（PRD G12 判定为"自动修复"）。"""
        for key in (name, name.strip().lstrip("@").strip("【】")):
            if key in self._defaults:
                return self._defaults[key]
        return ""

    def table_lines(self, declared: set[str] | None = None) -> list[str]:
        """``【参考素材】`` 段的清单行。

        给 ``declared`` 时只列**已在正文里声明用途**的素材 —— 这是原子性要求：
        未声明的素材不能出现在清单里，否则交付物会出现"引用了但没说用途"的素材。
        """
        lines: list[str] = []
        for handle in self._unique():
            if declared is not None and handle.token not in declared:
                continue
            lines.append(f"{handle.token} {handle.name}")
        return lines

    @property
    def all_tokens(self) -> list[str]:
        """全部素材句柄，按图/视/音顺序。"""
        return [h.token for h in self._unique()]

    def declaration(self, ref_name: str, purpose: str) -> str:
        """在正文里声明素材角色：``人物形象参考[图1]``。"""
        return f"{purpose}{self.token_for(ref_name)}"

    def _unique(self) -> list[Handle]:
        seen: set[str] = set()
        out: list[Handle] = []
        for handle in self._handles.values():
            if handle.token in seen:
                continue
            seen.add(handle.token)
            out.append(handle)
        out.sort(key=lambda h: (list(SeedanceRole).index(h.role), h.index))
        return out

    @property
    def lost_placeholders(self) -> int:
        """未绑定素材数量 —— 即交付物里会残留的 ``【】`` 数量。"""
        return len(self.unbound)


def _ref_display_name(ref: Mapping[str, Any]) -> str:
    slots = [str(s) for s in ref.get("slots", [])]
    if slots:
        return slots[0]
    placeholder = str(ref.get("placeholder", "")).strip("【】")
    return placeholder or f"素材{ref.get('prep_step', '')}".strip()


def _safe_ref_type(raw: Any) -> RefType:
    try:
        return RefType(raw)
    except (ValueError, TypeError):
        return RefType.MULTI_VIEW


# --------------------------------------------------------------------------
# 文本基元
# --------------------------------------------------------------------------
def ensure_quoted(line: str) -> str:
    """台词必须双引号包裹（G5，方舟官方建议，优化音频生成）。"""
    text = line.strip()
    if not text:
        return text
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text
    if len(text) >= 2 and text[0] in "“「" and text[-1] in "”」":
        return f'"{text[1:-1]}"'
    return f'"{text}"'


def estimate_speech_seconds(text: str, chars_per_sec: float = 5.0) -> float:
    """语速估算（AGENT §5.4）。当前为估算值，非实测。"""
    count = sum(1 for ch in text if not ch.isspace())
    if count == 0:
        return 0.0
    return round(count / chars_per_sec, 2)


def effective_duration(ir: Mapping[str, Any]) -> float:
    """本次生成要覆盖的时长（秒）。

    **``meta.total_duration_sec`` 优先**，因为它既承载 ffprobe 测得的原片时长，
    也是用户可调的**目标时长**入口。

    为什么不能只加节拍时长：G8 会建议"把目标时长拉到 20 秒"来容纳密集节拍，
    但如果时长是从节拍算出来的，用户就没有调整入口 —— 那句建议等于空话。
    取 max(节拍总和, 目标时长) 让"延长目标"立刻生效，同时不会把时长算得比内容还短。
    """
    beats = ir.get("creative", {}).get("C2_beats", [])
    beat_total = sum(float(b.get("duration_sec") or 0) for b in beats)
    target = float(ir.get("meta", {}).get("total_duration_sec") or 0)
    return round(max(beat_total, target), 2)


# --------------------------------------------------------------------------
# 关键一致性渲染
# --------------------------------------------------------------------------
def _render_scene(ir: Mapping[str, Any], handles: HandleResolver) -> list[str]:
    scene = ir.get("locked", {}).get("B1_scene", {})
    anchor = str(scene.get("anchor") or "").strip()
    if not anchor:
        return []
    return [_replace_placeholders(anchor, handles)]


def _render_person(
    ir: Mapping[str, Any],
    handles: HandleResolver,
    library: ph.PhraseLibrary,
) -> list[str]:
    lines: list[str] = []
    for person in ir.get("locked", {}).get("B2_person", []):
        name = str(person.get("name") or "").strip()
        ref = str(person.get("multi_view_ref") or "").strip()
        if not name and not ref:
            continue
        token = handles.token_for(ref or name, fallback=f"[{name}多视图]")
        parts = [
            f"{name}形象参考{token}，用于{name}的五官细节、面部轮廓、外貌、"
            f"发型、体态和气质参考，呈现视觉辨识度，保持自然真实质感。"
        ]
        level_text = _person_level_text(library, str(person.get("lock_strength") or "standard"))
        if level_text:
            parts.append(level_text + "。")
        wardrobe = str(person.get("wardrobe") or "").strip()
        if wardrobe:
            text = _replace_placeholders(wardrobe, handles)
            parts.append(text if text.endswith("。") else text + "。")
        skin_text = _skin_text(library, str(person.get("skin") or "brief"))
        if skin_text:
            parts.append(skin_text)
        blink_text = _blink_text(library, str(person.get("blink") or ""))
        if blink_text:
            parts.append(blink_text)
        lines.append("".join(parts))
    return lines


def _person_level_text(library: ph.PhraseLibrary, strength: str) -> str:
    """T-PERSON 的两档五官锁定句（standard / extreme），逐字来自规范文档。"""
    template = library.raw.get("templates", {}).get("T-PERSON", {})
    return str(template.get("levels", {}).get(strength, ""))


def _skin_text(library: ph.PhraseLibrary, key: str) -> str:
    template = library.templates.get("T-SKIN")
    if not template or not template.variants:
        return ""
    return template.variants.get(key, template.variants.get("brief", ""))


def _blink_text(library: ph.PhraseLibrary, key: str) -> str:
    template = library.templates.get("T-BLINK")
    if not template or not template.variants:
        return ""
    return template.variants.get(key, "")


def _render_blocking(ir: Mapping[str, Any]) -> list[str]:
    """站位渲染。**永不可裁**（框架 §9.2）。"""
    blocking = ir.get("locked", {}).get("B3_blocking", {})
    lines: list[str] = []
    positions = blocking.get("positions") or []
    if positions:
        joined = "、".join(str(p) for p in positions if str(p).strip())
        if joined:
            lines.append(f"{joined}，位置全程禁止变动。")
    for key in ("occlusion", "eye_line", "off_screen"):
        text = str(blocking.get(key) or "").strip()
        if text:
            lines.append(text if text.endswith("。") else text + "。")
    return lines


def _render_camera(ir: Mapping[str, Any]) -> list[str]:
    camera = ir.get("locked", {}).get("B4_camera", {})
    lines: list[str] = []
    rig = str(camera.get("rig") or "").strip()
    if rig and rig != UNKNOWN:
        lines.append(f"机位：{rig}。")
    return lines


def _render_quality(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
    *,
    include_preset_text: bool = True,
) -> list[str]:
    quality = ir.get("locked", {}).get("B5_quality", {})
    lines: list[str] = []
    grain = str(quality.get("grain") or "").strip()
    if grain:
        lines.append(grain)
    if include_preset_text:
        preset = str(quality.get("preset") or "")
        template = library.templates.get("T-QUALITY")
        variant = template.variants.get(preset, "") if template and template.variants else ""
        if variant:
            lines.append(f"画质：{variant}。")
    return lines


def _render_audio(ir: Mapping[str, Any], handles: HandleResolver) -> list[str]:
    """音色渲染。**永不可裁**（框架 §9.2）。"""
    lines: list[str] = []
    for track in ir.get("locked", {}).get("B6_audio", []):
        speaker = str(track.get("speaker") or "").strip()
        ref = str(track.get("audio_ref") or "").strip()
        if not speaker and not ref:
            continue
        token = handles.token_for(ref or speaker, fallback=f"[{speaker}音色]")
        line = f"{speaker}声音音色参考{token}，百分百还原{token}的声音音色，全片保持声音一致性。"
        if str(track.get("volume_mode") or "") == VolumeMode.WHISPER.value:
            line = "（极低压声）" + line + "气口自然顺滑，全程压低音量的悄悄话，不额外添加台词。"
        lines.append(line)
    return lines


# --------------------------------------------------------------------------
# 创作区渲染
# --------------------------------------------------------------------------
def _render_beats(ir: Mapping[str, Any], *, max_body_chars: int = 0) -> list[str]:
    """按 shot_mode 决定写法：一镜到底用时间戳分段，多镜头用「镜头N：」。

    ``max_body_chars`` > 0 时截断单条节拍的动作描述 —— 模型对场景的散文式
    描写常常上百字，而节拍需要的是**动作**。截断点选在句子边界，不切半句。
    """
    shot_mode = str(ir.get("meta", {}).get("shot_mode") or "one_take")
    lines: list[str] = []
    cursor = 0.0
    for beat in ir.get("creative", {}).get("C2_beats", []):
        shot_id = beat.get("shot_id") or 0
        duration = float(beat.get("duration_sec") or 0)
        framing = str(beat.get("framing") or "").strip()
        camera = str(beat.get("camera") or "").strip()
        location = str(beat.get("location") or "").strip()
        blocking = str(beat.get("blocking") or "").strip()
        expression = str(beat.get("expression") or "").strip()
        emotion = str(beat.get("emotion") or "").strip()

        if max_body_chars:
            blocking = _truncate_at_sentence(blocking, max_body_chars)

        head = "".join(p for p in (framing, camera) if p and p != UNKNOWN)
        if location:
            head = f"{head}，{location}" if head else location
        if shot_mode == "one_take":
            start, end = cursor, cursor + duration
            cursor = end
            prefix = f"{_fmt_sec(start)}-{_fmt_sec(end)}秒画面："
        else:
            prefix = f"镜头{shot_id}："

        # 技术头 / 动作 / 表情 / 情绪，逐段成句，避免标点拼接出错。
        # UNKNOWN 不写进正文：把"没反推出来"写成 unknown 是往提示词里灌噪声。
        segments = [
            _text_or_empty(head), _text_or_empty(blocking),
            _text_or_empty(expression),
        ]
        if _text_or_empty(emotion):
            segments.append(f"语气{_text_or_empty(emotion)}")
        body = "，".join(s.rstrip("。，") for s in segments if s)
        if not body:
            # 这个镜头什么都没反推出来：跳过，不产出 "镜头N：unknown。" 这种垃圾行
            continue
        lines.append(f"{prefix}{body}。")
    return lines


def _truncate_at_sentence(text: str, limit: int) -> str:
    """在句子边界截断，绝不留半句话。"""
    if len(text) <= limit:
        return text
    head = text[:limit]
    for mark in "。；！？，、":
        index = head.rfind(mark)
        if index > limit * 0.5:
            return head[: index + 1].rstrip("，、")
    return head


def _text_or_empty(value: str) -> str:
    return "" if value in UNKNOWN or value is None else str(value).strip()


def _fmt_sec(value: float) -> str:
    return f"{value:g}"


def _render_dialogue(ir: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for beat in ir.get("creative", {}).get("C2_beats", []):
        dialogue = beat.get("dialogue") or {}
        line = str(dialogue.get("line") or "").strip()
        if not line:
            continue
        shot_id = beat.get("shot_id") or 0
        speaker = str(dialogue.get("speaker") or "").strip()
        tone = str(dialogue.get("tone") or "").strip()
        head = f"镜头{shot_id}"
        if speaker:
            head += f"（{speaker}"
            head += f"，语气{tone}）" if tone else "）"
        lines.append(f"{head}：{ensure_quoted(line)}")
    if not lines:
        # 没有台词就不要输出口型段落 —— 空段落会被误读成"要求保持口型同步"
        return []
    return [DIALOGUE_PREFIX] + lines + [LIPSYNC_SENTENCE]


def _render_forbidden(
    ir: Mapping[str, Any],
    *,
    drop_groups: frozenset[str] = frozenset(),
) -> list[str]:
    return [
        str(item.get("text") or "").strip()
        for item in ir.get("forbidden", [])
        if str(item.get("text") or "").strip()
        and str(item.get("group") or "") not in drop_groups
    ]


def _replace_placeholders(text: str, handles: HandleResolver) -> str:
    """解析期把 ``【xxx】`` 换成 ``[图N]``（框架 §2.1 两步转换的第二步）。"""
    def swap(match: "_re.Match[str]") -> str:
        return handles.token_for(match.group(1), fallback=match.group(0))

    return _re.sub(r"【([^】]{1,24})】", swap, text)


_DECLARED_TOKEN_RE = _re.compile(r"\[(?:图|视频|音频)\d+\]")


def declared_tokens(text: str) -> set[str]:
    """从正文里提取"已被声明用途"的素材句柄。

    清单行（``[图1] 王姐多视图``）只是命名，不算声明；
    真正声明用途的是正文里的 ``王姐形象参考[图1]``。
    """
    return set(_DECLARED_TOKEN_RE.findall(text))


# --------------------------------------------------------------------------
# 拆条规划
# --------------------------------------------------------------------------
def plan_split(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
) -> tuple[list[SegmentPlan], list[SplitIssue]]:
    """拆条策略推荐 + 边界校验（PRD F4.1 / F4.2）。

    这里只**推荐**与**报问题**，不擅自改用户的镜头结构。
    """
    meta = ir.get("meta", {})
    shot_mode = str(meta.get("shot_mode") or "one_take")
    requested = str(ir.get("delivery", {}).get("strategy") or "")
    strategy = _resolve_strategy(requested, shot_mode)

    model = str(meta.get("target_model") or "seedance-2.0")
    capacity = library.capacity_for(model)
    clip_min, clip_max = capacity["clip_duration_sec"]

    beats = list(ir.get("creative", {}).get("C2_beats", []))
    durations = [float(b.get("duration_sec") or 0) for b in beats]
    total = sum(durations)
    issues: list[SplitIssue] = []

    for index, beat in enumerate(beats, start=1):
        duration = float(beat.get("duration_sec") or 0)
        if 0 < duration < clip_min:
            issues.append(SplitIssue(
                code="SPLIT-TOO-SHORT",
                severity="warn",
                title=f"镜头{beat.get('shot_id', index)} 仅 {duration:g} 秒，低于单次生成下限 {clip_min} 秒",
                detail="Seedance 单次生成时长为 4~15 秒，过短的动作段建议与相邻节拍合并。",
                suggestion="与相邻节拍合并，或补足动作细节至 4 秒以上。",
            ))

    if strategy is SplitStrategy.S1_ONE_TAKE and total > clip_max:
        issues.append(SplitIssue(
            code="SPLIT-S1-OVERFLOW",
            severity="block",
            title=f"一镜到底共 {total:g} 秒，超过单次生成上限 {clip_max} 秒",
            detail="S1 要求一次生成，超限必然丢细节 —— 这是「抽卡」的头号原因。",
            suggestion=f"改走 S4 同机位分段，按时间推进切成 {_ceil_div(total, clip_max)} 条，"
                       f"B3 站位与 B4 机位逐字复用。",
        ))
    elif total > clip_max:
        issues.append(SplitIssue(
            code="SPLIT-NEEDS-SPLIT",
            severity="warn",
            title=f"全片 {total:g} 秒，需拆成 {_ceil_div(total, clip_max)} 条",
            detail=f"{strategy.value} 策略下每条不超过 {clip_max} 秒。",
            suggestion="锁定位与禁止项在拆条时逐字复用，避免跨条人物/站位漂移。",
        ))

    # 密度：节拍数 ÷ 时长 ≤ 0.27（G8 的算法，这里提前算出来供 UI 预警）
    if total > 0 and beats:
        density = len(beats) / total
        if density > DENSITY_LIMIT:
            issues.append(SplitIssue(
                code="SPLIT-DENSITY",
                severity="block",
                title=f"内容密度超标：{len(beats)} 节拍 ÷ {total:g} 秒 = {density:.2f} > {DENSITY_LIMIT}",
                detail="官方边界：15 秒对应 2~4 个清晰节拍。塞太多节拍模型必然忽略细节。",
                suggestion="合并节拍或拆条。",
            ))

    return _build_segments(beats, strategy, clip_max), issues


# 密度上限（G8）：官方边界 —— 15 秒对应 2~4 个清晰节拍
DENSITY_LIMIT = 0.27


def _resolve_strategy(requested: str, shot_mode: str) -> SplitStrategy:
    if requested:
        try:
            return SplitStrategy(requested)
        except ValueError:
            pass
    if shot_mode == "multi_shot":
        return SplitStrategy.S2_PER_SHOT
    if shot_mode == "same_angle_segments":
        return SplitStrategy.S4_SAME_ANGLE
    return SplitStrategy.S1_ONE_TAKE


def _build_segments(
    beats: Sequence[Mapping[str, Any]],
    strategy: SplitStrategy,
    clip_max: float,
) -> list[SegmentPlan]:
    if not beats:
        return []
    if strategy is SplitStrategy.S2_PER_SHOT:
        return [
            SegmentPlan(
                index=i,
                strategy=strategy,
                beat_ids=(int(b.get("shot_id") or i),),
                duration_sec=float(b.get("duration_sec") or 0),
                note="每条共享同一套锁定项与参考图",
            )
            for i, b in enumerate(beats, start=1)
        ]

    # S1 / S4：按时间累积切段，不切断节拍
    segments: list[SegmentPlan] = []
    current: list[Mapping[str, Any]] = []
    current_duration = 0.0
    for beat in beats:
        duration = float(beat.get("duration_sec") or 0)
        if current and current_duration + duration > clip_max:
            segments.append(SegmentPlan(
                index=len(segments) + 1,
                strategy=strategy,
                beat_ids=tuple(int(b.get("shot_id") or 0) for b in current),
                duration_sec=current_duration,
                note="锁定位逐字复用",
            ))
            current, current_duration = [], 0.0
        current.append(beat)
        current_duration += duration
    if current:
        segments.append(SegmentPlan(
            index=len(segments) + 1,
            strategy=strategy,
            beat_ids=tuple(int(b.get("shot_id") or 0) for b in current),
            duration_sec=current_duration,
            note="锁定位逐字复用",
        ))
    return segments


def _ceil_div(value: float, divisor: float) -> int:
    if divisor <= 0:
        return 1
    whole = int(value // divisor)
    return whole + (1 if value % divisor else 0)


# --------------------------------------------------------------------------
# 压缩（框架 §9.2）
# --------------------------------------------------------------------------
# 丢弃顺序即优先级：从先丢到后丢。
# **永不可裁**（框架 §9.2 明文）：B3 站位/视线/遮挡、B6 音色参考、
# C3 台词原文、D0 通用底线、D7 对白锁定。它们不在下表里，也没有任何代码路径能删它们。
COMPRESSION_STEPS: tuple[str, ...] = (
    "quality_preset_text",   # ① B5 次要画质描述：保留预设名，丢弃展开句
    "person_longtail",       # ② B2 人物长尾：保留「参考[图N]锁定五官脸型」
    "forbidden_low_freq",    # ③ D 区低频条目（D5/D6）
    "beat_body_truncate",    # ④ C2 节拍里的散文式场景描写
)


@dataclass(slots=True)
class _CompressionPlan:
    """一次压缩的完整决策。组装器据此渲染，而不是渲染完再改文本 —— 后者容易漏改。"""

    quality_preset_text: bool = True
    person_longtail: bool = True
    forbidden_low_freq: bool = True
    beat_body_limit: int = 0
    entries: list[CompressionEntry] = field(default_factory=list)


def compress(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
    *,
    budget: int | None = None,
) -> _CompressionPlan:
    """按 PRD §9.2 的优先级**真正执行**压缩，直到落入预算。

    与"只给建议"的区别：这里返回的是**渲染计划**，
    组装器直接按计划渲染，所以交付物确实变短了。

    压缩不会触碰永不可裁的内容 —— 站位、音色、台词、D0、D7
    都不在可压缩集合里。裁不掉的部分会如实留在 ``entries`` 里报告。
    """
    model = str(ir.get("meta", {}).get("target_model") or "seedance-2.0")
    limit = budget if budget is not None else int(library.capacity_for(model)["prompt_char_budget"])

    plan = _CompressionPlan()
    # 先渲染"不压缩"版本，量出真实字数再决定压到哪一步
    baseline = _render_full(ir, library, plan).char_count
    if baseline <= limit:
        return plan

    for step in COMPRESSION_STEPS:
        running = _render_full(ir, library, plan).char_count
        if running <= limit:
            break
        before = running
        entry = _apply_step(step, ir, library, plan)
        if entry is None:
            continue
        after = _render_full(ir, library, plan).char_count
        # 记录真实节省量，而不是估算 —— 估算会让用户无法判断压得够不够
        plan.entries.append(replace_saved(entry, before - after))

    return plan


def replace_saved(entry: CompressionEntry, saved: int) -> CompressionEntry:
    return CompressionEntry(
        section=entry.section,
        what=entry.what,
        reason=entry.reason,
        saved_chars=max(0, saved),
    )


def _apply_step(
    step: str,
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
    plan: _CompressionPlan,
) -> CompressionEntry | None:
    if step == "quality_preset_text" and plan.quality_preset_text:
        preset = str(ir.get("locked", {}).get("B5_quality", {}).get("preset") or "")
        template = library.templates.get("T-QUALITY")
        variant = template.variants.get(preset, "") if template and template.variants else ""
        if not variant:
            return None
        plan.quality_preset_text = False
        return CompressionEntry(
            section="关键一致性",
            what=f"画质预设 {preset} 的展开描述",
            reason="保留预设档位名，丢弃展开句（框架 §9.2 优先级①）",
            saved_chars=0,
        )

    if step == "person_longtail" and plan.person_longtail:
        if not ir.get("locked", {}).get("B2_person"):
            return None
        plan.person_longtail = False
        return CompressionEntry(
            section="关键一致性",
            what=PERSON_LONGTAIL_TEXT,
            reason="保留「参考[图N]锁定五官脸型」，丢弃长尾（框架 §9.2 优先级②）",
            saved_chars=0,
        )

    if step == "forbidden_low_freq" and plan.forbidden_low_freq:
        groups = {
            str(item.get("group") or "")
            for item in ir.get("forbidden", [])
            if str(item.get("group") or "") in _DROPPABLE_FORBIDDEN_GROUPS
        }
        if not groups:
            return None
        plan.forbidden_low_freq = False
        return CompressionEntry(
            section="禁止项",
            what=f"{'、'.join(sorted(groups))} 低频禁止条目",
            reason="低频条目可在超预算时丢弃（框架 §9.2 优先级③）；D0/D7 永不可裁",
            saved_chars=0,
        )

    if step == "beat_body_truncate" and plan.beat_body_limit == 0:
        if not ir.get("creative", {}).get("C2_beats"):
            return None
        plan.beat_body_limit = 60
        return CompressionEntry(
            section="镜头节拍",
            what="节拍里散文式的场景描写（截断至 60 字，句子边界处断开）",
            reason="节拍需要的是动作，不是环境散文；截断不影响表演约束",
            saved_chars=0,
        )

    return None


_DROPPABLE_FORBIDDEN_GROUPS: frozenset[str] = frozenset({"D5", "D6"})
"""可在超预算时丢弃的禁止项组。

**D0 与 D7 永不在此列** —— 框架 §9.2 明确它们是永不可裁项。
"""


def _render_full(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
    plan: _CompressionPlan,
) -> PromptDocument:
    """按给定压缩计划渲染一次，用于量预算。"""
    return _build_document(ir, library, plan, with_param_header=False)


# --------------------------------------------------------------------------
# 主组装
# --------------------------------------------------------------------------
def assemble(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary | None = None,
    *,
    form: OutputForm = OutputForm.PLAIN,
    with_param_header: bool | None = None,
    budget: int | None = None,
    compress_to_budget: bool = True,
) -> PromptDocument:
    """把 IR 组装成一份可直接投喂的提示词。

    ``compress_to_budget`` 默认开启：PRD 明列"正文常超 800 字"是红线风险，
    所以组装阶段就按优先级压缩，而不是把问题丢给用户。
    压完仍超预算的话，闸门 G7 会如实报出还差多少。
    """
    lib = library or ph.load_library()
    model = str(ir.get("meta", {}).get("target_model") or "seedance-2.0")
    limit = budget if budget is not None else int(lib.capacity_for(model)["prompt_char_budget"])

    plan = compress(ir, lib, budget=limit) if compress_to_budget else _CompressionPlan(
        quality_preset_text=True, person_longtail=True,
        forbidden_low_freq=True, beat_body_limit=0,
    )

    include_header = with_param_header
    if include_header is None:
        mode = str(ir.get("delivery", {}).get("mode") or DeliveryMode.PHASE1_MANUAL.value)
        include_header = (
            mode == DeliveryMode.PHASE1_MANUAL.value
            and bool(ir.get("delivery", {}).get("include_param_header", True))
        )

    doc = _build_document(ir, lib, plan, with_param_header=include_header)
    doc.budget = limit
    doc.compression = plan.entries
    return doc


def _build_document(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
    plan: _CompressionPlan,
    *,
    with_param_header: bool,
) -> PromptDocument:
    handles = HandleResolver(ir)
    doc = PromptDocument()
    over: list[str] = []

    # ① 参数声明（★ 按交付模式决定是否写）
    if with_param_header:
        meta = ir.get("meta", {})
        doc.param_header = PARAM_HEADER_TEMPLATE.format(
            ratio=str(meta.get("aspect_ratio") or "9:16"),
            duration=_fmt_sec(effective_duration(ir)),
            resolution="720p",
        )

    # ③~⑨ 关键一致性
    locked_lines: list[str] = []
    locked_lines.extend(_render_scene(ir, handles))
    locked_lines.extend(_render_person(ir, handles, library))
    locked_lines.extend(_render_blocking(ir))
    locked_lines.extend(_render_camera(ir))
    locked_lines.extend(_render_quality(ir, library,
                                        include_preset_text=plan.quality_preset_text))
    locked_lines.extend(_render_audio(ir, handles))
    if not plan.person_longtail:
        locked_lines = [PERSON_LONGTAIL_RE.sub("。", line) for line in locked_lines]
    if not plan.quality_preset_text:
        preset = str(ir.get("locked", {}).get("B5_quality", {}).get("preset") or "")
        if preset:
            locked_lines.append(f"画质档位：{preset}。")
    if locked_lines:
        doc.sections.append(PromptSection(
            title=fw.SEC_LOCKED, lines=locked_lines, sources=["locked"],
        ))

    # ② 参考素材清单 —— 只列在正文里真正声明了用途的素材
    declared = declared_tokens("\n".join(locked_lines))
    ref_lines = handles.table_lines(declared)
    if ref_lines:
        doc.sections.insert(0, PromptSection(
            title=fw.SEC_REFS, lines=ref_lines, sources=["references"],
        ))
    missing_roles = [t for t in handles.all_tokens if t not in declared]

    # 剧情
    logline = str(ir.get("creative", {}).get("C1_logline") or "").strip()
    if logline:
        doc.sections.append(PromptSection(
            title=fw.SEC_LOGLINE, lines=[logline], sources=["creative.C1_logline"],
        ))

    # 镜头节拍
    beats = _render_beats(ir, max_body_chars=plan.beat_body_limit)
    if beats:
        doc.sections.append(PromptSection(
            title=fw.SEC_BEATS, lines=beats, sources=["creative.C2_beats"],
        ))

    # 对白与口型
    dialogue = _render_dialogue(ir)
    if dialogue:
        doc.sections.append(PromptSection(
            title=fw.SEC_DIALOGUE, lines=dialogue,
            sources=["creative.C2_beats[].dialogue"],
        ))

    # ⑩ 禁止项
    forbidden = _render_forbidden(
        ir,
        drop_groups=frozenset() if plan.forbidden_low_freq else _DROPPABLE_FORBIDDEN_GROUPS,
    )
    if forbidden:
        doc.sections.append(PromptSection(
            title=fw.SEC_FORBIDDEN, lines=forbidden, sources=["forbidden"],
        ))

    doc.strategy, split_issues = _split_summary(ir, library)
    doc.plan = _plans_of(ir, library)
    doc.warnings = [issue.title for issue in split_issues]
    if missing_roles:
        doc.warnings.append(
            f"{len(missing_roles)} 个参考素材未在正文声明用途：{'、'.join(missing_roles)}"
        )
    if handles.lost_placeholders:
        doc.warnings.append(f"{handles.lost_placeholders} 个素材位未绑定，交付物将残留占位符")
    if over:
        doc.warnings.extend(over)
    return doc


def _split_summary(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
) -> tuple[SplitStrategy, list[SplitIssue]]:
    plans, issues = plan_split(ir, library)
    strategy = plans[0].strategy if plans else _resolve_strategy(
        str(ir.get("delivery", {}).get("strategy") or ""),
        str(ir.get("meta", {}).get("shot_mode") or "one_take"),
    )
    return strategy, issues


def _plans_of(ir: Mapping[str, Any], library: ph.PhraseLibrary) -> list[SegmentPlan]:
    plans, _ = plan_split(ir, library)
    return plans