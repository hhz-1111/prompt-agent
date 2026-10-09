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

    ``hints`` 是**给用户看的操作提示**（如"↳ 可替换：[图1] = 王姐多视图"），
    **绝不进 ``body``/``text``/``full_text``**：它们不是提示词内容，
    混进去会污染字数预算、闸门校验和导出到生成平台的正文。
    前端单独渲染它们（可以做得比正文更醒目）。
    """

    title: str
    lines: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

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
            "hints": list(self.hints),
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
    def content_char_count(self) -> int:
        """**提交口径**的字数：中文 + 拉丁字母 + 数字，不含空白与标点。

        为什么 G7 用这个而不是中文字数（实测踩过）：被判预算的正是提交给平台的
        那串文本。原来只数中文，于是 ASR 在静音上编出来的
        「0, 1, 2, …, 100.」一个中文都没有 —— 736 中文字 / 1232 总字符却判"合规"。
        现在数字与字母算内容，标点与空白不算（它们是排版，不是内容）。
        """
        return sum(1 for ch in self.body_text if ch.isalnum())

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
        # 编号**按句柄标签各自计**（不是按角色）：这样 [视频1] 与 [参考动作视频1]
        # 可以并存 —— 用户 2026-10-07 要的正是这种可自解释的编号。
        counters: dict[str, int] = {}
        for ref in ir.get("references", []):
            ref_type = _safe_ref_type(ref.get("type"))
            role = ROLE_OF_REF.get(ref_type, SeedanceRole.IMAGE)
            # 句柄标签：默认按角色（图/视频/音频）；素材带 ``token_label`` 就用它 ——
            # 用户自己上传的素材需要自解释的标签（``[参考动作视频1]``）。
            label = str(ref.get("token_label") or "").strip() or SEEDANCE_ROLE_LABEL[role]
            counters[label] = counters.get(label, 0) + 1
            index = counters[label]
            token = f"[{label}{index}]"
            name = _ref_display_name(ref)
            handle = Handle(name=name, token=token, role=role, index=index)
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


def _render_ensemble(ir: Mapping[str, Any]) -> list[str]:
    """群体句：主体之外的那些人（学员/观众）**一句话**说清。

    用户 2026-10-07：「其他学员直接由 Seedance 自己生成，只需要给她们定位成学员们，
    然后穿着、年龄定位好，动作跟着老师做就行了 —— **这样就不需要那么多的提示词**。」
    又说：「台下的**观众们**是 AI 生成的群像，那么多人都定义为观众们，
    只需要定义年龄和穿着即可。」

    **两种群体行为分开写**（写错就很荒唐）：

    - ``follow``（学员/队员）：跟着主体做同一套动作；
    - ``watch``（观众/听众）：在旁边安静观看，不跟着做动作、不抢镜。
    """
    ensemble = ir.get("locked", {}).get("B2b_ensemble")
    if not isinstance(ensemble, Mapping):
        return []
    label = str(ensemble.get("label") or "").strip()
    look = str(ensemble.get("look") or "").strip()
    follows = str(ensemble.get("follows") or "").strip()
    mode = str(ensemble.get("mode") or "").strip() or ("follow" if follows else "watch")
    if not label and not look:
        return []
    label = label or "其他人物"
    head = f"{label}〔{look}〕：" if look else f"{label}："
    # **措辞要避开 E 级（泛化）表述**：实测第一版写「保持一致」被 G14 判为
    # 泛化约束直接阻断（资产里 "保持一致" 是 E 级模式）。
    parts = ["由画面自然生成，五官不单独锁定"]
    if mode == "follow" and follows:
        parts.append(f"动作全程跟随{follows}做同一套动作，不自行增减动作")
    else:
        # 非"跟随"型群体**不替它们编行为**：观众是坐着看的，
        # 而后景学员是在做瑜伽的 —— 一刀切写"安静观看"会把后者写错。
        # 它们的动作在哪一拍里，由【镜头节拍】逐镜描述（用户 2026-10-07 指出）。
        parts.append("按画面里的位置与动作自然存在，不遮挡主体、不抢镜")
    return [head + "，".join(parts) + "。"]


def _render_person(
    ir: Mapping[str, Any],
    handles: HandleResolver,
    library: ph.PhraseLibrary,
) -> tuple[list[str], list[str]]:
    """逐人物写形象参考句。

    正文行格式：``人物1〔紫色连衣裙女子 · 画面中央〕：…形象参考[图1]，…``

    返回 ``(正文行, 提示行)``。提示行通道（``PromptSection.hints``）保留但
    **当前恒为空** —— 用户 2026-09-29 明确要求去掉「↳ 可替换：…」这类辅助标记。

    **眨眼节拍与皮肤质感不在这里** —— 它们是无条件固定项，
    由 ``_render_universal_person_phrases`` 统一渲染，
    否则会在每个人物后面各重复一遍（实测过：双人片子渲出两遍）。
    """
    lines: list[str] = []
    hints: list[str] = []
    persons = ir.get("locked", {}).get("B2_person", [])
    for index, person in enumerate(persons, start=1):
        name = str(person.get("name") or "").strip()
        display = str(person.get("display_name") or "").strip() or name
        position = str(person.get("position") or "").strip()
        ref = str(person.get("multi_view_ref") or "").strip()
        if not name and not ref:
            continue
        token = handles.token_for(ref or name, fallback=f"[{name}多视图]")
        head = f"人物{index}〔{display}"
        if position:
            head += f" · {position}"
        head += "〕："
        parts = [
            f"{head}{display}形象参考{token}，用于{display}的五官细节、面部轮廓、外貌、"
            f"发型、体态和气质参考，呈现视觉辨识度，保持自然真实质感。"
        ]
        level_text = _person_level_text(library, str(person.get("lock_strength") or "standard"))
        if level_text:
            parts.append(level_text + "。")
        wardrobe = str(person.get("wardrobe") or "").strip()
        if wardrobe:
            text = _replace_placeholders(wardrobe, handles)
            parts.append(text if text.endswith("。") else text + "。")
        lines.append("".join(parts))
        # 「↳ 可替换：[图N] = xxx」提示行**已按用户要求去掉**（2026-09-29）：
        # 用户不要这类辅助标记，交付物就是干净的提示词正文。
        # ``PromptSection.hints`` 通道保留，以后要加回只需在这里 append。
    return lines, hints


def _render_universal_person_phrases(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
) -> list[str]:
    """眨眼节拍 + 皮肤质感：**一律写进交付物**。

    这两句是「模型默认会做错」的地方（眨眼频率不自然、过度磨皮），
    所以它们是**无条件固定项**，不挂在人物条目下面。

    取值规则：
      眨眼 —— 有人物就取第一个人物的节拍（同一片子里所有人用同一节拍）；
              没有人物条目就按内容类型取默认档（框架 §3 B5 的唯一参数化差异）；
      皮肤 —— 短款 T-SKIN-2。长款留作可选项（长度预算吃紧时短款够用）。
    """
    blink_key = ""
    for person in ir.get("locked", {}).get("B2_person", []):
        candidate = str(person.get("blink") or "").strip()
        if candidate:
            blink_key = candidate
            break
    if not blink_key:
        from engine.core.types import BlinkBeat, Genre

        genre_text = str(ir.get("meta", {}).get("genre") or "")
        try:
            genre = Genre(genre_text)
        except ValueError:
            genre = Genre.DIALOGUE
        blink_key = {
            Genre.DIALOGUE: BlinkBeat.FIVE,
            Genre.TEACHING: BlinkBeat.THREE_TO_FOUR,
            Genre.STUDIO: BlinkBeat.THREE_TO_FOUR,
            Genre.STREET_INTERVIEW: BlinkBeat.FIVE,
            Genre.REPLACEMENT: BlinkBeat.THREE,
        }.get(genre, BlinkBeat.FIVE).value

    lines: list[str] = []
    skin = _skin_text(library, "brief")
    if skin:
        lines.append(skin)
    blink = _blink_text(library, blink_key)
    if blink:
        lines.append(blink)
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
    """站位渲染。**永不可裁**（框架 §9.2）。

    固定尾句「位置全程禁止变动」在这里**统一追加**，不依赖数据里带没带 ——
    它是框架的硬约束，不能因为反推器没写就消失。
    但如果数据里已经有了（比如双人场景由 ``_fill_blocking`` 直接写出），
    就不要再叠一遍：否则会渲出「……位置全程禁止变动，位置全程禁止变动。」，
    实测踩过这个叠字。
    """
    blocking = ir.get("locked", {}).get("B3_blocking", {})
    lines: list[str] = []
    positions = [str(p).strip() for p in (blocking.get("positions") or []) if str(p).strip()]
    if positions:
        joined = "、".join(p.rstrip("。，") for p in positions)
        if "位置全程禁止变动" not in joined:
            joined = f"{joined}，位置全程禁止变动"
        lines.append(joined + "。")
    for key in ("occlusion", "eye_line", "off_screen"):
        text = str(blocking.get(key) or "").strip()
        if text:
            lines.append(text if text.endswith("。") else text + "。")
    return lines


def _render_camera(ir: Mapping[str, Any]) -> list[str]:
    """机位整句**不再单独渲染** —— 用户 2026-09-29 要求把它放进【镜头节拍】逐镜头写。

    单独一句"机位：固定机位全程不变"在双人/多镜头片子里是**重复且更弱**的信息：
    真正有用的是每个镜头怎么拍（「中景，平视机位拍摄，固定机位拍摄」）。
    逐镜头机位已在 ``_render_beats`` 里写出。

    函数保留是因为 ``locked.B4_camera.rig`` 仍是契约字段、
    闸门 G13 的"机位"维度仍靠它判覆盖 —— 那是**数据层**的事，与渲染无关。
    """
    _ = ir
    return []


_BODY_MOVEMENT_WORDS = (
    # 全身/四肢在动 —— 这些词出现才说明"有动作要复刻"
    "伸展", "拉伸", "弯腰", "俯身", "屈膝", "下蹲", "深蹲", "弓步", "抬腿", "踢腿",
    "压腿", "劈叉", "后仰", "前倾", "跳跃", "起跳", "转身", "转身", "翻滚", "俯卧撑",
    "平板", "举腿", "卷腹", "开合跳", "体式", "瑜伽", "普拉提", "健身操", "舞蹈",
    "手臂上举", "双臂", "双腿", "全身", "肢体",
)
"""**需要复刻的身体动作**的词。

刻意不收「抬手/举手/手势/比划/持话筒」这类**说话时的小动作**：
实测一条舞台口播片的动作描述是「右手持话筒于胸前，左手抬起做展示手势」——
按小动作判会把口播片误判成动作片，正是用户反馈的那个错。
"""


_SCENE_ACTION_WORDS = (
    "瑜伽", "普拉提", "健身", "舞蹈", "体式", "带练", "训练", "太极", "拉伸", "形体",
)
"""**场景本身就是动作场景**的词（「瑜伽教室带练」）。

场景名比逐镜头的动作描述更稳：它是对全片的归纳。
"""

_SCENE_SPEECH_WORDS = ("口播", "访谈", "对话", "讲解", "聊聊", "带货")
"""场景名里出现这些 → **一票否决**：口播/访谈/对话类的场景不写动作段。

（否则「健身房口播」会被「健身」这两个字误判成动作片。）
"""


def looks_like_action(ir: Mapping[str, Any]) -> bool:
    """按画面事实兜底判断"有没有要复刻的身体动作"。

    只在 ``meta.content_kind`` 为空时用（模型没判出来）。

    用户 2026-10-07 第二次强调：「**不需要模仿动作和不需要做动作的视频
    都不需要**动作与物理」—— 所以这里取保守判据，两层：

    1. **场景类型**优先：「瑜伽教室带练」这种本身就是动作场景；
       但场景里有「口播/访谈/对话」就**一票否决**（否则「健身房口播」会被"健身"误判）；
    2. 否则看**逐镜头动作描述**：要有身体动作的镜头达到 **60%**。
       剧情片里有人弯腰捡东西、口播片里有人起身，都不算"要复刻的动作"。
    """
    scene = str(ir.get("locked", {}).get("B1_scene", {}).get("type") or "")
    if any(word in scene for word in _SCENE_SPEECH_WORDS):
        return False
    if any(word in scene for word in _SCENE_ACTION_WORDS):
        return True

    beats = list(ir.get("creative", {}).get("C2_beats", []))
    if not beats:
        return False
    hits = sum(
        1 for beat in beats
        if any(word in str(beat.get("blocking") or "") for word in _BODY_MOVEMENT_WORDS)
    )
    return hits >= 1 and hits * 5 >= len(beats) * 3


def needs_action_physics(ir: Mapping[str, Any]) -> bool:
    """这条片子要不要写【动作与物理】（用户 2026-10-07 的判据）。

    **看 ``meta.content_kind``，不看 genre**：实测 genre 判不准这件事 ——
    瑜伽带练被判成 dialogue、舞台口播被判成 teaching。
    """
    kind = str(ir.get("meta", {}).get("content_kind") or "").strip().lower()
    if kind == "action":
        return True
    if kind in ("speech", "talking", "dialogue"):
        return False
    return looks_like_action(ir)


def _render_action_physics(
    ir: Mapping[str, Any],
    handles: "HandleResolver",
    library: ph.PhraseLibrary,
) -> list[str]:
    """动作与物理：**只有"有动作要复刻"的片子才写这一段**。

    用户 2026-10-07 两条反馈：

    1. 舞台纯口播的视频里**不该**出现这一段（人物只是站着说话 + 做展示手势）；
    2. 动作类/体式类/教学动作类**必须**有一句 1:1 复刻动作的话，原话大意：
       「视频里老师和学员的动作参考这个视频，1:1复刻视频里的肢体轨迹、
        身体角度、抬手弯腰幅度，不出现任何其他动作，动作自然连贯。」

    句子本身取自短语资产 ``T-ACTION``（单一来源，改措辞改资产、不改代码）；
    这里只负责决定"写不写"和填槽。
    """
    if not needs_action_physics(ir):
        return []

    persons = ir.get("locked", {}).get("B2_person", [])
    names = [
        str(p.get("display_name") or p.get("name") or "").strip()
        for p in persons
    ]
    who = "、".join(n for n in names if n) or "画面中人物"

    # 参考动作视频的句柄（``[参考动作视频1]``）。``to_ir`` 会为动作类自动登记这条素材；
    # 老数据（早于本轮）没有这条素材，退化成 ``[视频1]`` ——
    # 它符合 ``[视频N]`` 句柄格式，G1 不会因为"未声明的句柄"阻断（见 gates）。
    source = _video_handle(ir, handles) or "[视频1]"
    lines: list[str] = []
    template = library.templates.get("T-ACTION")
    if template is not None:
        try:
            lines.append(template.render(who=who, 参考视频句柄=source))
        except ValueError:  # pragma: no cover - 资产配置错误
            lines = []
    if not lines:
        lines.append(
            f"{who}的动作轨迹参考{source}，1:1复刻其中的肢体轨迹、身体角度、"
            f"抬手弯腰幅度，不出现任何其他动作，动作自然连贯。"
        )
    lines.append(
        f"{who}全程保持原有动作轨迹与幅度，肢体比例正常、动作连贯不变形；"
        f"不出现扭曲手指、多手指、错位关节或身份漂移。"
    )
    return lines


def _video_handle(ir: Mapping[str, Any], handles: "HandleResolver") -> str:
    """交付物里那条**参考动作视频**的句柄（``[参考动作视频1]``）。

    优先取带自定义标签（``token_label``）的那条 —— 动作类片子会由 ``to_ir``
    自动建一条 ``参考动作视频`` 素材，用户自己在生成平台把它传上去。
    """
    candidates = [
        ref for ref in ir.get("references", [])
        if str(ref.get("seedance_role") or "") == "reference_video"
        or str(ref.get("type") or "") == "video"
    ]
    if not candidates:
        return ""
    preferred = [ref for ref in candidates if str(ref.get("token_label") or "").strip()]
    for ref in (preferred or candidates):
        slots = [str(s) for s in (ref.get("slots") or []) if str(s).strip()]
        for candidate in (str(ref.get("placeholder") or ""), *slots):
            token = handles.token_for(candidate) if candidate else ""
            if token:
                return token
    return ""


def _render_quality(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
    *,
    include_preset_text: bool = True,
) -> list[str]:
    """画质相关内容。

    **用户 2026-09-29 明确全部去掉**，原因是两条：

      1. 「画质：auto white balance slightly cool, …」是**英文**，用户不要英文；
      2. 「画质档位：P1」是个内部代号，用户看不懂，也没法直接用。

    保留函数是因为 ``locked.B5_quality`` 仍是契约字段，
    且以后可能按平台要求重新引入 —— 现在是**显式不渲染**，不是忘了。
    """
    _ = (ir, library, include_preset_text)
    return []


_LATIN_RUN = _re.compile(r"[A-Za-z]{3,}")
_LATIN_WITH_SEP = _re.compile(r"[A-Za-z]{3,}\s*[、，]?")
_CJK_CHAR = _re.compile(r"[\u4e00-\u9fff]")


def _drop_english(lines: Sequence[str]) -> list[str]:
    """摘掉正文里的英文**片段**（用户 2026-09-29 明确要求不要英文）。

    **不能整行删**（实测踩过）：原来的判据是"含 ≥3 个连续拉丁字母的整行直接丢"，
    于是「场景始终为[图3]的演播厅环境，背景有大屏PILATES标识，保持同一位置和
    光线氛围」这种**中文句子**会因为一个品牌名被整句丢掉 —— 而场景锚定是 LOCK
    永不可裁字段，交付物会凭空少一整句、还没有任何提示。

    现在的做法：先把拉丁词摘掉，摘完**一个汉字都不剩**才丢整行
    （真正纯英文的行仍然被丢掉，这正是当初要的效果）。
    """
    out: list[str] = []
    for line in lines:
        if not _LATIN_RUN.search(line):
            out.append(line)
            continue
        cleaned = _LATIN_WITH_SEP.sub("", line)
        # 收拾落单的顿号：「禁止出现bgm、logo、水印」→「禁止出现水印」
        cleaned = _re.sub(r"[、，]\s*(?=[、，。；]|$)", "", cleaned)
        cleaned = _re.sub(r"[（(]\s*[)）]", "", cleaned)
        cleaned = _re.sub(r"\s{2,}", " ", cleaned).strip()
        # 英文占了大半、剩下的汉字又只有个标签（「画质：」「参数：」）→ 仍按英文行丢掉，
        # 这是当初"整行删"想达到的效果；而中文句子里的一个品牌名远达不到这个比例。
        removed = sum(len(run) for run in _LATIN_RUN.findall(line))
        if len(line) and removed / len(line) > 0.5 and len(_CJK_CHAR.findall(cleaned)) < 4:
            continue
        if _CJK_CHAR.search(cleaned):
            out.append(cleaned)
        # 一个汉字都不剩 → 整行本来就是英文，丢掉
    return out


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
def _render_beats(ir: Mapping[str, Any]) -> list[str]:
    """逐镜头写一行，**前缀一律是「镜头N：」**。

    **不再写秒数**（用户 2026-09-29 第三轮）：「0-9.102秒画面：」这种前缀
    既难读、又是生成平台用不上的信息 —— 时长在生成平台是独立控件。
    一镜到底的片子本来只有一镜，写「镜头1：」也不会歧义；
    而【对白与口型】本来就是按「镜头1（人物1）」写的，前面也得对得上。

    **机位角度与运镜都写进这一行**（用户 2026-09-29 要求）：
    例如「中全景，平视机位拍摄，固定机位拍摄，室内健身房…」。
    用户明确要求把机位从「关键一致性」里搬到这里 —— 逐镜头才说得清，
    而且放在一起读起来就是一个完整的分镜句。

    **场景只在变化时写**（用户 2026-09-29 第五轮）：
    「场景始终为[图N]的…环境，保持同一位置和光线氛围」已经在【关键一致性】里钉过，
    节拍行再把模型逐镜头写的 location 抄一遍就是纯冗余 ——
    实测一条四镜头的片子里，「同一室内演播厅舞台，背景弧形大屏与灯带不变」
    被抄了 3 遍。现在**同一个地方只说一次**（第一次带上细节），
    真的换场景时才再写一句。

    **不做节拍截断**。曾经这里有个 ``max_body_chars`` 参数把 ``blocking``
    截到 60 字，但它不在框架 §9.2 的压缩清单里，而且只截一个字段、
    承诺与交付物不一致。实测节拍各字段本来就只有 13~33 字，没有可裁的散文。
    """
    lines: list[str] = []
    # **没有台词的片子不写「语气」**（实测踩过）：语气是"话是怎么说的"，
    # 参考视频里没人说话时写「语气专注」是在描述不存在的声音。
    # 情绪本身不丢 —— 它就在同一行的 ``expression``（「表情专注」）里。
    has_dialogue = any(
        str((beat.get("dialogue") or {}).get("line") or "").strip()
        for beat in ir.get("creative", {}).get("C2_beats", [])
    )
    spoken_locations: list[str] = []
    for index, beat in enumerate(ir.get("creative", {}).get("C2_beats", []), start=1):
        framing = str(beat.get("framing") or "").strip()
        angle = str(beat.get("camera_angle") or "").strip()
        camera = str(beat.get("camera") or "").strip()
        location = _new_location(str(beat.get("location") or "").strip(), spoken_locations)
        blocking = str(beat.get("blocking") or "").strip()
        expression = str(beat.get("expression") or "").strip()
        emotion = str(beat.get("emotion") or "").strip()
        prefix = f"镜头{beat.get('shot_id') or index}："

        # 「中全景，平视机位拍摄，固定机位拍摄」
        head_parts = [p for p in (framing,) if p and p != UNKNOWN]
        if angle and angle != UNKNOWN:
            head_parts.append(f"{angle}机位拍摄")
        if camera and camera != UNKNOWN:
            # 模型的 ``camera`` 字段**不一定**是运镜：实测出现过「过肩」
            # （那是机位/构图）。放进运镜槽位会读成"镜头在过肩移动"，
            # 所以按词判定槽位，而不是一律当运镜写。
            if _is_movement(camera):
                head_parts.append(_motion_phrase(camera))
            elif "构图" not in camera:
                head_parts.append(f"{camera}构图")
        head = "，".join(head_parts)
        if location:
            head = f"{head}，{location}" if head else location

        # 技术头 / 动作 / 表情 / 情绪，逐段成句，避免标点拼接出错。
        # UNKNOWN 不写进正文：把"没反推出来"写成 unknown 是往提示词里灌噪声。
        segments = [
            _text_or_empty(head), _text_or_empty(blocking),
            _text_or_empty(expression),
        ]
        if has_dialogue and _text_or_empty(emotion):
            segments.append(f"语气{_text_or_empty(emotion)}")
        body = "，".join(s.rstrip("。，") for s in segments if s)
        if not body:
            # 这个镜头什么都没反推出来：跳过，不产出 "镜头N：unknown。" 这种垃圾行
            continue
        lines.append(f"{prefix}{body}。")
    return lines


def _text_or_empty(value: str) -> str:
    return "" if value in UNKNOWN or value is None else str(value).strip()


# --------------------------------------------------------------------------
# 场景去重：同一个地方只说一次
# --------------------------------------------------------------------------
_SAME_SCENE_MARKERS = ("同一", "同上", "同前", "如前", "不变", "保持一致")
"""模型亲口说"还是那个地方"的措辞。"""

_SAME_SCENE_OVERLAP = 4
"""两段场景描述共享多长的**连续**文字就算同一个地方。

实测依据：两条多人片子里「同一场景」都表现为共享 5 字地点短语 ——
「室内演播厅」「工作室前台」；而真正换了地方的描述共享不到这么长的片段
（中文里 4 字以上的连续重合基本就是地名/场所名，不是巧合）。
"""


def _new_location(candidate: str, written: list[str]) -> str:
    """这一镜的场景值不值得再写一遍。

    场景已经在【关键一致性】里钉死（「场景始终为[图N]的…环境」），
    所以这里只回答一个问题：**换地方了吗**。没换就返回空串，
    调用方就不会把它拼进节拍行。
    """
    if not candidate:
        return ""
    for seen in written:
        if _looks_like_same_scene(candidate, seen):
            return ""
    written.append(candidate)
    return candidate


def _looks_like_same_scene(candidate: str, established: str) -> bool:
    if any(marker in candidate for marker in _SAME_SCENE_MARKERS):
        return True
    return _longest_common_run(candidate, established) >= _SAME_SCENE_OVERLAP


def _longest_common_run(a: str, b: str) -> int:
    """最长公共**连续**子串的长度（短句够用）。"""
    if not a or not b:
        return 0
    best = 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def _fmt_sec(value: float) -> str:
    return f"{value:g}"


_MOTION_PHRASES = {
    "固定": "固定机位拍摄",
    "推近": "推近拍摄",
    "拉远": "拉远拍摄",
    "跟拍": "跟拍",
    "横摇": "横摇拍摄",
    "手持": "手持拍摄",
    "斯坦尼康跟拍": "跟拍",
    "固定机位全程不变": "固定机位拍摄",
    "稳定推近": "推近拍摄",
    "手持自拍": "手持拍摄",
}


def _motion_phrase(value: str) -> str:
    """运镜 → 句中说法（「固定」→「固定机位拍摄」）。未知值原样返回。"""
    return _MOTION_PHRASES.get(value, value)


_MOVEMENT_HINTS = (
    "固定", "推", "拉", "摇", "移", "跟", "手持", "升降", "环绕", "变焦", "晃动", "稳定",
)
"""词里带这些字才当运镜。实测踩过：模型把「过肩」写进 ``camera`` 字段，
而它是机位/构图，不是镜头怎么动。"""


def _is_movement(value: str) -> bool:
    return any(hint in value for hint in _MOVEMENT_HINTS)


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
    # 口型要求那句（「中文普通话，语速中等，吐字清晰」）用户 2026-09-29 明确去掉：
    # 它是一条无对象的泛化要求，对生成没有约束力。LIPSYNC_SENTENCE 保留 ——
    # 那句是"口型必须与台词同步"，属于有效约束。
    return lines + [LIPSYNC_SENTENCE]


_PROCESS_NOISE = (
    "取证", "非字幕", "分屏", "画中画", "属场景内元素", "确认为", "经查",
    "完整单画面", "不是字幕", "经核对",
)
"""**我们自己的分析口径**，不是画面内容 —— 不写进交付物。

用户 2026-10-08：「目前你给出的信息太复杂，看不懂 …… 目前你给出的括号里的内容就不要了。」
实测那条扩展段里就混着「经取证为完整单画面，屏幕属场景内元素，非字幕、分屏或画中画」——
那是我们判断"要不要单开一段"的理由，对生成模型毫无用处，用户也看不懂。
它仍然留在报告里（`reviews`/`evidence`）可追溯，只是不进提示词。
"""

_CONFIRM_NOTE = "这一处请人工确认后修改。"


def _render_extensions(ir: Mapping[str, Any]) -> list[str]:
    """渲染扩展段：**大白话描述画面里读到了什么**。

    用户 2026-10-08 的要求（原来那版太像工程报告）：

    - 直接写**AI 读到的画面信息**：有什么字、怎么排版、有什么东西；
    - **没读到的要说清原因**，并给出能读到的部分；
    - 结尾**提示人工确认修改**；
    - **括号里的依据不要了**（那是我给自己的解释）；
    - **"以下画面文字不要生成"这句前缀也不要**（用户 2026-10-08 追加）。
      去掉它是安全的：【禁止项】里本来就有「全程不生成任何字幕文字，100%透明字幕。」
      与「禁止出现…批注文字或设计说明元素。」—— D0 是全局约束，
      在扩展段里再写一遍只是重复。带「」的文字本身也看得出是画面上的字。
    """
    lines: list[str] = []
    for item in ir.get("creative", {}).get("C3_extensions", []) or []:
        if not isinstance(item, Mapping):
            continue
        title = str(item.get("title") or "").strip()
        body: list[str] = []
        for raw_line in item.get("lines") or []:
            text = str(raw_line).strip()
            if not text or any(word in text for word in _PROCESS_NOISE):
                continue
            # `_drop_english` 收一组行、返回一组行（摘掉英文片段）；这里处理单行
            cleaned = _drop_english([text])
            if cleaned:
                body.append(cleaned[0])
        unreadable = str(item.get("unreadable") or "").strip()
        if not title or not (body or unreadable):
            continue
        lines.append(f"［扩展］{title}")
        lines.extend(body)
        if unreadable:
            # 没看清的部分：说清**是什么**、**为什么没看清**，然后请人工确认
            lines.append(f"{_drop_english([unreadable])[0]}。{_CONFIRM_NOTE}")
        elif any(_QUOTED_TEXT.search(line) for line in body):
            # 读到了画面文字：这类地方最容易写错一个字，提醒人工核对
            lines.append(_CONFIRM_NOTE)
    return lines


_QUOTED_TEXT = _re.compile(r"[\"“”「」『』][^\"“”「」『』]{2,}[\"“”「」『』]")


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


PLACEHOLDER_RE = _re.compile(r"【([^】]+)】")
"""``【素材位名】`` 的唯一定义。

**不设长度上限**（实测踩过）：这里原来写 ``{1,24}``、而闸门 G1 的残留检测写
``{1,40}``，两个数字不一样。素材名超过 24 字时解析不了（正文里留着 ``【…】``），
超过 40 字时连 G1 也检测不到 —— 正文带着裸占位符交付，G1 还报「残留=0」。
红线 6 要求残留必须为 0，判据就不该有"够长就看不见"的死角。
"""


def _replace_placeholders(text: str, handles: HandleResolver) -> str:
    """解析期把 ``【xxx】`` 换成 ``[图N]``（框架 §2.1 两步转换的第二步）。"""
    def swap(match: "_re.Match[str]") -> str:
        return handles.token_for(match.group(1), fallback=match.group(0))

    return PLACEHOLDER_RE.sub(swap, text)


_DECLARED_TOKEN_RE = _re.compile(r"\[(?:图|视频|音频)\d+\]")


# --------------------------------------------------------------------------
# 人称统一：正文里一律用「人物N」
# --------------------------------------------------------------------------
_BRACE_SEGMENT_RE = _re.compile(r"(〔[^〕]*〕)")
"""``〔灰色运动外套女子 · 画面左侧〕`` 这种"人物N 是谁"的定义段。"""


def person_label_map(ir: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """``人物名写法 → 人物N``，**长写法排在前面**。

    用户 2026-09-29（第三轮）要求：「关键一致性里的人物1灰色运动外套女子、
    人物2米色开衫女子，已经给人物1和人物2标定了特征，所以下面的提示词
    直接用人物1和人物2标定就行了」。

    也就是说：``人物1〔灰色运动外套女子〕`` 已经把特征标定过了，
    后面就不必把「灰色运动外套女子」这种长描述再抄 5 遍 ——
    台词、音色、动作、镜头节拍统一用 ``人物1``。

    返回元组而不是 dict：调用方要的正是"长的先替换"这个顺序
    （短写法是长写法的子串时，先换长的那一次才不会被切碎）。
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, person in enumerate(ir.get("locked", {}).get("B2_person", []), start=1):
        label = f"人物{index}"
        for key in ("display_name", "name"):
            name = str(person.get(key) or "").strip()
            # 单字写法（"女"）做替换会把别的词一起吃进去，只收 2 字以上
            if len(name) < 2 or name == label or name in seen:
                continue
            seen.add(name)
            pairs.append((name, label))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return tuple(pairs)


def unify_person_labels(
    lines: Sequence[str],
    aliases: Sequence[tuple[str, str]],
) -> list[str]:
    """把行里的人名换成 ``人物N``，但 **〔…〕 里的外观描述原样保留**。

    〔…〕 就是"人物N 是谁"的定义，连它一起换掉的话 ``人物1`` 就没有指代对象了。

    换完还要**收掉「人物N（人物N）」这种重复**：模型（或自检回路）自己就会写
    「人物2（深灰色上衣女子）微笑」，括号里的名字被同一轮替换成 ``人物2`` 之后
    就变成了「人物2（人物2）微笑」—— 实测在生产链路里出现过。
    """
    if not aliases:
        return list(lines)
    out: list[str] = []
    for line in lines:
        parts = _BRACE_SEGMENT_RE.split(line)
        for position, part in enumerate(parts):
            if part.startswith("〔"):
                continue
            for name, label in aliases:
                if name in part:
                    part = part.replace(name, label)
            parts[position] = part
        out.append(_DUP_LABEL_RE.sub(r"\1", "".join(parts)))
    return out


_DUP_LABEL_RE = _re.compile(r"(人物(\d+))\s*[（(]\s*人物\2\s*[）)]")
"""同号重复：``人物2（人物2）`` → ``人物2``。**号不同不合并**（那是有信息量的）。"""


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
#
# 这里**曾经多出第 ④ 步「节拍里的散文式场景描写，截断至 60 字」** ——
# 它不在框架 §9.2 的五条清单里，是实现时自己加的；而且它只截 `blocking`
# 一个字段，`location`/`expression` 都没截，于是承诺与交付物不一致
# （报告写着"已截断至 60 字"，实际节拍行 79~124 字）。已删除。
#
# 删除前实测过该步有没有价值：节拍各字段的实际长度是
# location 17~33 字 / blocking 15~32 字 / expression 13~24 字，
# **本来就没有可裁的散文**，这一步省 0 字。留着只会污染压缩报告。
COMPRESSION_STEPS: tuple[str, ...] = (
    "quality_preset_text",   # ① B5 次要画质描述：保留预设名，丢弃展开句
    "person_longtail",       # ② B2 人物长尾：保留「参考[图N]锁定五官脸型」
    "forbidden_low_freq",    # ③ D 区低频条目（D5/D6）
)


@dataclass(slots=True)
class _CompressionPlan:
    """一次压缩的完整决策。组装器据此渲染，而不是渲染完再改文本 —— 后者容易漏改。"""

    quality_preset_text: bool = True
    person_longtail: bool = True
    forbidden_low_freq: bool = True
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
        forbidden_low_freq=True,
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
    #    用户 2026-09-29 明确去掉这一行（「9:16 竖屏，15.07 秒，720p，生成同步音频」）：
    #    比例/时长/分辨率在生成平台上是**独立控件**，写进提示词正文是冗余。
    #    是否写由 ``delivery.include_param_header`` 决定（反推默认置 False）。
    if with_param_header:
        meta = ir.get("meta", {})
        doc.param_header = PARAM_HEADER_TEMPLATE.format(
            ratio=str(meta.get("aspect_ratio") or "9:16"),
            duration=_fmt_sec(effective_duration(ir)),
            resolution="720p",
        )

    # ③~⑨ 关键一致性
    locked_lines: list[str] = []
    locked_hints: list[str] = []
    locked_lines.extend(_render_scene(ir, handles))
    person_lines, person_hints = _render_person(ir, handles, library)
    locked_lines.extend(person_lines)
    # 群体句紧跟在人物句后面：它说的就是"主体之外的其他人"
    locked_lines.extend(_render_ensemble(ir))
    locked_hints.extend(person_hints)
    # 眨眼节拍与皮肤质感是**无条件固定项**（用户 2026-09-29 指定）：
    # 它们是最容易被模型做错的默认项（眨眼频率不自然、过度磨皮），
    # 所以不管反推有没有人物条目，交付物里都必须出现这一句。
    locked_lines.extend(_render_universal_person_phrases(ir, library))
    locked_lines.extend(_render_blocking(ir))
    locked_lines.extend(_render_camera(ir))
    locked_lines.extend(_render_quality(ir, library,
                                        include_preset_text=plan.quality_preset_text))
    # 音色**单独成段**（SEC_AUDIO），但素材声明要一起算 —— 见下面 declared。
    audio_lines = _render_audio(ir, handles)
    if not plan.person_longtail:
        locked_lines = [PERSON_LONGTAIL_RE.sub("。", line) for line in locked_lines]
    # 画质档位行不再追加（_render_quality 已返回空）—— 见它的注释
    if locked_lines:
        doc.sections.append(PromptSection(
            title=fw.SEC_LOCKED, lines=_drop_english(locked_lines), sources=["locked"],
            hints=locked_hints,
        ))

    # ② 【参考素材】清单段**已按用户要求删除**（2026-09-29，第三轮）。
    #    原来这里插一段「[图1] 人物1多视图 / [音频1] 人物1音频…」。
    #    删掉的理由：素材的**用途**本来就在正文里逐句声明
    #    （「…形象参考[图1]，用于…」「…音色参考[音频1]」），
    #    清单只是把句柄和名字再抄一遍，对生成没有约束力。
    #    正文句柄不受影响 —— 句柄由 HandleResolver 统一编号，与这段无关。
    #
    #    仍然算一遍 declared：它是"素材有没有在正文里被声明用途"的体检口径，
    #    下面的 warning 与闸门 G12 都读它（音色段也要算进来：音频句柄是在那里声明的）。
    declared = declared_tokens("\n".join(locked_lines + audio_lines))
    missing_roles = [t for t in handles.all_tokens if t not in declared]

    # 剧情
    logline = str(ir.get("creative", {}).get("C1_logline") or "").strip()
    if logline:
        doc.sections.append(PromptSection(
            title=fw.SEC_LOGLINE, lines=[logline], sources=["creative.C1_logline"],
        ))

    # 镜头节拍
    beats = _render_beats(ir)
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

    # 音色参考（永不可裁）—— 紧跟台词，用户才能一眼看出"谁说话、用谁的声音"
    # （audio_lines 已在上面的素材声明处渲染过，这里直接复用，不重复渲染）
    if audio_lines:
        doc.sections.append(PromptSection(
            title=fw.SEC_AUDIO, lines=audio_lines, sources=["locked.B6_audio"],
        ))

    # 动作与物理（正面约束动作不变形）
    action_lines = _render_action_physics(ir, handles, library)
    if action_lines:
        doc.sections.append(PromptSection(
            title=fw.SEC_ACTION, lines=action_lines,
            sources=["locked.B2_person", "creative.C2_beats[].blocking"],
        ))

    # 扩展段：**框架外信息**的落脚点（屏上文案、分屏布局、变装点…）。
    # 放在【动作与物理】之后、【禁止项】之前：既不打乱用户确认过的段落顺序，
    # 也不让"禁止项"这个收尾段被挤到中间。
    extension_lines = _render_extensions(ir)
    if extension_lines:
        doc.sections.append(PromptSection(
            title=fw.SEC_EXTENSIONS, lines=extension_lines,
            sources=["creative.C3_extensions"],
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

    # 人称统一：`人物1〔灰色运动外套女子〕` 已经把特征标定过了，
    # 后面的正文一律用「人物1/人物2」（用户 2026-09-29 第三轮要求）。
    # **只换 lines，不换 hints** —— hints 是给用户看的"这行可以换成什么"，
    # 里面的素材名恰恰要让用户看清楚，换成人物N 反而看不懂了。
    aliases = person_label_map(ir)
    for section in doc.sections:
        section.lines = unify_person_labels(section.lines, aliases)

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