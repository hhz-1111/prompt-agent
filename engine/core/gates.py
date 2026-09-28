"""交付自检：14 项闸门（框架 §11、PRD §6.4.4）。

这是"一遍成功"的保障机制。设计纪律：

1. **纯函数**：输入 IR + 组装结果，输出结论。不读磁盘、不发请求、不改状态。
2. **逐项给结论**：每一项都可以是 ok / warn / block，不存在"整体还行"。
3. **修复必须可执行**：block 项一定附带具体动作，不写"请检查提示词"这种废话。
4. **有阻断项就不许标记可交付**（PRD F4.4）。

G8 是最容易被忽略但最致命的一项：把 30 秒的内容塞进一条 15 秒提示词，
模型必然丢细节 —— 这是"抽卡"的头号原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import framework as fw
from . import phrases as ph
from .assembler import PromptDocument, effective_duration
from .types import SplitStrategy, UNKNOWN

# 状态
OK = "ok"
WARN = "warn"
BLOCK = "block"

# 密度上限（G8）：官方边界 —— 15 秒对应 2~4 个清晰节拍
DENSITY_LIMIT = 0.27

# 互斥模式：首帧 / 首尾帧 / 多模态参考 三者不可混用
EXCLUSIVE_MODES = ("multimodal_reference", "first_frame", "first_last_frame")


@dataclass(frozen=True, slots=True)
class GateResult:
    """一项闸门的结论。"""

    code: str
    name: str
    status: str
    detail: str = ""
    suggestion: str = ""
    autofixed: bool = False
    evidence: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        return self.status == BLOCK

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "suggestion": self.suggestion,
            "autofixed": self.autofixed,
            "evidence": list(self.evidence),
        }


@dataclass(slots=True)
class GateReport:
    """14 项闸门的完整报告。"""

    results: list[GateResult] = field(default_factory=list)

    def add(self, result: GateResult) -> None:
        self.results.append(result)

    @property
    def blockers(self) -> list[GateResult]:
        return [r for r in self.results if r.blocking]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if r.status == WARN]

    @property
    def all_green(self) -> bool:
        return not self.blockers and not self.warnings

    @property
    def deliverable(self) -> bool:
        """有阻断项时不允许标记"可交付"（PRD F4.4 / 红线 10）。"""
        return not self.blockers

    @property
    def verdict(self) -> str:
        if self.blockers:
            return "BLOCKED"
        if self.warnings:
            return "PASS_WITH_WARNINGS"
        return "ALL_GREEN"

    def by_code(self, code: str) -> GateResult | None:
        for item in self.results:
            if item.code == code:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "deliverable": self.deliverable,
            "all_green": self.all_green,
            "blocker_count": len(self.blockers),
            "warning_count": len(self.warnings),
            "results": [r.to_dict() for r in self.results],
        }

    def summary_lines(self) -> list[str]:
        """PRD §5 步骤 5 里那种紧凑的自检结果块。"""
        lines: list[str] = []
        for item in self.results:
            mark = {OK: "✅", WARN: "⚠️", BLOCK: "❌"}[item.status]
            lines.append(f"{item.code:<4} {item.name:<16} {mark}  {item.detail}")
        lines.append(f"→ {'允许交付' if self.deliverable else '阻断交付'}")
        return lines


# --------------------------------------------------------------------------
# 闸门实现
# --------------------------------------------------------------------------
def g1_placeholder_residue(doc: PromptDocument, ir: Mapping[str, Any]) -> GateResult:
    """G1 ``【】`` 残留 = 0。

    只扫**正文**（不含段落标题）—— 段落标题本身就写作 ``【关键一致性】``，
    那是契约的固定格式，不是未绑定的占位符。
    """
    residues = _find_residues(doc.body_text)
    unbound = [
        str(ref.get("placeholder") or ref.get("slots") or "")
        for ref in ir.get("references", [])
        if not ref.get("asset_id")
    ]
    if not residues:
        return GateResult("G1", "【】残留", OK, "残留=0", evidence=())
    detail = f"残留 {len(residues)} 处：{'、'.join(residues[:5])}"
    if unbound:
        detail += f"；未绑定素材 {len(unbound)} 个"
    return GateResult(
        "G1", "【】残留", BLOCK, detail,
        suggestion="完成素材绑定；每个占位符都必须绑定到中台已预处理的真实素材句柄。",
        evidence=tuple(residues[:8]),
    )


def g2_quota(
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
) -> GateResult:
    """G2 素材配额（按探测到的模型能力校验）。"""
    model = str(ir.get("meta", {}).get("target_model") or "seedance-2.0")
    try:
        capacity = library.capacity_for(model)
    except KeyError:
        return GateResult(
            "G2", "素材配额", BLOCK,
            f"能力矩阵中没有模型 {model}",
            suggestion="补全 assets/phrases/phrases.json 的 capacity.models 配置。",
        )

    counts = {"image": 0, "video": 0, "audio": 0}
    from .assembler import ROLE_OF_REF, _safe_ref_type

    for ref in ir.get("references", []):
        role = ROLE_OF_REF.get(_safe_ref_type(ref.get("type")))
        if role is None:
            continue
        key = role.value.replace("reference_", "")
        counts[key] = counts.get(key, 0) + 1

    overflows: list[str] = []
    for key, count in counts.items():
        bounds = capacity.get(key)
        if not bounds:
            continue
        low, high = bounds
        if count > high:
            overflows.append(f"{_role_cn(key)} {count} 个 > 上限 {high}")
    total = sum(counts.values())
    mixed_cap = int(capacity.get("mixed_total", 0))
    if mixed_cap and total > mixed_cap:
        overflows.append(f"混合输入总数 {total} > 上限 {mixed_cap}")

    if overflows:
        return GateResult(
            "G2", "素材配额", BLOCK,
            "；".join(overflows),
            suggestion=f"减少素材数量，或改用更高配额的模型（2.5 支持图1~30/视0~10/音0~10）。",
            evidence=tuple(overflows),
        )
    return GateResult(
        "G2", "素材配额", OK,
        f"{counts['image']}图/{counts['video']}视/{counts['audio']}音",
    )


def g3_exclusive_modes(doc: PromptDocument, ir: Mapping[str, Any]) -> GateResult:
    """G3 互斥模式不得混用。"""
    mode = str(ir.get("delivery", {}).get("first_frame_mode") or "multimodal_reference")
    text = doc.full_text()
    used_refs = len(ir.get("references", [])) > 0
    signals: list[str] = []
    if used_refs:
        signals.append("多模态参考")
    if "first_frame" in text or "首帧" in text:
        signals.append("首帧")
    if "return_last_frame" in text or "首尾帧" in text or "尾帧" in text:
        signals.append("首尾帧")

    conflicts: list[str] = []
    if mode == "first_last_frame" and used_refs:
        conflicts.append("首尾帧模式与多模态参考素材同时出现")
    if mode == "first_frame" and used_refs:
        conflicts.append("首帧模式与多模态参考素材同时出现")

    if conflicts:
        return GateResult(
            "G3", "互斥模式", BLOCK,
            "；".join(conflicts),
            suggestion=(
                "三选一：走多模态参考 → 人物一致性最好；"
                "走首尾帧 → 跨段连续性最好。当前策略不可兼得。"
            ),
            evidence=tuple(signals),
        )
    return GateResult("G3", "互斥模式", OK, f"当前 {mode}，无混用")


def g4_contradiction(doc: PromptDocument, ir: Mapping[str, Any]) -> GateResult:
    """G4 正反矛盾。

    注意 PRD §6.2.5 的裁决：``着装100%还原@人物多视图`` 与
    ``禁止"服装保持一致"`` **不矛盾** —— 前者有具体参考对象（确定约束），
    后者禁的是无对象的泛化表述。这里必须实现这个例外，否则会误报。
    """
    text = doc.full_text()
    forbidden_text = "\n".join(str(f.get("text") or "") for f in ir.get("forbidden", []))
    conflicts: list[str] = []

    # 具体约束 vs 泛化禁止：共存，不算矛盾
    has_concrete_wardrobe = "着装100%还原" in text or "服装身形匹配参考图" in text
    generic_ban = "服装保持一致" in forbidden_text or "参考图延续" in forbidden_text
    if has_concrete_wardrobe and not generic_ban:
        conflicts.append("写了具体着装约束，但缺少 D4 泛化约束防补条目")

    # 语义互斥：一镜到底 vs 推近
    if ("一镜到底" in text or "机位全程不变" in text) and "缓慢推近" in text:
        if not ("一镜到底" in text and "轻微推近" in text):
            conflicts.append("「一镜到底/机位全程不变」与「缓慢推近」语义互斥")

    # 数值冲突：眨眼节拍
    blinks = [p.get("blink") for p in ir.get("locked", {}).get("B2_person", [])]
    if len({b for b in blinks if b}) > 1:
        conflicts.append(f"眨眼节拍冲突：{blinks}")

    if conflicts:
        return GateResult(
            "G4", "正反矛盾", BLOCK,
            "；".join(conflicts),
            suggestion="逐条裁决：语义互斥需择一；数值冲突需统一；泛化防补条目请补 D4。",
            evidence=tuple(conflicts),
        )
    return GateResult("G4", "正反矛盾", OK, "无冲突")


def g5_dialogue_quotes(doc: PromptDocument, ir: Mapping[str, Any]) -> GateResult:
    """G5 台词双引号 —— 自动修复。"""
    fixed: list[str] = []
    from .assembler import ensure_quoted

    for beat in ir.get("creative", {}).get("C2_beats", []):
        dialogue = beat.get("dialogue") or {}
        line = str(dialogue.get("line") or "").strip()
        if line and ensure_quoted(line) != line:
            fixed.append(line)
    if fixed:
        return GateResult(
            "G5", "台词双引号", OK,
            f"已自动修复 {len(fixed)} 条（方舟官方建议双引号包裹以优化音频生成）",
            autofixed=True,
            evidence=tuple(fixed[:5]),
        )
    return GateResult("G5", "台词双引号", OK, "全部双引号")


def g6_lipsync_binding(ir: Mapping[str, Any]) -> GateResult:
    """G6 口型与说话人一致。"""
    audio_speakers = {
        str(track.get("speaker") or "").strip()
        for track in ir.get("locked", {}).get("B6_audio", [])
        if str(track.get("speaker") or "").strip()
    }
    dialogue_speakers = {
        str((beat.get("dialogue") or {}).get("speaker") or "").strip()
        for beat in ir.get("creative", {}).get("C2_beats", [])
        if str((beat.get("dialogue") or {}).get("line") or "").strip()
    }
    dialogue_speakers.discard("")

    if not dialogue_speakers:
        return GateResult("G6", "口型与说话人", OK, "无对白，无需绑定")

    unbound = sorted(dialogue_speakers - audio_speakers)
    if unbound:
        return GateResult(
            "G6", "口型与说话人", BLOCK,
            f"说话人缺音色绑定：{'、'.join(unbound)}",
            suggestion=(
                "为每个有台词的角色绑定一条音色参考（一角色一音频）；"
                "否则口型与音色对不上。"
            ),
            evidence=tuple(unbound),
        )
    return GateResult("G6", "口型与说话人", OK, f"{len(dialogue_speakers)} 位说话人均已绑定")


def g7_char_budget(
    doc: PromptDocument,
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
) -> GateResult:
    """G7 字数预算（中文 ≤500 字**建议值**）。

    为什么是告警而不是阻断：PRD §6.4.4 表格里 G7 的处置写的是
    "按 §9.2 压缩并标注，或提示精简"，阻断清单里没有它；而
    §11 表格更明确写成"≤500 字建议值"。把它当硬阻断会挡住**所有**
    合法提示词 —— 实测 24 条真实爆款视频压到极限后仍在 650~850 字，
    因为永不可裁的部分（D0+D7 约 227 字、站位、音色、台词）本身就很长。

    组装阶段已按 §9.2 优先级压缩过。这里报的是**压完仍超出的部分**，
    需要用户自己精简创作区内容 —— 所以建议必须可执行。
    """
    model = str(ir.get("meta", {}).get("target_model") or "seedance-2.0")
    limit = int(library.capacity_for(model)["prompt_char_budget"])
    count = doc.char_count
    extra = doc.total_char_count - count
    detail = f"{count}/{limit}"
    if extra:
        detail += f"（+{extra} 字符英文/标点，不计入预算）"
    if count <= limit:
        return GateResult("G7", "字数预算", OK, detail)

    saved = sum(entry.saved_chars for entry in doc.compression)
    detail += f"，超出 {count - limit} 字"
    if saved:
        detail += f"（已按优先级压掉 {saved} 字）"
    return GateResult(
        "G7", "字数预算", WARN, detail,
        suggestion=(
            f"已自动压缩到极限，仍需手动精简约 {count - limit} 字。"
            "可从创作区入手：合并镜头节拍、精简动作描述、删减非必要禁止项。"
            "永不可裁：站位、音色、台词原文、D0 通用禁止项、D7 对白锁定。"
        ),
    )


def g8_density(ir: Mapping[str, Any]) -> GateResult:
    """G8 内容量与时长匹配 —— 节拍数 ÷ 时长 ≤ 0.27。

    这是**最容易被忽略但最致命**的一项：把 30 秒的内容塞进一条 15 秒提示词，
    模型必然丢细节 —— 这是"抽卡"的头号原因。

    注意判定的是**目标时长**（``duration``），不一定等于原片时长。
    原片是 9 秒 5 个快切，用户完全可以把目标时长拉到 20 秒来容纳同样多的节拍。
    """
    beats = list(ir.get("creative", {}).get("C2_beats", []))
    duration = effective_duration(ir)
    model = str(ir.get("meta", {}).get("target_model") or "seedance-2.0")

    if duration <= 0:
        return GateResult(
            "G8", "内容量匹配", BLOCK,
            "时长为 0，无法校验密度",
            suggestion="先用 ffprobe 取到真实时长（唯一可信来源）。",
        )
    if not beats:
        return GateResult(
            "G8", "内容量匹配", BLOCK,
            "没有节拍，无法校验密度",
            suggestion="完成镜头切分与节拍表反推。",
        )

    density = len(beats) / duration
    per_beat = duration / len(beats)
    detail = f"密度 {len(beats)}÷{duration:g}={density:.2f}（每节拍 {per_beat:.1f} 秒）"
    if density <= DENSITY_LIMIT:
        return GateResult("G8", "内容量匹配", OK, detail)

    # 给出**两个方向**的可执行方案：减节拍，或延长目标时长。
    # 只说"拆条"是误导 —— 拆条解决不了"单位时间内节拍过多"。
    allowed_beats = max(1, int(duration * DENSITY_LIMIT))
    needed_seconds = round(len(beats) / DENSITY_LIMIT, 1)
    clip_max = 15.0
    segments = max(1, -(-int(needed_seconds) // int(clip_max)))
    return GateResult(
        "G8", "内容量匹配", BLOCK,
        f"{detail} > {DENSITY_LIMIT}",
        suggestion=(
            f"这 {len(beats)} 个节拍对 {duration:g} 秒来说太密了（每节拍 {per_beat:.1f} 秒）。"
            f"二选一：① 把目标时长拉到 {needed_seconds:g} 秒（按 {clip_max:g} 秒上限拆成 {segments} 条逐条生成）；"
            f"② 把节拍合并到 {allowed_beats} 个以内。"
            f"原片快切不等于要逐镜复刻 —— 保住钩子与收尾，中间可以合并。"
        ),
    )


def g9_split_strategy(doc: PromptDocument, ir: Mapping[str, Any]) -> GateResult:
    """G9 拆条策略已选且与镜头模式相符。"""
    strategy = doc.strategy
    shot_mode = str(ir.get("meta", {}).get("shot_mode") or "one_take")
    matrix: dict[str, set[SplitStrategy]] = {
        "one_take": {SplitStrategy.S1_ONE_TAKE, SplitStrategy.S4_SAME_ANGLE,
                     SplitStrategy.S3_FIRST_LAST_FRAME},
        "multi_shot": {SplitStrategy.S2_PER_SHOT, SplitStrategy.S3_FIRST_LAST_FRAME},
        "same_angle_segments": {SplitStrategy.S4_SAME_ANGLE, SplitStrategy.S3_FIRST_LAST_FRAME},
    }
    allowed = matrix.get(shot_mode, set())
    if strategy not in allowed:
        return GateResult(
            "G9", "拆条策略", BLOCK,
            f"{strategy.value} 与镜头模式 {shot_mode} 不符",
            suggestion=f"镜头模式 {shot_mode} 下应选：{'、'.join(sorted(s.value for s in allowed))}",
        )
    return GateResult(
        "G9", "拆条策略", OK,
        f"{strategy.value}（{len(doc.plan)} 条）",
    )


def g10_locked_diff(
    ir: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
) -> GateResult:
    """G10 锁定位 diff 为空 —— 这是复刻的底线。"""
    if baseline is None:
        return GateResult(
            "G10", "LOCK diff", OK,
            "无基准可比（首次反推）",
            suggestion="改写过原始样本后，本项会逐字段比对锁定位。",
        )
    diffs = diff_locked(baseline, ir)
    if diffs:
        return GateResult(
            "G10", "LOCK diff", BLOCK,
            f"{len(diffs)} 处锁定位被改动",
            suggestion="锁定位一字不可动。若确需改动，那是新产品而不是复刻。",
            evidence=tuple(f"{path}: {before!r} → {after!r}" for path, before, after in diffs[:8]),
        )
    return GateResult("G10", "LOCK diff", OK, "LOCK diff 为空")


def g11_hook_preserved(
    ir: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
) -> GateResult:
    """G11 钩子类型保留。

    **基准缺失时的语义要说实话**：首次反推没有"原片基准"可比，
    这里只能确认"存在钩子节拍"，无法证明"与原片同类"。
    结论里会写明这一点，不假称已校验保留。
    """
    from .types import BeatRole

    hooks = [
        beat for beat in ir.get("creative", {}).get("C2_beats", [])
        if str(beat.get("beat_role") or "") == BeatRole.HOOK.value
    ]
    if not hooks:
        return GateResult(
            "G11", "钩子保留", BLOCK,
            "新文案中没有任何钩子节拍（hook）",
            suggestion=(
                "保留原片的钩子节拍类型（已归纳 11 种钩子类型）。"
                "钩子丢了，爆款结构就没了。"
            ),
        )
    if baseline is None:
        return GateResult(
            "G11", "钩子保留", OK,
            f"存在 {len(hooks)} 个钩子节拍（首次反推，无原片基准可比对类型）",
            suggestion="改写过文案后，本项会与反推基准逐条比对钩子类型是否同类。",
        )

    base_hooks = [
        beat for beat in baseline.get("creative", {}).get("C2_beats", [])
        if str(beat.get("beat_role") or "") == BeatRole.HOOK.value
    ]
    if base_hooks and len(hooks) < len(base_hooks):
        return GateResult(
            "G11", "钩子保留", BLOCK,
            f"钩子数量减少：{len(base_hooks)} → {len(hooks)}",
            suggestion="逐条核对钩子节拍是否被合并或删除。",
        )
    return GateResult(
        "G11", "钩子保留", OK,
        f"{len(hooks)} 个钩子节拍，与原片同类（基准 {len(base_hooks)} 个）",
    )


def g12_ref_roles(doc: PromptDocument, ir: Mapping[str, Any]) -> GateResult:
    """G12 参考素材角色已声明 —— 自动修复。

    判据是**用途声明的关键片段**是否出现在正文，而不是素材名有没有出现：
    ``[图3] 菜市场场景图`` 只是清单命名，真正声明用途的是
    正文里的 ``场景环境参考[图3]``。
    """
    from .assembler import DEFAULT_PURPOSE, HandleResolver, _ref_display_name, declared_tokens

    text = doc.full_text()
    declared = declared_tokens(text)
    handles = HandleResolver(ir)
    missing: list[str] = []
    fixes: list[str] = []
    for ref in ir.get("references", []):
        ref_type = _ref_type_of(ref)
        name = _ref_display_name(ref)
        token = handles.token_for(name)
        if token in declared:
            continue
        # 同一类型只要有一条已声明，就认为该类用途写法已确立
        purpose = DEFAULT_PURPOSE.get(ref_type, "")
        keyword = purpose.split("{token}")[0] if purpose else ""
        if keyword and keyword in text:
            continue
        missing.append(name)
        if purpose:
            fixes.append(purpose.format(token=token))
    if missing:
        return GateResult(
            "G12", "素材角色声明", OK,
            f"已自动补默认声明 {len(missing)} 项，含：{'、'.join(missing[:4])}",
            autofixed=True,
            suggestion="请确认素材用途声明是否准确。建议写入正文：" + "；".join(fixes[:3]),
            evidence=tuple(missing),
        )
    return GateResult("G12", "素材角色声明", OK, "每个引用都有用途")


def _ref_type_of(ref: Mapping[str, Any]):
    from .types import RefType

    try:
        return RefType(ref.get("type"))
    except (ValueError, TypeError):
        return RefType.MULTI_VIEW


def g13_mandatory_dimensions(
    doc: PromptDocument,
    ir: Mapping[str, Any],
    library: ph.PhraseLibrary,
) -> GateResult:
    """G13 必写维度齐全 —— 告警 + 补写建议。

    审的是**最终提示词正文**：IR 里存的是键名（如 ``blink="5s"``），
    逐字表述要到组装才展开。只审 IR 会把合规提示词误判成缺维度。
    """
    coverage = ph.check_mandatory_dimensions(ir, library, prompt_text=doc.body_text)
    missing = [item for item in coverage if not item.covered]
    implied = [item for item in coverage if item.covered and item.implied]
    hit = f"{len(coverage) - len(missing)}/{len(coverage)}"
    if not missing:
        detail = f"{hit} 全覆盖"
        if implied:
            detail += f"（{len(implied)} 项仅靠素材引用隐含成立）"
        return GateResult("G13", "必写维度", OK, detail)
    return GateResult(
        "G13", "必写维度", WARN,
        f"{hit}，缺：{'、'.join(item.label for item in missing)}",
        suggestion="；".join(item.suggestion for item in missing[:3]),
        evidence=tuple(item.label for item in missing),
    )


def g14_constraint_strength(
    doc: PromptDocument,
    library: ph.PhraseLibrary,
) -> GateResult:
    """G14 约束强度未降级 —— 无 E/F 级、无强→弱降级。

    只体检约束段落（参考素材 / 关键一致性 / 禁止项）。
    创作区是用户的表达，在里面说"参考一下"是正常说话，不该误报。
    """
    issues = ph.scan_texts(library, doc.text_lines())
    if not issues:
        return GateResult("G14", "约束强度", OK, "无降级、无 E/F 级")
    return GateResult(
        "G14", "约束强度", BLOCK,
        f"{len(issues)} 处问题：{'；'.join(i.title for i in issues[:3])}",
        suggestion="；".join(i.suggestion for i in issues[:3] if i.suggestion),
        evidence=tuple(i.title for i in issues[:6]),
    )


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def run_gates(
    ir: Mapping[str, Any],
    doc: PromptDocument,
    library: ph.PhraseLibrary | None = None,
    *,
    baseline: Mapping[str, Any] | None = None,
) -> GateReport:
    """跑完 14 项。顺序与 PRD 表格一致，便于对照。"""
    lib = library or ph.load_library()
    report = GateReport()
    report.add(g1_placeholder_residue(doc, ir))
    report.add(g2_quota(ir, lib))
    report.add(g3_exclusive_modes(doc, ir))
    report.add(g4_contradiction(doc, ir))
    report.add(g5_dialogue_quotes(doc, ir))
    report.add(g6_lipsync_binding(ir))
    report.add(g7_char_budget(doc, ir, lib))
    report.add(g8_density(ir))
    report.add(g9_split_strategy(doc, ir))
    report.add(g10_locked_diff(ir, baseline))
    report.add(g11_hook_preserved(ir, baseline))
    report.add(g12_ref_roles(doc, ir))
    report.add(g13_mandatory_dimensions(doc, ir, lib))
    report.add(g14_constraint_strength(doc, lib))
    return report


# --------------------------------------------------------------------------
# LOCK diff
# --------------------------------------------------------------------------
def diff_locked(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[tuple[str, Any, Any]]:
    """逐字段比对锁定位。返回 ``(路径, 原值, 新值)``。

    只比对契约里标注为 LOCK 的位置 —— 用户能改的地方变了不算违规。
    """
    before = dict(fw.flatten(baseline))
    after = dict(fw.flatten(current))
    diffs: list[tuple[str, Any, Any]] = []
    for path, old_value in before.items():
        spec = fw.spec_for(path)
        if spec is None or spec.category.value != "LOCK":
            continue
        new_value = after.get(path, _MISSING)
        if new_value is _MISSING:
            diffs.append((path, old_value, None))
        elif _normalize(new_value) != _normalize(old_value):
            diffs.append((path, old_value, new_value))
    return diffs


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return "" if text == UNKNOWN else text
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _find_residues(text: str) -> list[str]:
    import re

    return [f"【{m}】" for m in re.findall(r"【([^】]{1,40})】", text)]


def _role_cn(key: str) -> str:
    return {"image": "参考图", "video": "参考视频", "audio": "参考音频"}.get(key, key)
