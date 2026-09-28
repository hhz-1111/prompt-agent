"""反推结果 → Framework IR。

这个模块把"分析结论"翻译成"契约结构"。它的纪律只有一条，但最要紧：

    **低置信度宁可留空，也不猜。**（PRD 红线 7）

所以每个字段的赋值前都要过 ``_apply``：置信度 <0.50 就不写，
并把该位置登记为"待用户补填"。猜错会导致后续全链路偏移，而用户很难发现。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from engine.core import framework as fw
from engine.core.contracts import InferenceReport, SemanticSummary
from engine.core.types import (
    CONFIDENCE_REVIEW,
    UNKNOWN,
    BlinkBeat,
    ConfidenceBasis,
    Genre,
    Provenance,
    QualityPreset,
    RefType,
    ShotMode,
    Span,
)

# 场景类型 → 场景锚定句（有具体参考对象才算确定约束）
SCENE_ANCHOR_WITH_TYPE = (
    "场景始终为{场景句柄}的{场景类型}环境，保持同一位置和光线氛围，"
    "镜头无跳切，背景不虚化，环境清晰可见，自然色调"
)
SCENE_ANCHOR_NO_TYPE = (
    "场景始终为{场景句柄}，保持同一位置和光线氛围，"
    "镜头无跳切，背景不虚化，环境清晰可见，自然色调"
)

# 内容类型 → 眨眼节拍（框架 §3 B5）
BLINK_BY_GENRE: Mapping[Genre, BlinkBeat] = {
    Genre.DIALOGUE: BlinkBeat.FIVE,
    Genre.TEACHING: BlinkBeat.THREE_TO_FOUR,
    Genre.STUDIO: BlinkBeat.THREE_TO_FOUR,
    Genre.STREET_INTERVIEW: BlinkBeat.FIVE,
    Genre.REPLACEMENT: BlinkBeat.THREE,
}


def build_ir(report: InferenceReport) -> dict:
    """把一次反推组装成 Framework IR 树。"""
    ir = fw.empty_ir(
        genre=(report.semantic.genre.value if report.semantic else Genre.DIALOGUE.value),
        shot_mode=(
            report.semantic.shot_mode.value if report.semantic else ShotMode.ONE_TAKE.value
        ),
        aspect_ratio=report.video.aspect_ratio,
    )
    # 时长为 0 是致命错误，必须如实记录而不是编一个
    ir["meta"]["total_duration_sec"] = round(report.video.duration_sec, 3)
    ir["meta"]["replacement_level"] = "L2"

    _fill_scene(ir, report)
    _fill_persons(ir, report)
    _fill_camera(ir, report)
    _fill_blocking(ir, report)
    _fill_quality(ir, report)
    _fill_beats(ir, report)
    _fill_logline(ir, report)
    _fill_audio(ir, report)
    _fill_forbidden(ir, report)
    _fill_delivery(ir, report)
    # 素材清单最后填：它依赖人物与音色清单已经就位
    ir["references"] = reference_manifest(ir)
    return ir


# ----------------------------------------------------------------------
# 清洗
# ----------------------------------------------------------------------
_PLACEHOLDER_VALUES = frozenset({
    "", "unknown", "Unknown", "UNKNOWN", "未知", "无", "none", "None", "null", "N/A", "n/a",
})


def _text(value: Any) -> str:
    """把模型的"空值表述"统一成空串。

    模型经常把 unknown 写成中文"未知"或"无"。这些**不是内容**，
    必须当成空处理 —— 否则会拼出"的未知环境"这种垃圾句子，
    而且会让闸门以为该维度已填充。
    """
    text = str(value or "").strip()
    return "" if text in _PLACEHOLDER_VALUES else text


def _placeholder(value: Any) -> str:
    """需要落库为 UNKNOWN 的字段。"""
    return _text(value) or UNKNOWN


def _fill_scene(ir: dict, report: InferenceReport) -> None:
    scene_type = _text(report.semantic.scene_type if report.semantic else "")
    if scene_type:
        ir["locked"]["B1_scene"]["type"] = scene_type
        ir["locked"]["B1_scene"]["anchor"] = SCENE_ANCHOR_WITH_TYPE.format(
            场景句柄="【场景参考图】", 场景类型=scene_type
        )
    else:
        # 反推不出场景类型：留空 + 锚定句不带场景名，**不写"未知环境"这种垃圾句子**
        ir["locked"]["B1_scene"]["type"] = UNKNOWN
        ir["locked"]["B1_scene"]["anchor"] = SCENE_ANCHOR_NO_TYPE.format(
            场景句柄="【场景参考图】"
        )
    ir["locked"]["B1_scene"]["cross_shot_consistency"] = True


# 复刻场景里"人物"的上限。
# 超过这个数的多半是模型把群体（学员/观众/顾客）当成了独立个体 ——
# 实测有一条带练视频被抽出 10 个"人物"，于是生成 10 张多视图，
# 直接撞爆 Seedance 2.0 的 9 张图配额（G2 阻断）。
MAX_REVERSE_ENGINEERED_PERSONS = 4

# 群体词：出现这些的"人物"标签应视为群体，不进人物清单（PRD D2 群体防细化）
_GROUP_WORDS = ("观众", "学员", "顾客", "路人", "人群", "大家", "其他人", "群众", "粉丝")


def _fill_blocking(ir: dict, report: InferenceReport) -> None:
    """从跨镜头的站位检测结果归纳出"谁一直在左、谁一直在右"。

    站位是**致命级**维度（错了人物关系错乱），而它恰好是 CV 算得最准的：
    人体框中心 x 是几何量测。所以只要有检测结果就一定要写进 LOCK 位，
    而不是留给用户从零填。

    判据：某侧出现在**多数镜头**里才算稳定站位。只出现一次不算 ——
    那可能是人物走动，不是固定站位。
    """
    per_shot = [s.blocking for s in report.shots if s.blocking]
    if not per_shot:
        ir["locked"]["B3_blocking"]["positions"] = []
        return

    def dominant(side: str) -> tuple[int, float]:
        hits = [b.x_ratio for shot in per_shot for b in shot if b.side == side]
        if not hits:
            return 0, 0.0
        return len(hits), round(sum(hits) / len(hits), 3)

    threshold = max(1, int(len(per_shot) * 0.6))
    parts: list[str] = []
    for side, label in (("left", "左侧"), ("right", "右侧")):
        count, average = dominant(side)
        if count >= threshold:
            parts.append(f"人物始终在画面{label}（中心 x≈{average}）")

    if len(parts) == 2:
        # 多人且左右稳定：相对关系是"复刻像不像"的关键约束
        parts.append("两人位置全程禁止变动")
    elif not parts:
        # 单人成片最常见的情形：**始终居中**。这本身是有效约束
        # （PRD 把站位列为致命级维度），不能因为"不左不右"就丢掉。
        count, average = dominant("center")
        if count and count >= threshold:
            parts.append(f"人物始终位于画面中央（中心 x≈{average}），位置全程禁止变动")

    ir["locked"]["B3_blocking"]["positions"] = parts


def _fill_persons(ir: dict, report: InferenceReport) -> None:
    """人物清单：有几个人、叫什么。

    名字来源有两路，优先级从高到低：

    1. **ASR 说话人**（``说话人：内容`` 或声纹分离）—— 这是"谁在说"的直接证据；
    2. **视觉模型的人物标签**（如"左侧女子"）—— 只能区分形象，不知道名字。

    **群体标签会被过滤掉**：``台下观众``、``8名学员`` 是群体不是个体，
    给群体建多视图既浪费配额又违反 D2 群体防细化。
    两路都拿不到就留空等用户补，绝不编角色名（红线 5）。
    """
    people: list[str] = []
    for segment in report.transcript:
        if segment.speaker and segment.speaker not in people:
            people.append(segment.speaker)
    for understanding in report.understandings:
        for name in understanding.people:
            cleaned = _text(name)
            if not cleaned or cleaned in people:
                continue
            if any(word in cleaned for word in _GROUP_WORDS):
                continue
            people.append(cleaned)

    blink = BLINK_BY_GENRE.get(
        report.semantic.genre if report.semantic else Genre.DIALOGUE,
        BlinkBeat.FIVE,
    )
    if not people:
        ir["locked"]["B2_person"] = []
        return
    ir["locked"]["B2_person"] = [
        {
            "name": name,
            "multi_view_ref": f"{name}多视图",
            "lock_strength": "standard",
            "wardrobe": "",
            "skin": "brief",
            "blink": blink.value,
        }
        for name in people[:MAX_REVERSE_ENGINEERED_PERSONS]
    ]


def person_overflow_note(report: InferenceReport) -> str:
    """人物数超上限时的说明 —— 由调用方放进 degradations，如实告知。"""
    names: list[str] = []
    for understanding in report.understandings:
        for name in understanding.people:
            cleaned = _text(name)
            if cleaned and cleaned not in names:
                names.append(cleaned)
    if len(names) <= MAX_REVERSE_ENGINEERED_PERSONS:
        return ""
    return (
        f"识别到 {len(names)} 个人物标签，已截取前 {MAX_REVERSE_ENGINEERED_PERSONS} 个"
        f"（其余可能是群体或次要人物）：{'、'.join(names[MAX_REVERSE_ENGINEERED_PERSONS:])}。"
        f"请按实际需要增删角色，每多一个角色就多占一张参考图配额。"
    )


def _fill_camera(ir: dict, report: InferenceReport) -> None:
    camera = ir["locked"]["B4_camera"]
    framings = [
        signal.framing.kind
        for signal in report.shots
        if signal.framing is not None and signal.framing.kind != "unknown"
    ]
    camera["framing_set"] = framings

    rig = ""
    for signal in report.shots:
        if signal.motion is not None:
            from .infer import RIG_BY_MOTION

            rig = RIG_BY_MOTION.get(signal.motion.kind, "")
            if rig:
                break
    camera["rig"] = rig or UNKNOWN


def _fill_quality(ir: dict, report: InferenceReport) -> None:
    semantic = report.semantic
    ir["locked"]["B5_quality"]["preset"] = (
        semantic.quality_preset.value if semantic else QualityPreset.P1_PHONE.value
    )
    ir["locked"]["B5_quality"]["skin_text"] = "brief"


def _fill_beats(ir: dict, report: InferenceReport) -> None:
    """逐镜头填节拍。台词来自 ASR，**不做任何代写**。"""
    by_index = {u.shot_index: u for u in report.understandings}
    hook_index = report.semantic.hook_index if report.semantic else 0
    # 全片只有一个人物时，说话人没有歧义 —— 这是推导而非猜测
    sole_speaker = (
        ir["locked"]["B2_person"][0]["name"]
        if len(ir["locked"]["B2_person"]) == 1 else ""
    )

    # 先按时间把台词分配到镜头，避免整段文本被复制到每个重叠镜头
    lines_by_shot = _distribute_transcript(report)

    beats: list[dict] = []
    for signal in report.shots:
        shot = signal.boundary
        understanding = by_index.get(shot.index)
        dialogue = _dialogue_for(lines_by_shot.get(shot.index, ()), sole_speaker)

        beats.append({
            "shot_id": shot.index,
            "framing": (
                _placeholder(signal.framing.kind) if signal.framing else UNKNOWN
            ),
            "camera": _placeholder(understanding.camera if understanding else ""),
            "location": _text(understanding.location if understanding else ""),
            "beat_role": _beat_role(shot.index, hook_index, len(report.shots)),
            "duration_sec": round(shot.duration_sec, 3),
            "blocking": _text(understanding.blocking if understanding else ""),
            "expression": _text(understanding.expression if understanding else ""),
            "emotion": _text(understanding.emotion if understanding else ""),
            "dialogue": dialogue,
        })
    ir["creative"]["C2_beats"] = beats


def _beat_role(index: int, hook_index: int, total: int) -> str:
    """节拍功能（LOCK 位）。只做**结构推断**，不臆造内容。"""
    if hook_index and index == hook_index:
        return "hook"
    if index == 1:
        return "hook"
    if index == total:
        return "cta"
    if index == total - 1:
        return "proof"
    return "demo"


def _distribute_transcript(report: InferenceReport) -> dict[int, list]:
    """把 ASR 片段分配到镜头。

    两种情形分开处理，这是本模块最容易出错的地方：

    - **时间戳来自 ASR 词/句对齐**（``has_timestamps=True``）：按真实时间重叠
      分配。一句话横跨两个镜头是正常的；
    - **区间只是音频总时长推导的**（``has_timestamps=False``）：把整段归给
      **时序重叠最多**的那个镜头，其余镜头留空。

    为什么第二类不做比例切分：按比例切是在**编造对齐关系**，
    而 PRD 红线 7 要求"低置信度宁可留空也不猜"。硬切还会产生
    "你这气色也太好了吧，是 / 不是偷偷做项目啦？" 这种半句话。
    整段归位是诚实的：它只说"这段时间在说话"，切割交给 M2 的对齐环节由用户定。
    """
    precise = [s for s in report.transcript if s.has_timestamps and s.text.strip()]
    if precise:
        out: dict[int, list] = {}
        for signal in report.shots:
            out[signal.boundary.index] = [
                seg for seg in precise if _overlaps(seg.span, signal.boundary.span)
            ]
        return out

    coarse = [s for s in report.transcript if s.text.strip()]
    if not coarse or not report.shots:
        return {}

    out: dict[int, list] = {}
    for segment in coarse:
        # 只有一条镜头时没有歧义，直接归它
        if len(report.shots) == 1:
            target = report.shots[0]
        else:
            target = max(
                report.shots,
                key=lambda x: _overlap_ms(segment.span, x.boundary.span),
            )
        out.setdefault(target.boundary.index, []).append(segment)
    return out


def _overlap_ms(a: Span, b: Span) -> int:
    return max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


def _dialogue_for(lines: Sequence, sole_speaker: str = "") -> dict[str, Any]:
    """从候选片段里选置信度最高的一条落地。**低置信度不填**。

    ASR 多数情况下不返回说话人。此时若全片只有一个人物，
    说话人不存在歧义 —— 这是**推导**（从"只有一个角色"推出"就是他说的"），
    不是猜测，所以可以填。多于一个角色时留空，交给用户对齐。
    """
    candidates = [seg for seg in lines if str(seg.text).strip()]
    if not candidates:
        return _empty_dialogue()
    best = max(candidates, key=lambda s: s.confidence)
    if best.confidence < CONFIDENCE_REVIEW:
        # 低置信度：宁可留空让用户填，也不猜
        return _empty_dialogue()
    return {
        "speaker": best.speaker or sole_speaker,
        "line": best.text.strip(),
        "tone": "",
        "lip_sync": "",
        "pause_at": [],
    }


def _empty_dialogue() -> dict[str, Any]:
    return {"speaker": "", "line": "", "tone": "", "lip_sync": "", "pause_at": []}


def _overlaps(a, b) -> bool:
    if a.end_ms <= a.start_ms:  # 无时间戳的整段
        return False
    return a.start_ms < b.end_ms and b.start_ms < a.end_ms


def _fill_logline(ir: dict, report: InferenceReport) -> None:
    logline = report.semantic.logline if report.semantic else ""
    ir["creative"]["C1_logline"] = logline or ""


def _fill_audio(ir: dict, report: InferenceReport) -> None:
    """音色绑定清单。

    说话人有三路来源，必须**都收进来**：

    1. ASR 返回的说话人（声纹分离或 ``说话人：内容`` 格式）；
    2. 台词里已经落地的说话人 —— 包括"全片只有一个人物"时推导出来的那个。
       漏掉这一路会导致：台词有说话人、音色列表却是空的，
       闸门 G6 阻断、G13 缺音色维度、素材清单也少了音频素材；
    3. 没有任何台词时，为已识别的人物预留音色位（有角色就可能要配音）。
    """
    speakers: list[str] = []
    for segment in report.transcript:
        if segment.speaker and segment.speaker not in speakers:
            speakers.append(segment.speaker)

    for beat in ir.get("creative", {}).get("C2_beats", []):
        dialogue = beat.get("dialogue") or {}
        speaker = str(dialogue.get("speaker") or "").strip()
        if str(dialogue.get("line") or "").strip() and speaker and speaker not in speakers:
            speakers.append(speaker)

    has_dialogue = any(
        str((b.get("dialogue") or {}).get("line") or "").strip()
        for b in ir.get("creative", {}).get("C2_beats", [])
    )
    if not speakers and not has_dialogue:
        for person in ir.get("locked", {}).get("B2_person", []):
            name = str(person.get("name") or "").strip()
            if name:
                speakers.append(name)

    ir["locked"]["B6_audio"] = [
        {"speaker": name, "audio_ref": f"{name}音频", "volume_mode": "normal"}
        for name in speakers
    ]


def _fill_forbidden(ir: dict, report: InferenceReport) -> None:
    """禁止项**不反推，直接套 D0 全库**（框架 §8 关键原则）。"""
    from engine.core import phrases as ph

    library = ph.load_library()
    ir["forbidden"] = library.forbidden_items()


def _fill_delivery(ir: dict, report: InferenceReport) -> None:
    shot_mode = ir["meta"]["shot_mode"]
    strategy = {
        ShotMode.MULTI_SHOT.value: "S2",
        ShotMode.SAME_ANGLE_SEGMENTS.value: "S4",
    }.get(shot_mode, "S1")
    ir["delivery"]["strategy"] = strategy
    ir["delivery"]["mode"] = "phase1_manual"
    ir["delivery"]["include_param_header"] = True
    ir["delivery"]["first_frame_mode"] = "multimodal_reference"


# ----------------------------------------------------------------------
# 素材清单：指名需要哪些素材（Agent 不生成素材，只指名）
# ----------------------------------------------------------------------
def reference_manifest(ir: Mapping[str, Any]) -> list[dict[str, Any]]:
    """列出这条片子需要哪些素材，供上游照单准备（PRD F5.1）。"""
    manifest: list[dict[str, Any]] = []
    step = 0
    for person in ir.get("locked", {}).get("B2_person", []):
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        step += 1
        manifest.append({
            "slots": [f"{name}多视图"],
            "type": RefType.MULTI_VIEW.value,
            "placeholder": f"【{name}多视图】",
            "asset_id": None,
            "seedance_role": "reference_image",
            "prep_step": step,
            "prep_prompt": "人物多视图：正面 / 左右 3/4 侧 / 侧面，五官清晰无遮挡",
        })
    step += 1
    manifest.append({
        "slots": ["场景参考图"],
        "type": RefType.SCENE.value,
        "placeholder": "【场景参考图】",
        "asset_id": None,
        "seedance_role": "reference_image",
        "prep_step": step,
        "prep_prompt": "场景图：先生成人物位置关系图，再去掉人物生成场景图效果更稳定",
    })
    for track in ir.get("locked", {}).get("B6_audio", []):
        speaker = str(track.get("speaker") or "").strip()
        if not speaker:
            continue
        step += 1
        manifest.append({
            "slots": [f"{speaker}音频"],
            "type": RefType.AUDIO.value,
            "placeholder": f"【{speaker}音频】",
            "asset_id": None,
            "seedance_role": "reference_audio",
            "prep_step": step,
            "prep_prompt": "音色克隆样本：任意完整一段，2~15 秒，环境安静无混响",
        })
    return manifest
