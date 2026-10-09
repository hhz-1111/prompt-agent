"""用户意见改写（PRD F8.4「按建议重写」）—— 纯函数部分。

## 它和自检-重写回路的区别（这是本模块存在的理由）

自检回路（:mod:`engine.core.review`）问的是"**我写错了吗**"，
所以它只能碰描述性文字，站位/景别/台词一律不许改 ——
那是量测结果与用户原文，模型用判断去推翻它们等于把"量测"降级成"猜"。

改写问的是另一件事："**你要什么**"。授权来自用户本人，
所以可改范围**宽得多**，而且是**结构级**的：

    改场景 / 增加人物 / 优化站位 / 改台词 / 增加或减少镜头
    调景别 / 调机位与拍摄方式 / 调人物运动轨迹

用户 2026-10-08 的原话：「1:1复刻是基础，根据用户想法再更改提示词去创新生成视频是进阶，
改可以，但是不能脱离实际，你也不能乱改给用户看」——
所以规则是**用户意志优先，但每一步都要能解释、能看见、能撤销**：

1. **能落地就落地**：路径按契约校验（``REVISABLE_ROOTS``），值可以是对象
   （往 ``locked.B2_person`` / ``creative.C2_beats`` 里插一条）。
2. **落不了就直说**：``refusals`` 把"哪条做不到、为什么"原样带回给用户 ——
   返回空数组假装"改好了"是这个功能最容易犯的错。
3. **真的红线仍然守**：素材绑定（红线 1）与交付形态 AI 不许动；
   ``forbidden`` 也不在这里改（D0/D7 强制保留，走固定提示词开关）。
4. **不在文本上动手**：改动全部落回 IR，再重新组装 + 重跑闸门 ——
   IR 是唯一事实来源，这条纪律不因为"是用户要的"而破例。
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence

from . import framework as fw
from .contracts import (
    Extension,
    ReviewNote,
    RevisionChange,
    RevisionOp,
    RevisionOutcome,
    RevisionRefusal,
)
from .types import PositionCategory

# --------------------------------------------------------------------------
# 可改范围
# --------------------------------------------------------------------------
REVISABLE_ROOTS: tuple[str, ...] = (
    "meta.genre",
    "meta.shot_mode",
    "meta.content_kind",
    "meta.aspect_ratio",
    "meta.target_model",
    "meta.replacement_level",
    "meta.total_duration_sec",
    "locked.B1_scene",
    "locked.B2_person",
    "locked.B2b_ensemble",
    "locked.B3_blocking",
    "locked.B4_camera",
    "locked.B5_quality",
    "locked.B6_audio",
    "creative.C1_logline",
    "creative.C2_beats",
    "creative.C3_extensions",
    "references",
)
"""用户意见可以落到的子树。

**为什么连"锁定位"都在里面**：LOCK 的意思是"改了就不再是这条爆款的复刻"，
不是"用户不许改"。用户说「场景换成咖啡馆」时，他明确接受"不再 1:1"，
所以场景锚定、站位、音色这些都能改 —— 但**每一次都要如实告诉他
这次改动让它偏离了原片**（见 :func:`deviation_warning`）。
"""

_OFF_LIMITS: tuple[tuple[str, str], ...] = (
    ("references[].asset_id",
     "素材绑定不能由 AI 编造（红线 1）：素材要你自己上传或绑定"),
    ("delivery",
     "交付形态（导出方式/参数声明）不是提示词内容，改它请用交付设置"),
    ("forbidden",
     "禁止项走固定提示词清单（D0/D7 强制保留），不在改写范围里"),
)
"""AI 一律不许碰的东西。用户意见指向这里时**如实拒绝**并说明原因。"""


def _off_limit_reason(path: str) -> str:
    normalized = fw.normalize_path(path)
    for blocked, why in _OFF_LIMITS:
        if normalized == blocked or normalized.startswith(blocked + ".") \
                or normalized.startswith(blocked + "["):
            return why
    return ""


def is_revisable(path: str) -> bool:
    """路径是否落在可改范围内（``[]`` 模板与具体下标都接受）。"""
    normalized = fw.normalize_path(str(path or ""))
    if not normalized or _off_limit_reason(normalized):
        return False
    return any(
        normalized == root or normalized.startswith(root + ".")
        or normalized.startswith(root + "[")
        for root in REVISABLE_ROOTS
    )


OBJECT_LISTS: tuple[str, ...] = (
    "locked.B2_person",
    "locked.B6_audio",
    "creative.C2_beats",
    "creative.C3_extensions",
    "references",
)
"""可以 ``add`` / ``remove`` 的对象数组（"增加人物""增加镜头"就落在这里）。"""

LIST_LABELS: Mapping[str, str] = {
    "locked.B2_person": "人物",
    "locked.B6_audio": "音色",
    "creative.C2_beats": "镜头",
    "creative.C3_extensions": "扩展段",
    "references": "素材位",
}
"""增删操作在界面上显示的名字（契约里数组本身没有中文标签，
直接用路径给用户看像报错信息）。"""

# 封闭枚举：值必须逐字来自契约白名单。
# 判据是"它驱动渲染或平台控件"——写错就是交付物坏了。
CLOSED_ENUMS: frozenset[str] = frozenset({
    "meta.aspect_ratio",
    "meta.target_model",
    "meta.replacement_level",
    "meta.shot_mode",
    "locked.B4_camera.shot_mode",
    "locked.B5_quality.preset",
    "locked.B2_person[].lock_strength",
    "locked.B2_person[].skin",
    "locked.B2_person[].blink",
    "locked.B6_audio[].volume_mode",
    "creative.C2_beats[].beat_role",
    "references[].type",
})
"""**封闭枚举**：必须逐字取自契约白名单。

其余枚举（场景类型/光线/机位/运镜/景别/机位角度/genre/content_kind）故意**开放**：
管线自己（``to_ir`` + 视觉模型）写进去的也是自由中文（「室内客厅」「手持自拍」），
契约里的 choices 是**示例而不是硬化约束**。用户说「改成航拍」「换成咖啡馆」时，
把它们挡在白名单外等于把用户的正当要求挡掉 —— 那正是这个功能要解决的问题。
"""

PLUMBING_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "locked.B2_person[]": {"lock_strength": "standard", "skin": "brief", "blink": "5s"},
    "locked.B6_audio[]": {"volume_mode": "normal"},
    "references[]": {"type": "multi_view", "seedance_role": "reference_image"},
}
"""新增条目时由**代码**补齐的机械字段（不是内容）。

它们要么是产品的无条件固定项（眨眼节拍 / 皮肤质感 / 音色锁定的档位），
要么是句柄解析的管道字段。让模型去猜这些等于让它编内容，所以代码补。
**注意：绝不补内容字段**（场景、动作、台词一律不补 —— 那是"不猜"这条红线）。
"""

_REF_TYPE_ROLE: Mapping[str, str] = {
    "multi_view": "reference_image",
    "scene": "reference_image",
    "blocking": "reference_image",
    "pose_sheet": "reference_image",
    "video": "reference_video",
    "audio": "reference_audio",
}

_REF_PREP_PROMPT: Mapping[str, str] = {
    "multi_view": "人物多视图：正面 / 左右 3/4 侧 / 侧面，五官清晰无遮挡",
    "scene": "场景图：先生成人物位置关系图，再去掉人物生成场景图效果更稳定",
    "blocking": "位置关系图：标清人物左右与前后关系，不带文字",
    "pose_sheet": "体式图 / 动作拆解图：关键帧逐格拆解，肢体轨迹清晰",
    "video": "参考视频：只保留要复刻的运镜与动作，时长与目标一致",
    "audio": "音色克隆样本：任意完整一段，2~15 秒，环境安静无混响",
}

# --------------------------------------------------------------------------
# 给模型看的"当前状态"
# --------------------------------------------------------------------------
def revision_view(ir: Mapping[str, Any]) -> dict[str, Any]:
    """把可改字段摊成 ``{路径: 现值}`` —— 模型据此提出**精确路径**的修改。

    为什么不直接把整份 IR 丢给它：IR 里有大量系统字段（素材绑定、句柄、
    provenance），模型看到就会想去改；只给它"能改的这些"最省心。
    """
    view: dict[str, Any] = {}
    for path, value in fw.flatten(ir):
        if not is_revisable(path):
            continue
        view[path] = value
    return view


def pacing_facts(ir: Mapping[str, Any], *, chars_per_sec: float = 5.0) -> dict[str, Any]:
    """节奏事实：让模型有依据地改镜头数（而不是拍脑袋）。

    用户 2026-10-08 的原话：「假设用户想增减台词量，原有的镜头数量不匹配，
    那 AI 就要通过自己的理解去更改镜头量」。要判断"匹不匹配"，
    它得先知道每句台词需要多少秒 —— 中文常规语速按 5 字/秒估。
    """
    beats = ir.get("creative", {}).get("C2_beats", []) or []
    rows: list[dict[str, Any]] = []
    for beat in beats:
        if not isinstance(beat, Mapping):
            continue
        dialogue = beat.get("dialogue") or {}
        line = str(dialogue.get("line") or "").strip()
        try:
            duration = float(beat.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        row: dict[str, Any] = {
            "镜头": beat.get("shot_id"),
            "时长秒": round(duration, 2),
        }
        if line:
            needed = round(len(line) / max(chars_per_sec, 0.1), 1)
            row["台词字数"] = len(line)
            row["念完需要秒"] = needed
            if duration and needed > duration + 0.3:
                row["提示"] = "这一镜的时长不够念完这句台词"
        rows.append(row)
    try:
        total = float(ir.get("meta", {}).get("total_duration_sec") or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    return {
        "总时长秒": round(total, 2),
        "镜头数": len(beats),
        "逐镜头": rows,
        "语速假设": f"{chars_per_sec:g} 字/秒（中文常规）",
    }


def item_templates() -> str:
    """把"新增一条人物/一条镜头"要长什么样，用人话+JSON 骨架写出来。

    没有这份骨架，模型会自己造字段名（实测这类任务上很常见），
    然后被契约拒掉、白跑一次。
    """
    return (
        "新增**人物**（往 locked.B2_person 里加一条）：\n"
        '  {"name": "外观特征+性别（如 深蓝运动上衣男子）", "display_name": "同上或更短",'
        ' "position": "画面位置（如 画面左侧）", "multi_view_ref": "<name>多视图"}\n'
        "  说明：multi_view_ref 只是**素材位名字**，代码会自动登记对应素材位"
        "（asset_id 留空，绝不假装已绑好），你不用管句柄编号。\n"
        "新增**镜头**（往 creative.C2_beats 里加一条）：\n"
        '  {"shot_id": 递增整数, "framing": "景别", "camera_angle": "机位角度",'
        ' "camera": "运镜", "location": "场景", "blocking": "人物在做什么/在哪",'
        ' "expression": "表情", "emotion": "情绪",'
        ' "dialogue": {"speaker": "说话人", "line": "台词", "tone": "语气"}}\n'
        "新增**音色**（往 locked.B6_audio 里加一条）：\n"
        '  {"speaker": "人物名", "audio_ref": "<speaker>音频", "volume_mode": "normal"}\n'
        "减少镜头/人物用 remove（path 带下标，如 creative.C2_beats[2]）。\n"
    )


def closed_enum_reference() -> str:
    """把封闭枚举的可选值渲染给模型（避免它写出渲染层认不出的值）。"""
    lines: list[str] = []
    for path in sorted(CLOSED_ENUMS):
        spec = fw.FIELDS.get(path)
        choices = list(getattr(spec, "choices", ()) or ()) if spec else []
        if choices:
            lines.append(f"- {path}：{' / '.join(choices)}")
    return "\n".join(lines)


SOFT_ENUMS: tuple[str, ...] = (
    "locked.B1_scene.type",
    "locked.B1_scene.lighting",
    "locked.B4_camera.rig",
    "creative.C2_beats[].framing",
    "creative.C2_beats[].camera_angle",
    "creative.C2_beats[].camera",
)
"""**推荐词表**（不强制）：告诉模型常规写法，但用户要新说法时**不拦**。

为什么不强制：这些字段在真实管线里本来就由模型用自由中文填
（``to_ir`` 会写入「室内客厅」「手持自拍」这类词），
而且用户很容易说出词表外但完全合理的要求（「航拍」「一镜到底的长镜头」）。
硬拦等于把用户的正当要求挡掉；不告诉他词表，他又容易写出
「超广角航拍」这种落在景别槽位上的怪值。所以给推荐、不给枷锁。
"""


def soft_enum_reference() -> str:
    lines: list[str] = []
    for path in SOFT_ENUMS:
        spec = fw.FIELDS.get(path)
        choices = list(getattr(spec, "choices", ()) or ()) if spec else []
        if choices:
            lines.append(f"- {path}：常用 {' / '.join(choices)}（也可以按用户的说法写别的）")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 解析模型输出
# --------------------------------------------------------------------------
def parse_outcome(raw: Mapping[str, Any]) -> RevisionOutcome:
    """宽容解析模型返回的 JSON。

    一律 ``or []`` 兜底：结构化输出里字段可能是 ``null``
    （实测踩过：``{"shots": null}`` 让推导式 TypeError，整段结果丢掉）。
    也兼容 ``edits`` 这个旧键名（与自检同一套 schema 时不会崩）。
    """
    raw_ops = raw.get("ops")
    if raw_ops is None:
        raw_ops = raw.get("edits")
    ops: list[RevisionOp] = []
    for item in raw_ops or []:
        if not isinstance(item, Mapping):
            continue
        op = str(item.get("op") or item.get("action") or "set").strip().lower()
        if op not in ("set", "clear", "add", "remove"):
            op = "set"
        ops.append(RevisionOp(
            op=op,
            path=str(item.get("path") or "").strip(),
            value=item.get("value"),
            reason=str(item.get("reason") or "").strip(),
        ))

    extensions: list[Extension] = []
    for item in raw.get("extensions") or []:
        if not isinstance(item, Mapping):
            continue
        lines = item.get("lines")
        if isinstance(lines, str):
            lines = [lines]
        extensions.append(Extension(
            title=str(item.get("title") or "").strip(),
            lines=tuple(str(line).strip() for line in (lines or []) if str(line).strip()),
            evidence=str(item.get("evidence") or "").strip(),
        ))

    refusals: list[RevisionRefusal] = []
    for item in raw.get("refusals") or raw.get("unresolved") or []:
        if isinstance(item, Mapping):
            refusals.append(RevisionRefusal(
                instruction=str(item.get("instruction") or item.get("text") or "").strip(),
                reason=str(item.get("reason") or "").strip(),
            ))
        elif isinstance(item, str) and item.strip():
            refusals.append(RevisionRefusal(instruction=item.strip(), reason=""))

    return RevisionOutcome.of(
        ops=ops, extensions=extensions, refusals=refusals,
        note=str(raw.get("note") or raw.get("summary") or "").strip(),
    )


# --------------------------------------------------------------------------
# 校验并应用
# --------------------------------------------------------------------------
def _reject(path: str, op: RevisionOp, detail: str) -> ReviewNote:
    return ReviewNote(path=path, action=op.op, reason=op.reason,
                      applied=False, detail=detail)


def _coerce_scalar(spec: Any, value: Any) -> tuple[bool, Any, str]:
    """按字段类型把模型给的值收敛一下。返回 ``(成功, 值, 原因)``。"""
    if spec is not None and spec.kind == fw.KIND_NUMBER:
        try:
            return True, float(value), ""
        except (TypeError, ValueError):
            return False, None, f"「{value}」不是数字"
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return True, items, ""
    if isinstance(value, Mapping):
        return False, None, "这里是单值字段，不接受对象"
    return True, str(value if value is not None else "").strip(), ""


def _check_closed_enum(template: str, value: str) -> str:
    """封闭枚举的取值校验。返回空串表示通过。"""
    if template not in CLOSED_ENUMS:
        return ""
    spec = fw.FIELDS.get(template)
    choices = [str(item) for item in (getattr(spec, "choices", ()) or ())] if spec else []
    if choices and value not in choices:
        return f"「{value}」不在可选值里（只能填：{' / '.join(choices)}）"
    return ""


def _validate_item(item_template: str, value: Any, *, depth: int = 0) -> list[str]:
    """递归校验新增条目的键与值。返回错误列表（空 = 通过）。"""
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return [f"{item_template} 需要一个对象"]
    if depth > 3:
        return errors
    for key, item in value.items():
        child = f"{item_template}.{key}"
        spec = fw.spec_for(child)
        if spec is None:
            errors.append(f"「{key}」不是契约里的字段（{item_template}）")
            continue
        if isinstance(item, Mapping):
            errors.extend(_validate_item(child, item, depth=depth + 1))
            continue
        if isinstance(item, (list, tuple)):
            if any(isinstance(entry, Mapping) for entry in item):
                for entry in item:
                    if isinstance(entry, Mapping):
                        errors.extend(_validate_item(child + "[]", entry, depth=depth + 1))
            continue
        if spec.category == PositionCategory.AUTO and child.endswith("asset_id"):
            errors.append("素材绑定不能由 AI 编造（红线 1）")
            continue
        ok, coerced, why = _coerce_scalar(spec, item)
        if not ok:
            errors.append(f"{child}：{why}")
            continue
        if isinstance(coerced, str):
            bad = _check_closed_enum(fw.normalize_path(child), coerced)
            if bad:
                errors.append(f"{child}：{bad}")
    return errors


def _parse_list_path(path: str) -> tuple[str, int | None]:
    """``locked.B2_person`` / ``locked.B2_person[1]`` → (列表路径, 插入下标或 None)。"""
    match = re.match(r"^(.*)\[(\d+)\]$", path.strip())
    if match:
        return match.group(1), int(match.group(2))
    return path.strip(), None


def _next_shot_id(beats: Sequence[Any]) -> int:
    used: list[int] = []
    for beat in beats:
        if isinstance(beat, Mapping):
            try:
                used.append(int(beat.get("shot_id") or 0))
            except (TypeError, ValueError):
                continue
    return (max(used) + 1) if used else 1


SPLIT_STRATEGY_BY_SHOT_MODE: Mapping[str, tuple[str, ...]] = {
    "one_take": ("S1", "S3", "S4"),
    "multi_shot": ("S2", "S3"),
    "same_angle_segments": ("S4", "S3"),
}
"""镜头模式 → 允许的拆条策略（与 ``gates.g9_split_strategy`` 的矩阵同一份事实）。

第一个是该模式下最常用的那个：用户说「加一个镜头」时，策略就该从 S1 一镜到底
变成 S2 逐镜头 —— 否则 G9 直接阻断「S1 与镜头模式 multi_shot 不符」。
"""


def sync_split_strategy(ir: dict[str, Any]) -> str:
    """镜头模式变了就把拆条策略调成相符的那个。返回变化描述（没变则空串）。

    为什么由**代码**做：这是机械联动。用户说"再加一个镜头"时，
    他不可能同时想着"顺便把拆条策略从 S1 改成 S2"——
    实测第一次 live 改写就是这么翻车的：场景、人物、镜头全改对了，
    却因为策略没跟着动，交付判定从 ALL_GREEN 掉成 BLOCKED。
    """
    shot_mode = str(ir.get("meta", {}).get("shot_mode") or "one_take")
    options = SPLIT_STRATEGY_BY_SHOT_MODE.get(shot_mode)
    if not options:
        return ""
    delivery = ir.setdefault("delivery", {})
    if not isinstance(delivery, dict):
        return ""
    current = str(delivery.get("strategy") or "")
    if current in options:
        return ""
    delivery["strategy"] = options[0]
    return f"{current or '（空）'} → {options[0]}（镜头模式 {shot_mode} 对应）"


def normalize_durations(ir: dict[str, Any]) -> bool:
    """镜头数变了就把总时长**平摊**到每个镜头（机械字段，不是内容）。

    为什么必须做：镜头数变了之后，``duration_sec`` 还停在旧值，
    下一轮改写拿到的「节奏事实」就是错的（会告诉模型"这一镜有 15 秒"），
    它据此判断"台词念得完"，于是改出一个真的念不完的方案。
    实测第一版就是这样：新增镜头的时长是从旧镜头**抄**来的，
    两个镜头加起来 30 秒 > 总时长 15 秒。

    只动 AUTO 字段，且只在镜头数变化后调用 —— 纯粹的文本改写不碰它。
    """
    beats = ir.get("creative", {}).get("C2_beats", []) or []
    if not beats:
        return False
    try:
        total = float(ir.get("meta", {}).get("total_duration_sec") or 0.0)
    except (TypeError, ValueError):
        return False
    if total <= 0:
        return False
    each = round(total / len(beats), 3)
    changed = False
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        try:
            current = float(beat.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            current = 0.0
        if abs(current - each) > 0.01:
            beat["duration_sec"] = each
            changed = True
    return changed


def _fill_plumbing(
    item_template: str,
    item: dict[str, Any],
    *,
    existing: Sequence[Any],
    index: int,
) -> list[str]:
    """补齐机械字段（不是内容）。返回补了哪些，供记账。

    ``item_template`` 一律是**带** ``[]`` 的模板（如 ``locked.B2_person[]``）——
    第一版这里踩过：有的分支按带 ``[]`` 的键查默认值、有的按不带 ``[]`` 的比较，
    结果新建镜头时 ``shot_id`` 没补上，交付物里出现「镜头0」。
    """
    filled: list[str] = []
    for key, default in (PLUMBING_DEFAULTS.get(item_template) or {}).items():
        if key in item and str(item.get(key) or "").strip():
            continue
        if fw.spec_for(f"{item_template}.{key}") is None:
            continue
        item[key] = default
        filled.append(key)

    if item_template == "creative.C2_beats[]":
        if not str(item.get("shot_id") or "").strip():
            item["shot_id"] = _next_shot_id(existing)
            filled.append("shot_id")
        item.setdefault("dialogue", {"speaker": "", "line": "", "tone": ""})
    if item_template == "locked.B2_person[]":
        name = str(item.get("name") or "").strip()
        if not str(item.get("display_name") or "").strip() and name:
            item["display_name"] = name
            filled.append("display_name")
        # 素材位名字：人物句渲染成「人物2形象参考[图3]」，句柄靠 multi_view_ref
        # 去 references 里查。漏了它就是正文里冒出一个 [人物2多视图] 的假句柄
        # （G1 直接阻断）。这是**管道**不是内容，所以代码补。
        if not str(item.get("multi_view_ref") or "").strip() and name:
            item["multi_view_ref"] = f"{name}多视图"
            filled.append("multi_view_ref")
    if item_template == "references[]":
        slots = [str(s).strip() for s in (item.get("slots") or []) if str(s).strip()]
        if slots:
            item.setdefault("placeholder", f"【{slots[0]}】")
        item["asset_id"] = None          # 新素材位一律未绑定，绝不假装绑好
        ref_type = str(item.get("type") or "multi_view").strip() or "multi_view"
        role = _REF_TYPE_ROLE.get(ref_type, "reference_image")
        item["seedance_role"] = role
        if not str(item.get("prep_prompt") or "").strip():
            item["prep_prompt"] = _REF_PREP_PROMPT.get(ref_type, "")
        if not str(item.get("prep_step") or "").strip():
            steps = [int(r.get("prep_step") or 0) for r in existing
                     if isinstance(r, Mapping) and str(r.get("prep_step") or "").isdigit()]
            item["prep_step"] = (max(steps) + 1) if steps else 1
        filled.append("asset_id(留空待你绑定)")
    return filled


def apply_ops(
    ir: Mapping[str, Any],
    ops: Sequence[RevisionOp],
) -> tuple[dict[str, Any], list[RevisionChange], list[ReviewNote]]:
    """逐条校验并应用用户意见。返回 ``(新的 IR, 改动明细, 记账)``。

    **原 IR 不被改动**。拒绝的情况一律记账而不是静默丢弃 ——
    "听不懂用户的话"本身就是要反馈给用户的信息。
    """
    updated = copy.deepcopy(dict(ir))
    changes: list[RevisionChange] = []
    notes: list[ReviewNote] = []
    beats_touched = False
    shot_mode_touched = False

    for op in ops:
        path = str(op.path or "").strip()
        if not path:
            notes.append(_reject("", op, "路径为空"))
            continue
        if not is_revisable(path):
            reason = _off_limit_reason(path)
            notes.append(_reject(path, op, reason or "不在可改范围里"))
            continue
        if op.op in ("add", "remove"):
            list_path, index = _parse_list_path(path)
            template = fw.normalize_path(list_path)
            if template not in OBJECT_LISTS:
                notes.append(_reject(path, op, "这里不是可增删的列表"))
                continue
            current = fw.get_in(updated, list_path)
            if not isinstance(current, list):
                notes.append(_reject(path, op, "路径不存在"))
                continue
            if template == "creative.C2_beats":
                beats_touched = True
            if op.op == "remove":
                if index is None or index < 0 or index >= len(current):
                    notes.append(_reject(path, op, "下标越界"))
                    continue
                if template == "creative.C2_beats" and len(current) <= 1:
                    notes.append(_reject(path, op, "至少要留一个镜头，否则提示词没有画面内容"))
                    continue
                before = current[index]
                current.pop(index)
                changes.append(RevisionChange(
                    path=f"{list_path}[{index}]",
                    label=LIST_LABELS.get(template) or fw.label_of(template) or "条目",
                    before=_brief(before), after="（已删除）", reason=op.reason,
                ))
                notes.append(ReviewNote(path=f"{list_path}[{index}]", action="remove",
                                        reason=op.reason, applied=True,
                                        detail=f"删除「{_brief(before)}」"))
                continue
            # add
            errors = _validate_item(template + "[]", op.value)
            if errors:
                notes.append(_reject(path, op, "；".join(errors[:3])))
                continue
            item = dict(op.value or {})
            filled = _fill_plumbing(template + "[]", item, existing=current,
                                    index=index if index is not None else len(current))
            at = len(current) if index is None else max(0, min(index, len(current)))
            current.insert(at, item)
            summary = _brief(item)
            detail = f"新增「{summary}」"
            if filled:
                detail += f"（代码补齐机械字段：{'、'.join(filled)}）"
            changes.append(RevisionChange(
                path=f"{list_path}[{at}]",
                label=LIST_LABELS.get(template) or fw.label_of(template) or "条目",
                before="（新增）", after=summary, reason=op.reason,
            ))
            notes.append(ReviewNote(path=f"{list_path}[{at}]", action="add",
                                    reason=op.reason, applied=True, detail=detail))
            continue

        # set / clear
        spec = fw.spec_for(path)
        if spec is None:
            notes.append(_reject(path, op, "契约里没有这个字段"))
            continue
        before = fw.get_in(updated, path)
        if before is None:
            notes.append(_reject(path, op, "路径不存在（下标越界？）"))
            continue
        if op.op == "clear":
            value: Any = "" if spec.kind != fw.KIND_NUMBER else 0
        else:
            ok, value, why = _coerce_scalar(spec, op.value)
            if not ok:
                notes.append(_reject(path, op, why))
                continue
            if isinstance(value, str):
                bad = _check_closed_enum(fw.normalize_path(path), value)
                if bad:
                    notes.append(_reject(path, op, bad))
                    continue
        if value == before:
            notes.append(_reject(path, op, "与现状相同，无需修改"))
            continue
        try:
            fw.set_in(updated, path, value)
        except Exception as exc:  # noqa: BLE001 - 契约拒写
            notes.append(_reject(path, op, f"写入被契约拒绝：{exc}"))
            continue
        changes.append(RevisionChange(
            path=path, label=fw.label_of(path) or path,
            before=_brief(before), after=_brief(value), reason=op.reason,
        ))
        notes.append(ReviewNote(path=path, action="set", reason=op.reason, applied=True,
                                detail=f"{_brief(before)} → {_brief(value)}"))
        if fw.normalize_path(path) in ("meta.shot_mode", "locked.B4_camera.shot_mode"):
            shot_mode_touched = True

    if shot_mode_touched:
        detail = sync_split_strategy(updated)
        if detail:
            notes.append(ReviewNote(
                path="delivery.strategy", action="repair", reason="镜头模式变了",
                applied=True,
                detail=f"拆条策略已自动跟着改：{detail}（否则 G9 会阻断）",
            ))
    if beats_touched and normalize_durations(updated):
        notes.append(ReviewNote(
            path="creative.C2_beats[].duration_sec", action="normalize", reason="镜头数变了",
            applied=True,
            detail="镜头数变了，已把总时长平摊到各镜头（机械字段，供下一轮判断台词念不念得完）",
        ))
    return updated, changes, notes


def _brief(value: Any, *, limit: int = 42) -> str:
    """把任意值压成一行给人看的摘要。"""
    if isinstance(value, Mapping):
        if "shot_id" in value:      # 镜头条目：shot_id + 景别 + 在做什么最好认
            parts = [f"镜头{value.get('shot_id')}"]
            for key in ("framing", "blocking", "location", "expression"):
                text = str(value.get(key) or "").strip()
                if text:
                    parts.append(text)
                    break
            return _clip("：".join(parts[:2]) if len(parts) > 1 else parts[0])
        for key in ("name", "title", "text", "line"):
            if str(value.get(key) or "").strip():
                return _clip(str(value[key]))
        return _clip("；".join(f"{k}:{v}" for k, v in list(value.items())[:2]))
    if isinstance(value, (list, tuple)):
        if not value:
            return "（空）"
        return _clip("、".join(str(item) for item in value[:2]))
    text = str(value or "").strip()
    return _clip(text) if text else "（空）"


def _clip(text: str, limit: int = 42) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# 素材位自动补齐（让"增加人物"真的能用）
# --------------------------------------------------------------------------
def ensure_reference_slots(ir: dict[str, Any]) -> list[str]:
    """给"引用了但没登记"的素材位补一条 references（asset_id 留空）。

    **为什么必须由代码补**：人物句渲染成
    ``人物2：人物2形象参考[图2]``，句柄是靠 ``references`` 里的 slot 名解析出来的
    （``HandleResolver``）。模型加了一条人物但忘了加素材位时，
    渲染会留下 ``【…多视图】`` 残留 → G1 阻断 → 交付物不能用。
    这是**管道问题不是内容问题**，所以代码兜底，且如实记账"这里要你上传素材"。

    与红线 1（不编造参考素材）不冲突：补的是**空素材位**（``asset_id=None``），
    正是"我需要一张图，请你上传"，不是"我已经有这张图"。
    """
    references = ir.setdefault("references", [])
    if not isinstance(references, list):
        ir["references"] = references = []

    def known(name: str) -> bool:
        for ref in references:
            if not isinstance(ref, Mapping):
                continue
            keys = {str(ref.get("placeholder") or "").strip("【】")}
            keys.update(str(item) for item in (ref.get("slots") or []))
            if name in keys:
                return True
        return False

    added: list[str] = []

    def register(name: str, ref_type: str) -> None:
        name = name.strip()
        if not name or known(name):
            return
        item: dict[str, Any] = {
            "slots": [name], "type": ref_type, "placeholder": f"【{name}】",
            "asset_id": None,
        }
        _fill_plumbing("references[]", item, existing=references, index=len(references))
        references.append(item)
        added.append(name)

    for person in ir.get("locked", {}).get("B2_person", []) or []:
        if isinstance(person, Mapping):
            register(str(person.get("multi_view_ref") or ""), "multi_view")
    for ensemble in (ir.get("locked", {}).get("B2b_ensemble"),):
        if isinstance(ensemble, Mapping):
            register(str(ensemble.get("multi_view_ref") or ""), "multi_view")
    for audio in ir.get("locked", {}).get("B6_audio", []) or []:
        if isinstance(audio, Mapping):
            register(str(audio.get("audio_ref") or ""), "audio")
    return added


def ensure_audio_entries(ir: dict[str, Any]) -> list[str]:
    """有台词的人物没登记音色时，补一条 ``B6_audio``（素材位留空待绑定）。

    实测踩过：模型按"增加人物 + 给这个人物一句台词"改完之后，正文里出现了
    新说话人，但 ``locked.B6_audio`` 里没有他 —— **G6 直接阻断**
    「说话人缺音色绑定」。这不是用户做错了，而是**登记动作没人做**，
    属于管道，所以代码补。

    只对**已经存在的人物名**补：泛称（「说话人」）与群体名不补 ——
    否则等于替用户认了一个画面里不存在的人物（红线 5 不臆造人物信息）。
    """
    locked = ir.get("locked", {})
    if not isinstance(locked, dict):
        return []
    people: set[str] = set()
    for person in locked.get("B2_person", []) or []:
        if not isinstance(person, Mapping):
            continue
        for key in ("name", "display_name"):
            text = str(person.get(key) or "").strip()
            if text:
                people.add(text)
    if not people:
        return []

    audio = locked.get("B6_audio")
    if not isinstance(audio, list):
        locked["B6_audio"] = audio = []
    bound = {
        str(item.get("speaker") or "").strip()
        for item in audio if isinstance(item, Mapping)
    }
    added: list[str] = []
    for beat in ir.get("creative", {}).get("C2_beats", []) or []:
        if not isinstance(beat, Mapping):
            continue
        dialogue = beat.get("dialogue") or {}
        if not isinstance(dialogue, Mapping):
            continue
        speaker = str(dialogue.get("speaker") or "").strip()
        line = str(dialogue.get("line") or "").strip()
        if not speaker or not line or speaker in bound or speaker not in people:
            continue
        item: dict[str, Any] = {
            "speaker": speaker, "audio_ref": f"{speaker}音频", "volume_mode": "normal",
        }
        _fill_plumbing("locked.B6_audio[]", item, existing=audio, index=len(audio))
        audio.append(item)
        bound.add(speaker)
        added.append(speaker)
    return added


# --------------------------------------------------------------------------
# 给模型的提示词（纯文本，放在 core 便于单测断言）
# --------------------------------------------------------------------------
def system_prompt() -> str:
    """改写助手的系统提示词。

    与自检回路（``CRITIQUE_SYSTEM``）的区别写在最前面：自检问"我写错了吗"，
    这里问"**你要什么**"。所以纪律也从"宁少勿多"变成
    "**用户意志优先，但每一步都要能解释、能看见、不能脱离实际**"。
    """
    return (
        "你是**视频生成提示词的改写助手**。你拿到一份已经组装好的提示词（用于"
        "Seedance 这类视频生成模型，通常是 9:16 竖屏短视频），以及用户对它的**修改意见**。"
        "\n你的任务：把用户的想法**落到具体字段**上，让他拿到他想要的那版提示词。"
        "\n用户的意见优先于原片的复刻忠实度：他明确要改的地方就改 —— "
        "换场景、加人物、改站位、改台词、增减镜头、调景别、调机位与拍摄方式、"
        "调人物的运动轨迹，都在可改范围内。但**不能脱离实际**："
        "改动后的提示词必须是一份**能真的生成出来、内部自洽**的提示词。"
        "\n"
        "\n【最重要的一条：联动更新】改一处常常要连带改另一处，漏了就自相矛盾："
        "\n- **加人物** → 同步把站位/人数那句改对（`locked.B3_blocking.positions` 里"
        "写着「画面中固定 N 位人物出镜」这类话）；素材位与音色**不用你管**，代码会自动登记。"
        "\n- **加镜头** → `shot_id` 递增；镜头数变了要回头看总时长与每镜时长是否还成立"
        "（`meta.total_duration_sec` 可以改）。"
        "\n- **改台词（或增减台词量）** → 必须让「念完这句需要多少秒」≤「这镜的时长」"
        "（中文按 5 字/秒估，节奏事实里已经替你算好了）。"
        "台词变长/变短导致镜头装不下时，**要么拆到多个镜头、要么增减镜头、要么改镜头时长** ——"
        "这正是用户说的「AI 要通过自己的理解去更改镜头量」。"
        "\n- **调景别/机位/运镜/运动轨迹** → 写进对应镜头的"
        "`framing` / `camera_angle` / `camera` / `blocking` 字段。"
        "\n- **改场景** → 改 `locked.B1_scene.anchor`（一句话说清在哪里、什么环境），"
        "必要时也改 `locked.B1_scene.type`。"
        "\n- **改了镜头模式**（`meta.shot_mode` / `locked.B4_camera.shot_mode`）→"
        "拆条策略由代码自动跟着改，你**不用管**；但要同步景别序列 "
        "`locked.B4_camera.framing_set`（逐镜头写景别）。"
        "\n"
        "\n【群像规则（用户 2026-10-07 反复强调，违反必被拒）】"
        "\n- **需要「人物参考」的人物不超过 2 个**。其余人（学员/观众/背景里的人）一律写成"
        "**群像**：`locked.B2b_ensemble` 里给「称呼 + 年龄 + 穿着」一句话就够，"
        "他们由生成模型自行生成、不需要参考图。"
        "\n- **删人物时只删人物清单，不要删画面事实**：镜头动作里其他人（"
        "「其余数人在她身后做动作」这类）的描述**必须原样保留** —— "
        "那是生成时唯一说明「背景里有人在做什么」的地方。"
        "删了它，背景人群就变成一片空白（用户 2026-10-07 实测踩过，代码也会拒掉这一条）。"
        "\n- 群像是**看**还是**跟着做**：学员/队员用 `mode: follow` + `follows: 人物1`；"
        "观众/听众/背景里的人用 `mode: watch`，**不要替他们编动作**"
        "（他们的动作写在镜头节拍里）。"
        "\n"
        "\n【措辞要过得了交付闸门】这两条实测踩过，改完直接阻断："
        "\n- **不许写泛化表述**：「保持一致」「自然流畅」「注意细节」这类没有对象或量化的句子"
        "会被判成无效约束（G14 阻断）。要写成带对象/量化的说法，例如"
        "「两镜之间场景与光线100%相同」。"
        "\n- **运镜要逐字出现**「一镜到底 / 机位全程不变 / 镜头无跳切」之一（G13 逐字校验）——"
        "多镜头片子写「机位全程不变」，一镜到底写「一镜到底」「镜头无跳切」。"
        "\n"
        "\n【输出格式（只输出 JSON，不要解释文字）】"
        '\n{"ops": [{"op": "set|clear|add|remove", "path": "字段路径",'
        ' "value": 值, "reason": "为什么改这一处"}],'
        ' "extensions": [{"title": "段落名", "lines": ["一行"], "evidence": "依据"}],'
        ' "refusals": [{"instruction": "用户哪句话", "reason": "为什么做不到"}],'
        ' "note": "一句话总结你做了什么（给用户看，用大白话）"}'
        "\n"
        "\n【纪律（违反会被代码直接拒绝并记账）】"
        "\n- `path` 必须逐字取自下面列出的**可改字段**，下标写具体数字"
        "（如 `creative.C2_beats[1].framing`）。不在清单里的路径一律被拒。"
        "\n- **每一条 op 都要写 reason**，说清「用户哪句话让你改的」。"
        "\n- **做不到的必须写进 `refusals`**，并说清为什么（例如"
        "「加一个画外音旁白」需要新素材位和音色，而这不是提示词能凭空生成的）。"
        "**绝不能返回空的 ops 假装改好了** —— 那比直说做不到更糟。"
        "\n- **不要顺手改用户没提的地方**。只改「为了让他的改动成立」所必需的字段。"
        "\n- **不要编造画面里没有的事实**（性别、年龄、职业、道具、品牌）。"
        "用户明确要求的除外 —— 那时按他说的写。"
        "\n- 素材绑定（`asset_id`）、交付形态、禁止项清单**不在可改范围**，"
        "提了也会被拒（前两个是红线：素材必须由用户上传）。"
        "\n- 眨眼节拍、皮肤质感这两条是产品的无条件固定项，除非用户明确要求去掉，否则别动。"
        "\n- 改完如果提示词变短/变长很多，那是正常的；**不要为了显得改动多而制造改动**。"
    )


def user_prompt(
    *,
    instruction: str,
    ir: Mapping[str, Any],
    prompt_text: str = "",
    evidence: Mapping[str, Any] | None = None,
    max_view_chars: int = 7000,
) -> str:
    """把"用户意见 + 现状 + 事实"拼成一条 user 消息。

    ``prompt_text`` 是**交付物原文**（用户实际看到的那一版）——
    他说话时会引用"第三句""那个镜头"，把那一段给他看，指代才对得上。
    """
    import json

    view = json.dumps(revision_view(ir), ensure_ascii=False, indent=1)
    if len(view) > max_view_chars:
        view = view[:max_view_chars] + "\n…（后续字段略）"

    parts = [f"【用户的修改意见】\n{str(instruction or '').strip()}"]
    if prompt_text:
        body = prompt_text[-6000:] if len(prompt_text) > 6000 else prompt_text
        parts.append(f"【当前提示词（用户看到的正文）】\n{body}")
    parts.append(f"【当前可改字段与现值】\n{view}")
    parts.append("【节奏事实（判断镜头数够不够用）】\n"
                 + json.dumps(pacing_facts(ir), ensure_ascii=False, indent=1))
    if evidence:
        parts.append("【逐镜头画面事实（改的时候不要与它冲突）】\n"
                     + json.dumps(dict(evidence), ensure_ascii=False, indent=1))
    parts.append("【可以增删的列表】\n" + "\n".join(f"- {path}" for path in OBJECT_LISTS))
    parts.append("【新增条目的骨架】\n" + item_templates())
    parts.append("【这些字段只能填下列值之一】\n" + closed_enum_reference())
    parts.append("【这些字段的常用词表（只是推荐，用户可以要新说法）】\n"
                 + soft_enum_reference())
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# 授权偏离的提醒
# --------------------------------------------------------------------------
LOCK_PREFIXES: tuple[str, ...] = (
    "locked.B1_scene.anchor",
    "locked.B2_person[].wardrobe",
    "locked.B2b_ensemble",
    "locked.B3_blocking",
    "locked.B4_camera",
    "locked.B6_audio",
    "creative.C2_beats[].dialogue",
)
"""改动这些位置意味着"**不再是这条爆款的 1:1 复刻**"（框架 §9.2 永不可裁区）。

注意最后一个：台词。红线 3 防的是 Agent **自己编**台词；
这里是**用户点名要改**，所以放行，但必须在下一次交付自检里说清楚
"台词已按你的要求改写"——否则用户会以为台词还是原片那句。
"""


def deviation_warning(changed_paths: Sequence[str]) -> str:
    """改动里有没有"偏离复刻"的部分？返回一句给人看的话（没有则空串）。"""
    hit: list[str] = []
    for path in changed_paths:
        normalized = fw.normalize_path(str(path))
        for prefix in LOCK_PREFIXES:
            if normalized == prefix or normalized.startswith(prefix + ".") \
                    or normalized.startswith(prefix + "["):
                label = fw.label_of(prefix) or prefix
                if label not in hit:
                    hit.append(label)
                break
    if not hit:
        return ""
    return (
        f"这次按你的要求改了{'、'.join(hit)} —— 这几处属于原片的锁定信息，"
        "改完之后就不再是原片的 1:1 复刻，而是基于它的再创作。闸门也会据此提示。"
    )
