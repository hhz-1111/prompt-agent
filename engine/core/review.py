"""自检回路的**核心（纯函数）**：证据打包、白名单校验、补丁应用。

放在 ``engine/core`` 是因为它没有任何副作用与外部依赖 ——
调用模型、跑闸门、回滚这些编排都在 ``engine/app/pipeline/review.py``。

**为什么只允许改这几个字段**：IR 里有一部分是"复刻的底线"，改了就不再是复刻
（框架 §9.2 的永不可裁清单 + PRD 红线）。自检回路的职责是修**语义问题**
（重复、矛盾、没对象的约束），不是重写产品定义。所以：

    允许改：场景锚定句、人物显示名/画面位置、剧情一句话、逐镜头节拍里的**描述性文字**
    一律拒绝：站位 / 音色 / 台词原文 / 禁止项 / 人物名 / 素材绑定 / meta
    **也拒绝景别与运镜**：它们是 CV 的确定性量测，让模型用自己先前的判断去推翻量测，
    等于把"量测"降级成"猜"（实测自检第一版就是这么干的：它把 CV 测出的景别
    按自己的复述改回去了）。这类 CV↔模型分歧已经进 `conflicts` 与待确认栏，由人裁决。
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence

from . import framework as fw
from .contracts import (
    CritiqueEdit,
    Extension,
    InferenceReport,
    ProbeFinding,
    Question,
    ReviewNote,
)

ALLOWED_EDIT_PREFIXES: tuple[str, ...] = (
    "locked.B1_scene.anchor",
    "locked.B2_person[].display_name",
    "locked.B2_person[].position",
    "creative.C1_logline",
    "creative.C2_beats[].location",
    "creative.C2_beats[].blocking",
    "creative.C2_beats[].expression",
    "creative.C2_beats[].emotion",
)
"""自检回路可以动的字段（模板路径）。其余一律拒绝。

**为什么没有 occlusion / eye_line / off_screen**：这三个是**逐镜头**的维度
（谁挡住谁、谁在看谁），而契约里它们是**单值自由文本**。实测让自检填它们时，
它把"镜头1 只有 1 人、镜头2/3 是 2 人"这种**人数矛盾**塞进了 `off_screen` ——
观察是对的，落点错了：那句话与 `positions` 里的「固定 2 位人物出镜」并排出现，
既矛盾又重复（正是用户抱怨过的那类冗余）。这两个问题的正解是
**在确定性管线里按镜头说清人数**（`to_ir._arrangement_sentence` 已改），
不是让模型往自由文本字段里写批注。要做"自检补遮挡/视线"得先给它们
逐镜头的结构，不能靠一个字符串。
"""

FORBIDDEN_EDIT_NOTES = {
    "locked.B3_blocking.positions": "站位是复刻的底线（框架 §9.2 永不可裁）",
    "locked.B6_audio": "音色绑定是复刻的底线（框架 §9.2 永不可裁）",
    "creative.C2_beats[].dialogue": "台词原文由用户提供，Agent 绝不代写（红线 3）",
    "forbidden": "禁止项可增不可减，且通用禁止项必须逐字（D 区）",
    "locked.B2_person[].name": "角色名是身份键，改了会连带素材与音色对不上",
    "references": "素材绑定由用户决定，Agent 只指名需要哪些素材",
}


def is_allowed(path: str) -> bool:
    """路径是否落在白名单里。``[]`` 模板与具体下标都接受。"""
    normalized = fw.normalize_path(str(path or ""))
    return any(
        normalized == prefix or normalized.startswith(prefix + ".")
        or normalized.startswith(prefix)
        for prefix in ALLOWED_EDIT_PREFIXES
    )


def _rejection_reason(path: str) -> str:
    normalized = fw.normalize_path(str(path or ""))
    for blocked, why in FORBIDDEN_EDIT_NOTES.items():
        if normalized.startswith(blocked):
            return why
    return "不在自检回路的可改字段白名单里"


def apply_edits(
    ir: Mapping[str, Any],
    edits: Sequence[CritiqueEdit],
) -> tuple[dict[str, Any], list[ReviewNote]]:
    """逐条校验并应用。返回 ``(新的 IR, 记账)``。**原 IR 不被改动**。

    拒绝的情况一律**记账**而不是静默丢弃 —— 模型越界本身是审计线索
    （说明 rubric 或提示词需要收紧）。
    """
    updated = copy.deepcopy(dict(ir))
    notes: list[ReviewNote] = []
    for edit in edits:
        path = str(edit.path or "").strip()
        if not path:
            notes.append(ReviewNote(path="", action=edit.action, reason=edit.reason,
                                    applied=False, detail="空路径"))
            continue
        if not is_allowed(path):
            notes.append(ReviewNote(path=path, action=edit.action, reason=edit.reason,
                                    applied=False, detail=_rejection_reason(path)))
            continue
        if fw.spec_for(path) is None:
            notes.append(ReviewNote(path=path, action=edit.action, reason=edit.reason,
                                    applied=False, detail="契约里没有这个字段"))
            continue
        before = fw.get_in(updated, path)
        if before is None and edit.action != "set":
            notes.append(ReviewNote(path=path, action=edit.action, reason=edit.reason,
                                    applied=False, detail="路径不存在（下标越界？）"))
            continue
        value = "" if edit.action == "clear" else str(edit.value or "").strip()
        if edit.action not in ("set", "clear"):
            notes.append(ReviewNote(path=path, action=edit.action, reason=edit.reason,
                                    applied=False, detail="只支持 set / clear"))
            continue
        if value == str(before or ""):
            notes.append(ReviewNote(path=path, action=edit.action, reason=edit.reason,
                                    applied=False, detail="值与现状相同，无需修改"))
            continue
        try:
            fw.set_in(updated, path, value)
        except Exception as exc:  # noqa: BLE001 - 契约拒写
            notes.append(ReviewNote(path=path, action=edit.action, reason=edit.reason,
                                    applied=False, detail=f"写入被契约拒绝：{exc}"))
            continue
        notes.append(ReviewNote(
            path=path, action=edit.action, reason=edit.reason, applied=True,
            detail=f"{str(before or '（空）')[:24]} → {value[:24] or '（清空）'}",
        ))
    return updated, notes


def critique_evidence(
    ir: Mapping[str, Any],
    report: InferenceReport,
    open_issues: Sequence[str] = (),
    findings: Sequence[ProbeFinding] = (),
) -> dict[str, Any]:
    """打包给自检模型的**画面证据**。

    只给"能核对的事实"，不给我们的结论 —— 否则模型会顺着我们的说法
    给自己打分（实测里模型很擅长把"我写得对"论证得很圆）。

    ``findings`` 是**取证结果**（模型自己要求"再看一眼"之后拿到的画面回答）。
    它们必须和原始画面事实分开放：原始事实是"一次看全片的印象"，
    取证是"针对某个疑点放大看过之后的结论"——后者更可信，模型应当据此改判。
    """
    shots = []
    for understanding in report.understandings:
        row = {
            "镜头": understanding.shot_index,
            "景别": understanding.framing,
            "机位": understanding.camera_angle,
            "运镜": understanding.camera,
            "场景": understanding.location,
            "画面里是谁": list(understanding.people),
            "动作": understanding.blocking,
            "表情": understanding.expression,
            "情绪": understanding.emotion,
        }
        # **丢掉空字段**：这份证据要占用模型的推理预算，
        # 实测塞满空值时它会把 8192 个输出 token 全花在推理上、正文返回空。
        shots.append({key: value for key, value in row.items() if value not in ("", None, [])})
    evidence: dict[str, Any] = {
        "逐镜头画面事实": shots,
        "人物清单": [
            {"名字": person.get("name"), "位置": person.get("position")}
            for person in ir.get("locked", {}).get("B2_person", [])
        ],
        "有没有台词": any(
            str((beat.get("dialogue") or {}).get("line") or "").strip()
            for beat in ir.get("creative", {}).get("C2_beats", [])
        ),
        "台词": [
            {
                "镜头": beat.get("shot_id"),
                "说话人": (beat.get("dialogue") or {}).get("speaker"),
                "内容": (beat.get("dialogue") or {}).get("line"),
            }
            for beat in ir.get("creative", {}).get("C2_beats", [])
            if str((beat.get("dialogue") or {}).get("line") or "").strip()
        ],
        "画面人数（模型看到）": [
            len(u.people) for u in report.understandings if u.people
        ],
    }
    if open_issues:
        evidence["上一轮没解决或越界的问题"] = list(open_issues)
    if findings:
        evidence["定向取证结果（比上面的一次性印象更可信）"] = [
            {
                "时间点": f"{finding.request.time_start:g}~{finding.request.time_end:g}秒",
                "看的是": finding.request.region,
                "问题": finding.request.question,
                "看到": finding.answer,
            }
            for finding in findings
        ]
    return evidence


def shrink_evidence(evidence: Mapping[str, Any], char_cap: int) -> dict[str, Any]:
    """把证据裁到体积上限之内（用户 2026-10-07 批准的"裁小自检输入"）。

    **先说实测结论**：证据从 193 字到 2500 字，同一个模型耗时是 60 秒 vs 129 秒 ——
    有影响，但**主要耗时在模型思考，不在输入长度**。所以这是个**防爆护栏**
    （长视频十几个镜头时证据会膨胀到几千字），不是提速主力。

    裁剪顺序（先扔信息量最低的）：

    1. 逐镜头事实：优先保留**有冲突/没看清/有画面文字**的镜头，
       其余的按镜头号截断；
    2. 台词、人物清单：整体保留（它们是核对的关键，而且很短）；
    3. 兜底：还超就丢掉"画面人数（模型看到）"这类派生字段。
    """
    import json

    if char_cap <= 0:
        return dict(evidence)

    def size(payload: Mapping[str, Any]) -> int:
        return len(json.dumps(payload, ensure_ascii=False))

    trimmed = dict(evidence)
    if size(trimmed) <= char_cap:
        return trimmed

    shots = list(trimmed.get("逐镜头画面事实") or [])
    if len(shots) > 1:
        # 信息量低的先扔：既没冲突也没疑点、字段又少的镜头排最后
        def value_of(row: Mapping[str, Any]) -> tuple:
            text = json.dumps(row, ensure_ascii=False)
            suspicious = any(
                key in text for key in ("没看清", "未知", "看不清", "冲突", "文字", "不一致", "空")
            )
            return (0 if suspicious else 1, -len(row))

        ordered = sorted(shots, key=value_of)
        while ordered and size({**trimmed, "逐镜头画面事实": ordered}) > char_cap:
            ordered.pop()                      # 从最没信息量的开始扔
            trimmed["逐镜头画面事实"] = ordered
            trimmed["（镜头证据已按体积上限裁剪）"] = True
        if size(trimmed) <= char_cap:
            return trimmed

    for key in ("画面人数（模型看到）", "有没有台词", "上一轮没解决或越界的问题"):
        trimmed.pop(key, None)
        if size(trimmed) <= char_cap:
            break
    return trimmed


def minimal_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """只留逐镜头画面事实与人物清单。

    自检模型的推理预算很紧（实测：输入 2511 字符时它把 8192 输出 token
    全花在推理上、正文返回空）。第一次调用失败就退到这份最小证据 ——
    **这也是实测里唯一跑通的那份**。
    """
    keep = ("逐镜头画面事实", "人物清单")
    return {key: evidence[key] for key in keep if key in evidence}


def allowed_path_reference() -> str:
    """把白名单渲染成人话，交给自检模型的提示词用。"""
    return "\n".join(f"- {path}" for path in ALLOWED_EDIT_PREFIXES)


# --------------------------------------------------------------------------
# 结构化提问：真的两难才问人
# --------------------------------------------------------------------------
MAX_QUESTIONS = 2
MIN_OPTIONS = 2
MAX_OPTIONS = 4


def apply_questions(
    ir: Mapping[str, Any],
    questions: Sequence[Question],
) -> tuple[list[Question], list[ReviewNote]]:
    """校验自检提出的选择题。返回 ``(可用的题, 记账)``。

    **为什么先过滤再给人看**：问题一多用户就不看了（和告警疲劳是同一件事）。
    所以这里卡三条：

    1. **落点必须可写** —— 答案要被写回字段，路径不在白名单里的题根本无法作答；
    2. **选项 2~4 个**；
    3. **最多 2 条**，同一条路径只留第一题。
    """
    accepted: list[Question] = []
    notes: list[ReviewNote] = []
    seen_paths: set[str] = set()
    for index, question in enumerate(questions):
        path = str(question.path or "").strip()
        options = tuple(
            dict.fromkeys(str(o).strip() for o in question.options if str(o).strip())
        )
        text = str(question.text or "").strip()
        if not text or not options:
            continue
        if not is_allowed(path) or fw.spec_for(path) is None:
            notes.append(ReviewNote(
                path=path, action="ask", reason=text, applied=False,
                detail="这道题的落点不在可写字段里，答案没法用上，已丢弃",
            ))
            continue
        if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
            notes.append(ReviewNote(
                path=path, action="ask", reason=text, applied=False,
                detail=f"选项必须是 {MIN_OPTIONS}~{MAX_OPTIONS} 个（收到 {len(options)} 个）",
            ))
            continue
        if path in seen_paths:
            continue
        if len(accepted) >= MAX_QUESTIONS:
            notes.append(ReviewNote(
                path=path, action="ask", reason=text, applied=False,
                detail=f"已经问了 {MAX_QUESTIONS} 个，其余不再打扰用户",
            ))
            continue
        seen_paths.add(path)
        accepted.append(Question(
            id=f"q{index + 1}-{abs(hash(path)) % 10000:04d}",
            path=path, text=text, options=options,
            reason=str(question.reason or "").strip(),
        ))
    return accepted, notes


def answer_question(question: Mapping[str, Any], choice: str) -> tuple[str, str]:
    """把用户的选择落到字段上。返回 ``(路径, 值)``。

    只做**取值校验**；真正的写入仍走 ``framework.can_write`` 那道门禁 ——
    UI 拦、API 也拦，两边同一判据。
    """
    options = [str(o) for o in (question.get("options") or [])]
    value = str(choice or "").strip()
    if value not in options:
        raise ValueError(f"「{value}」不在选项里")
    return str(question.get("path") or ""), value


# --------------------------------------------------------------------------
# 扩展段：框架外信息的落脚点
# --------------------------------------------------------------------------
MAX_EXTENSIONS = 3
MAX_EXTENSION_LINES = 3
MAX_EXTENSION_LINE_CHARS = 80


def apply_extensions(
    ir: Mapping[str, Any],
    extensions: Sequence[Extension],
) -> tuple[dict[str, Any], list[ReviewNote]]:
    """把自检提出的扩展段并进 IR。返回 ``(新的 IR, 记账)``。

    为什么由代码把关（而不是信模型）：扩展段是**新增段落**，
    一旦写坏就是"用户看不懂的交付物"。四条不可协商：

    1. **必须有画面证据** —— 没证据的段落就是编的，一律拒绝；
    2. **段落名不得与既有段落重名** —— 否则分不清哪段是框架的；
    3. 条数与长度有上限 —— 交付物字数预算本来就紧张（G7）；
    4. 最多 3 个扩展段 —— 多了就变成"框架外的东西在主导交付物"。

    ⚠️ **曾经还有一条"不许写引号"，被测试当场否掉**：屏上文案本身就是带引号的
    （实测要放进去的正是「美体普拉提」），这条规则恰好挡住了扩展段的正当用途。
    真正的红线不是引号，而是**不许新增台词** —— 而台词只能落进
    ``C2_beats[].dialogue``，那个字段不在自检白名单里，写不进来。
    带引号的内容由渲染层标明"画面文字、不要生成"（见 assembler）。

    第 5 条（用户 2026-10-08 加）：**"没看清"也要如实写进来**（``unreadable``）。
    以前看不清就整条不提，用户以为画面里只有那些字。
    """
    updated = copy.deepcopy(dict(ir))
    notes: list[ReviewNote] = []
    current = updated.setdefault("creative", {}).setdefault("C3_extensions", [])
    existing_titles = {
        str(item.get("title") or "").strip() for item in current if isinstance(item, Mapping)
    }
    for extension in extensions:
        title = str(extension.title or "").strip()
        lines = [str(line).strip() for line in extension.lines if str(line).strip()]
        evidence = str(extension.evidence or "").strip()
        # 没看清的部分要**说清是什么 + 为什么**（用户 2026-10-08）：
        # 它本身可以单独成为一条扩展段（"这里有字但看不清"），所以不要求 lines 非空。
        unreadable = str(getattr(extension, "unreadable", "") or "").strip()
        path = f"creative.C3_extensions[{len(current)}]"
        if not title or not (lines or unreadable):
            notes.append(ReviewNote(path=path, action="extend", reason="扩展段",
                                    applied=False, detail="段落名或内容为空"))
            continue
        if title in fw.FIXED_SECTIONS:
            notes.append(ReviewNote(path=path, action="extend", reason="扩展段",
                                    applied=False,
                                    detail=f"段落名「{title}」与既有段落重名，已拒绝"))
            continue
        if title in existing_titles:
            continue
        if not evidence:
            notes.append(ReviewNote(path=path, action="extend", reason="扩展段",
                                    applied=False, detail="没有画面证据，已拒绝（无证据即编造）"))
            continue
        if len(current) >= MAX_EXTENSIONS:
            notes.append(ReviewNote(path=path, action="extend", reason="扩展段",
                                    applied=False,
                                    detail=f"扩展段已达上限 {MAX_EXTENSIONS} 个"))
            break
        trimmed = [line[:MAX_EXTENSION_LINE_CHARS] for line in lines[:MAX_EXTENSION_LINES]]
        current.append({
            "title": title,
            "lines": trimmed,
            "evidence": evidence,
            "unreadable": unreadable[:MAX_EXTENSION_LINE_CHARS],
        })
        existing_titles.add(title)
        notes.append(ReviewNote(
            path=f"creative.C3_extensions[{len(current) - 1}]", action="extend",
            reason=f"框架外信息：{evidence[:40]}", applied=True,
            detail=(
                f"新增扩展段「{title}」（{len(trimmed)} 行）"
                + ("，含一处没看清（已提示人工确认）" if unreadable else "")
            ),
        ))
    return updated, notes
