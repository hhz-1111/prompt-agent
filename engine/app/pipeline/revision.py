"""用户意见改写：**编排**（调用模型 → 校验 → 落 IR → 拼出正文 → 记账）。

## 是"一次模型调用"，不是"重跑一遍链路"

用户 2026-10-08 明确要求：「修改提示词不需要再重跑了，直接用 AI 改写提示词
填回去就可以了」。所以默认路径只有**一次**模型调用：

    AI 按你的意见改 → 字段级校验 → 落回 IR → 本地重新拼出正文（毫秒级）→ 填回文本框

重新拼文本这一步**必须留着**：文本是 IR 渲染出来的，不拼就没有新提示词可填。
但它不调模型、不跑闸门，用户感觉不到。

**默认不跑 14 项交付闸门**（``check_gates=False``）：那套评分是"交付前自检"，
不是"改一下就要重新体检一遍"。打开它（``REVISE_CHECK_GATES=1``）行为回到上一版：
判定变差会如实告知，并让 AI 按闸门反馈再修一轮。

## 但底线还在（跟"评分"是两件事）

不管跑不跑闸门，这两种情况一律**整体回滚**并把原因说清：

1. **组装不出来**（IR 写坏了）；
2. **正文里留下没解析的【…】句柄**（红线 6）—— 那种提示词粘进生成平台直接失败。

## 与自检-重写回路的关系

形态相同（模型只提字段级修改 → 代码按契约把关 → 记账），
但**授权与回滚策略不同**：

======================  ==========================  ==============================
                       自检-重写                    用户意见改写
======================  ==========================  ==============================
触发                    系统自己回头看               用户明确要求
授权                    只能改描述性文字             用户意志优先（可增删人物/镜头、
                                                     改站位/台词/景别/运镜）
要不要跑闸门            每轮都跑                     默认不跑（一次调用）
判定变差                **整体回滚**                 打开闸门时：保留 + 如实告知
坏到不能用              回滚                         回滚
======================  ==========================  ==============================
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from engine.core import assembler, gates, review as review_core, revision as revision_core
from engine.core.contracts import (
    ReviewNote,
    RevisionChange,
    RevisionRecord,
    RevisionRefusal,
)
from engine.core.phrases import PhraseLibrary
from engine.core.ports import PromptReviser

HARD_BLOCK_CODES: tuple[str, ...] = ("G1",)
"""出现这些阻断就**整体回滚**：交付物已经不能用了。

只放 G1（句柄解析）：正文里留着 ``[人物2多视图]`` 这种解析不出来的引用，
粘进生成平台就是直接失败 —— 这种版本给用户没有意义。
其余阻断（G2 配额、G3 互斥模式、G4 矛盾…）保留并告知：
它们往往正是"用户要的创新"的代价（例如加了一个人物就撞素材配额），
由用户决定要不要接着调。
"""


@dataclass(slots=True)
class RevisionRun:
    """一次改写的编排结果。"""

    ir: dict[str, Any]
    """改写后的 IR。**没改成时就是原 IR**（原对象不变）。"""

    record: RevisionRecord
    applied: bool = False
    notes: tuple[ReviewNote, ...] = ()
    """连同改动一起要露给用户的记账（含素材位补齐、偏离复刻的提醒）。"""


def evidence_from_report(
    report: Mapping[str, Any] | None,
    ir: Mapping[str, Any],
) -> dict[str, Any]:
    """从**落盘的报告**里取出画面证据（交给改写模型，避免改出与画面冲突的东西）。

    为什么不用 ``review.critique_evidence``：那个函数吃的是
    :class:`InferenceReport` 实例，而改写发生在反推之后很久 ——
    项目里存的是 ``report.to_dict()``。这里按同一套字段名重新拼一份。
    """
    shots: list[dict[str, Any]] = []
    for item in (report or {}).get("understandings") or []:
        if not isinstance(item, Mapping):
            continue
        row = {
            "镜头": item.get("shot_index"),
            "景别": item.get("framing"),
            "机位": item.get("camera_angle"),
            "运镜": item.get("camera"),
            "场景": item.get("location"),
            "画面里是谁": list(item.get("people") or []),
            "动作": item.get("blocking"),
            "表情": item.get("expression"),
            "情绪": item.get("emotion"),
        }
        shots.append({key: value for key, value in row.items() if value not in ("", None, [])})
    people = [
        {"名字": person.get("name"), "位置": person.get("position")}
        for person in (ir.get("locked", {}).get("B2_person") or [])
        if isinstance(person, Mapping)
    ]
    dialogue = [
        {
            "镜头": beat.get("shot_id"),
            "说话人": (beat.get("dialogue") or {}).get("speaker"),
            "内容": (beat.get("dialogue") or {}).get("line"),
        }
        for beat in (ir.get("creative", {}).get("C2_beats") or [])
        if isinstance(beat, Mapping) and str((beat.get("dialogue") or {}).get("line") or "").strip()
    ]
    return {"逐镜头画面事实": shots, "人物清单": people, "台词": dialogue}


def _verdict_of(ir: Mapping[str, Any], library: PhraseLibrary) -> tuple[str, Any, Any]:
    """跑一次组装 + 闸门（只在 ``check_gates=True`` 时用）。"""
    doc = assembler.assemble(ir, library)
    report = gates.run_gates(ir, doc, library)
    return str(getattr(report, "verdict", "")), doc, report


def unresolved_placeholders(doc: Any) -> list[str]:
    """正文行里还剩哪些没解析成句柄的 ``【…】``（红线 6）。

    **只看正文行，不看段落标题** —— 标题本来就长成 ``【镜头节拍】``，
    拿整篇文本去匹配会把每个标题都当成残留（第一版就这么写的，
    测试当场抓到：所有默认路径的改写都被误判成"交付物坏了"而回滚）。

    **这是底线检查，不是评分**：正文里留着 ``【人物2多视图】`` 这种
    没绑定的素材引用，粘进生成平台就是直接失败。所以即便不跑 14 项闸门，
    这一条也照查 —— 用户要的是"别重跑"，不是"给我一个坏版本"。
    """
    found: list[str] = []
    for section in getattr(doc, "sections", ()) or ():
        for line in getattr(section, "lines", ()) or ():
            found.extend(
                match for match in assembler.PLACEHOLDER_RE.findall(line) if match.strip()
            )
    return found


def _blocker_codes(report: Any) -> list[str]:
    return [str(getattr(item, "code", "")) for item in (getattr(report, "blockers", ()) or ())]


def gate_feedback(report: Any, limit: int = 6) -> list[dict[str, str]]:
    """把闸门的结论压成给模型看的"待修清单"。

    闸门的 ``suggestion`` 里常常**已经写好了替换句**
    （实测 G14 会给出一整句「场景始终为[图2]的…两镜之间场景与光线100%相同…」），
    把它一并递回去，模型就只需要照抄着改，不用再猜我们想要什么措辞。
    """
    rows: list[dict[str, str]] = []
    items = list(getattr(report, "blockers", ()) or ()) + list(getattr(report, "warnings", ()) or ())
    for item in items[:limit]:
        rows.append({
            "代码": str(getattr(item, "code", "")),
            "问题": str(getattr(item, "name", "")),
            "说明": str(getattr(item, "detail", "")),
            "建议": str(getattr(item, "suggestion", "")),
        })
    return rows


_BACKGROUND_CUES = (
    "身后", "背后", "背景", "周围", "旁边", "两侧", "两侧", "其余", "其他",
    "数人", "多人", "人群", "后排", "远处", "前景",
)
"""**画面里其他人**的动作线索（用户 2026-10-07 报的坑要用它兜住）。

用户原话：「我直接用改写功能改，叫你去掉那些多余的人物，你倒好，人物是删掉了，
**连着阿姨后景的学员做动作的提示词也给我去掉了**。」
"""


def protect_background_actions(
    base_ir: Mapping[str, Any],
    candidate: dict[str, Any],
) -> list[ReviewNote]:
    """改写删人物时，**不许把画面里其他人的动作一起删掉**。

    后景学员/观众是交给生成模型自行生成的 —— 但**他们做什么动作必须留在节拍里**，
    否则提示词里就再没有任何地方说明"背景有人在练瑜伽"。
    所以这里做一次机械兜底：改后的镜头动作如果**丢掉了原文里"其他人"那一句**，
    就把这一条改动撤掉并如实记账（用户可以再换说法，但不会静默丢掉信息）。
    """
    notes: list[ReviewNote] = []
    base_beats = list((base_ir.get("creative", {}) or {}).get("C2_beats") or [])
    new_beats = list((candidate.get("creative", {}) or {}).get("C2_beats") or [])
    for index, new_beat in enumerate(new_beats):
        if not isinstance(new_beat, Mapping):
            continue
        before = next(
            (
                str(item.get("blocking") or "")
                for item in base_beats
                if isinstance(item, Mapping)
                and item.get("shot_id") == new_beat.get("shot_id")
            ),
            "",
        )
        after = str(new_beat.get("blocking") or "")
        if not before or not after or before == after:
            continue
        if any(cue in before for cue in _BACKGROUND_CUES) and not any(
            cue in after for cue in _BACKGROUND_CUES
        ):
            new_beat["blocking"] = before
            notes.append(ReviewNote(
                path=f"creative.C2_beats[{index}].blocking",
                action="revise",
                reason="这次改写把画面里其他人的动作描述删掉了",
                applied=False,
                detail=(
                    f"已保留原文（其他人由生成模型自行生成，但他们的动作要留在节拍里）："
                    f"{before[:60]}"
                ),
            ))
    return notes


def run_revision(
    ir: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    reviser: PromptReviser,
    library: PhraseLibrary,
    instruction: str,
    *,
    check_gates: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> RevisionRun:
    """跑一次用户意见改写。**任何异常都不该炸掉请求** —— 失败也返回一条可读的记账。

    ``check_gates=False``（默认，用户 2026-10-08 的要求）时只有一次模型调用：
    改完直接拼出正文填回去，不跑那套 14 项交付闸门、也没有修复轮。
    """
    started = time.monotonic()
    instruction = str(instruction or "").strip()
    base_ir = copy.deepcopy(dict(ir))
    record_id = uuid.uuid4().hex[:8]
    model = str(getattr(reviser, "model_id", "") or "")

    def finish(
        *,
        new_ir: Mapping[str, Any],
        applied: bool,
        note: str = "",
        changes: Sequence[RevisionChange] = (),
        rejected: Sequence[ReviewNote] = (),
        refusals: Sequence[RevisionRefusal] = (),
        verdict_before: str = "",
        verdict_after: str = "",
        notes: Sequence[ReviewNote] = (),
    ) -> RevisionRun:
        payload_notes = tuple(notes)
        return RevisionRun(
            ir=dict(new_ir),
            applied=applied,
            notes=payload_notes,
            record=RevisionRecord(
                id=record_id, instruction=instruction, note=note, model=model,
                created_at=time.time(), seconds=time.monotonic() - started,
                changes=tuple(changes), notes=payload_notes,
                rejected=tuple(rejected), refusals=tuple(refusals),
                verdict_before=verdict_before, verdict_after=verdict_after,
                ok=applied,
            ),
        )

    # ---- 改动前的状态（用来给模型看"现在的正文长什么样"）----
    current_doc: Any = None
    before_report: Any = None
    verdict_before = ""
    try:
        if check_gates:
            verdict_before, current_doc, before_report = _verdict_of(base_ir, library)
        else:
            current_doc = assembler.assemble(base_ir, library)   # 本地毫秒级，不是重跑
    except Exception as exc:  # noqa: BLE001 - 组装/闸门坏了不该阻断改写
        verdict_before = f"（判定失败：{exc}）"

    # ---- 调模型 ----
    if on_progress:
        on_progress("正在按你的意见改写")
    try:
        outcome = reviser.revise(
            instruction=instruction,
            ir=base_ir,
            prompt=current_doc.full_text() if current_doc is not None else "",
            evidence=evidence_from_report(report, base_ir),
        )
    except Exception as exc:  # noqa: BLE001 - 模型/网络失败：如实回报，不假装改好
        return finish(
            new_ir=base_ir, applied=False,
            note=f"这次没改成：{type(exc).__name__}: {exc}",
            refusals=(RevisionRefusal(instruction=instruction,
                                      reason=f"模型调用失败：{exc}"),),
            verdict_before=verdict_before, verdict_after=verdict_before,
        )

    # ---- 校验并应用（越界的一律拒绝并记账）----
    candidate, changes, rejected = revision_core.apply_ops(base_ir, outcome.ops)
    notes: list[ReviewNote] = []
    # **机械兜底：改写删人物时不许连"后景其他人的动作"一起删**
    # （用户 2026-10-07 实测踩过：去掉多余人物 → 后景学员做动作那句也没了）
    protected = protect_background_actions(base_ir, candidate)
    if protected:
        rejected = list(rejected) + protected
    if outcome.extensions:
        candidate, extension_notes = review_core.apply_extensions(candidate, outcome.extensions)
        rejected = list(rejected) + list(extension_notes)
        changes = list(changes) + [
            RevisionChange(path=note.path, label="扩展段",
                           before="（新增）", after=note.detail, reason=note.reason)
            for note in extension_notes if note.applied
        ]

    if not changes:
        # 一条都没落地：把"为什么"如实带回（refusals + 被拒的路径）
        return finish(
            new_ir=base_ir, applied=False,
            note=outcome.note or "这次没有需要改动的地方",
            rejected=rejected, refusals=outcome.refusals,
            verdict_before=verdict_before, verdict_after=verdict_before,
        )

    # ---- 机械补齐（管道，不是内容）：新说话人的音色、新素材位 ----
    if revision_core.ensure_audio_entries(candidate):
        notes.append(ReviewNote(
            path="locked.B6_audio", action="repair", reason="有人物说了话但没有音色绑定",
            applied=True,
            detail="已补上音色素材位（asset_id 留空）—— 记得给新人物上传音频样本",
        ))
    added_slots = revision_core.ensure_reference_slots(candidate)
    if added_slots:
        notes.append(ReviewNote(
            path="references", action="repair", reason="有新引用没有对应素材位",
            applied=True,
            detail=f"已登记素材位：{'、'.join(added_slots)}（asset_id 留空，等你上传）",
        ))

    # ---- 拼出新正文（本地毫秒级；默认**不**跑 14 项闸门）----
    if on_progress:
        on_progress("把改动拼回提示词")
    if check_gates:
        try:
            verdict_after, new_doc, new_report = _verdict_of(candidate, library)
        except Exception as exc:  # noqa: BLE001
            return finish(
                new_ir=base_ir, applied=False,
                note=f"按你的意见改完之后组装不起来，已放弃这次改动：{exc}",
                rejected=rejected, refusals=outcome.refusals,
                verdict_before=verdict_before, verdict_after=verdict_before,
            )
    else:
        try:
            new_doc = assembler.assemble(candidate, library)
        except Exception as exc:  # noqa: BLE001
            return finish(
                new_ir=base_ir, applied=False,
                note=f"按你的意见改完之后组装不起来，已放弃这次改动：{exc}",
                rejected=rejected, refusals=outcome.refusals,
                verdict_before=verdict_before, verdict_after=verdict_before,
            )
        verdict_after, new_report = "", None

    # ---- 底线：交付物不能坏（跟"评分"是两件事，所以不跑闸门也要查）----
    text = new_doc.full_text()
    residue = unresolved_placeholders(new_doc)
    if not text.strip() or residue:
        detail = ("正文变成了空内容" if not text.strip()
                  else "正文里留下了没解析的素材引用：" + "、".join(residue[:3]))
        return finish(
            new_ir=base_ir, applied=False,
            note=f"改完之后提示词不能用了，已回滚：{detail}。你可以换个说法再试一次。",
            rejected=list(rejected) + [
                ReviewNote(path="", action="revise", reason=instruction, applied=False,
                           detail=detail),
            ],
            refusals=outcome.refusals,
            verdict_before=verdict_before, verdict_after=verdict_after,
        )

    if check_gates:
        hard = [code for code in _blocker_codes(new_report) if code in HARD_BLOCK_CODES]
        if hard:
            detail = "交付物已损坏（" + "、".join(hard) + "）"
            return finish(
                new_ir=base_ir, applied=False,
                note=f"改完之后提示词不能用了，已回滚：{detail}。你可以换个说法再试一次。",
                rejected=list(rejected) + [
                    ReviewNote(path="", action="revise", reason=instruction, applied=False,
                               detail=detail),
                ],
                refusals=outcome.refusals,
                verdict_before=verdict_before, verdict_after=verdict_after,
            )

    # ---- 闸门反馈修复（只在开了闸门时才有；默认一次调用，不做这一步）----
    # 实测第一次 live 改写：场景/人物/镜头全改对了，却因为"改了镜头模式没同步
    # 拆条策略""写了泛化表述『保持一致』"被 G9/G14 阻断。
    # 这类问题**闸门自己能说清**，所以把它递回给模型修一次 ——
    # 这就是这个项目一直在用的模式："让链路自己发现缺口，然后回去补"
    # （感知补看、台词归属补判都是同一套）。
    if check_gates:
        before_codes = set(_blocker_codes(before_report))
        new_blockers = [code for code in _blocker_codes(new_report) if code not in before_codes]
    else:
        new_blockers = []
    if new_blockers and not any(code in HARD_BLOCK_CODES for code in new_blockers):
        if on_progress:
            on_progress("交付闸门报了新问题，让 AI 再修一轮")
        try:
            repair_instruction = (
                f"{instruction}\n\n"
                "【上一次改完之后，交付闸门报出了这些问题，请**只修这些**】\n"
                + json.dumps(gate_feedback(new_report), ensure_ascii=False, indent=1)
            )
            repair = reviser.revise(
                instruction=repair_instruction,
                ir=candidate,
                prompt=new_doc.full_text(),
                evidence=evidence_from_report(report, candidate),
            )
            repaired_ir, repair_changes, repair_rejected = revision_core.apply_ops(
                candidate, repair.ops)
            if repair_changes:
                revision_core.ensure_audio_entries(repaired_ir)
                revision_core.ensure_reference_slots(repaired_ir)
                verdict2, doc2, report2 = _verdict_of(repaired_ir, library)
                blocked2 = _blocker_codes(report2)
                # 只在**真的变好**且没有引入硬阻断时采纳（宁可少改，不可改坏）
                if (len(blocked2) < len(_blocker_codes(new_report))
                        and not any(code in HARD_BLOCK_CODES for code in blocked2)):
                    candidate, new_doc, new_report = repaired_ir, doc2, report2
                    verdict_after = verdict2
                    changes = list(changes) + list(repair_changes)
                    rejected = list(rejected) + list(repair_rejected)
                    notes.append(ReviewNote(
                        path="", action="repair-round", reason="按闸门反馈再修一轮",
                        applied=True,
                        detail=(f"第一次改完闸门报了 {'、'.join(new_blockers)}，"
                                f"已让 AI 再修一轮：现在阻断 "
                                f"{'、'.join(blocked2) if blocked2 else '没了'}"),
                    ))
        except Exception as exc:  # noqa: BLE001 - 第二轮失败不影响第一轮的改动
            notes.append(ReviewNote(
                path="", action="repair-round", reason="按闸门反馈再修一轮",
                applied=False,
                detail=f"第二轮修复没成（第一轮的改动仍然生效）：{exc}",
            ))

    # ---- 落账：偏离复刻的提醒 + （开了闸门时）判定变化 ----
    warning = revision_core.deviation_warning([change.path for change in changes])
    if warning:
        notes.append(ReviewNote(path="", action="revise-locked", reason="用户要求偏离复刻",
                                applied=True, detail=warning))
    if check_gates and verdict_after != verdict_before:
        blockers = "、".join(_blocker_codes(new_report)) or "无"
        notes.append(ReviewNote(
            path="", action="verdict", reason="按用户要求改动后的交付判定",
            applied=True,
            detail=(f"交付判定：{verdict_before} → {verdict_after}"
                    f"（阻断项：{blockers}）。这是你要的改动的代价，"
                    "要不要接着调由你定。"),
        ))

    return finish(
        new_ir=candidate, applied=True,
        note=outcome.note or f"按你的意见改了 {len(changes)} 处",
        changes=changes, rejected=rejected, refusals=outcome.refusals,
        verdict_before=verdict_before, verdict_after=verdict_after,
        notes=notes,
    )
