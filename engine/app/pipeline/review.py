"""自检-重写回路：让**写提示词的那个模型回头看自己写的东西**。

## 为什么这是"把管线变成 agent"的最小一步

原来的链路是严格单向的：切分 → 量测 → 视觉理解 → 塞进 IR → 渲染 → 闸门出报告。
模型写完**再也没有看过自己写的东西**，而闸门是确定性的浅匹配
（关键词、字数、正则）—— 语义问题它接不住。实测抓到的三件事全是语义问题：

    「人物1 出现了但从来没定义过」
    「同一场景在四个镜头里抄了四遍」
    「静音视频里编出了一整段台词」（这一条后来靠闸门+上游堵住了）

所以这里补一个**回路**：把渲染好的交付物连同画面证据再喂回模型，
按 rubric 找问题 → 提**字段级**修改 → 代码按契约校验并应用 → 重新组装 → 再过闸门。
最多跑 ``rounds`` 轮，问题清空或改不动就停。

## 三条不能破的纪律

1. **只能改白名单字段**。模型提的是 ``{path, action, value, reason}``，
   路径必须落在 :data:`engine.core.review.ALLOWED_EDIT_PREFIXES` 里，
   否则拒绝并记账 —— 永不可裁的内容（站位、音色、台词原文、D0/D7）一个字都不能动。
2. **只能变好，不能变坏**。每轮应用后重新跑 14 项闸门；判定**变差就整体回滚**，
   回到上一轮的 IR。宁可少改，不可改坏。
3. **改了什么必须记下来**（``ReviewNote`` 进 ``report.reviews``），
   并在结果页的「待确认」栏露出来 —— 否则用户不知道交付物被"重写"过。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from engine.core import assembler, gates, review as review_core
from engine.core.contracts import (
    CritiqueEdit,
    CritiqueOutcome,
    InferenceReport,
    ProbeFinding,
    ProbeRequest,
    ReviewNote,
)
from engine.core.ports import PromptCritic, VisionProbe
from engine.core.phrases import PhraseLibrary
from typing import Any, Callable, Mapping, Sequence

VERDICT_RANK = {"ALL_GREEN": 0, "PASS_WITH_WARNINGS": 1, "BLOCKED": 2}


def _as_outcome(value: Any) -> CritiqueOutcome:
    """兼容两种返回：``CritiqueOutcome`` 或裸的 edits 列表。

    替身/老实现只给一个列表也应当能用 —— 回路不该因为"少写一个包装类"而崩。
    """
    if isinstance(value, CritiqueOutcome):
        return value
    if value is None:
        return CritiqueOutcome()
    if isinstance(value, (list, tuple)):
        return CritiqueOutcome.of([e for e in value if isinstance(e, CritiqueEdit)])
    return CritiqueOutcome()


def _run_probes(
    probe: VisionProbe,
    video_path: str,
    requests: Sequence[ProbeRequest],
    *,
    budget: int,
    notes: list[ReviewNote],
    deadline: float | None = None,
) -> tuple[list[ProbeFinding], int]:
    """执行取证请求。返回 ``(结果, 实际用掉的次数)``。

    取证**失败不算反推失败**：它只是"这一次没查到"，记一条降级说明继续。
    单次取证的异常也被单条吞掉，不能让一条坏请求带走其它几条。

    ``deadline``（``time.monotonic()`` 基准）到点了就**不再开始新的取证** ——
    实测踩过：取证走的是全局默认超时（180 秒 × 3 次），能把自检的实际耗时
    顶到 563.7 秒、冲破 480 秒预算。预算必须是"整段"的，不能只管自检那一次调用。
    """
    import time

    findings: list[ProbeFinding] = []
    used = 0
    for request in requests[: max(0, budget)]:
        if deadline is not None and time.monotonic() > deadline:
            notes.append(ReviewNote(
                path=f"probe@{request.time_start:g}-{request.time_end:g}s",
                action="probe", reason=request.reason, applied=False,
                detail="自检的取证时间预算已用完，这一条没有去看（可以在设置里调大预算）",
            ))
            continue
        used += 1
        try:
            finding = probe.look(video_path, request)
        except Exception as exc:  # noqa: BLE001
            notes.append(ReviewNote(
                path=f"probe@{request.time_start:g}-{request.time_end:g}s",
                action="probe", reason=request.reason, applied=False,
                detail=f"取证失败：{exc}",
            ))
            continue
        notes.append(ReviewNote(
            path=f"probe@{request.time_start:g}-{request.time_end:g}s",
            action="probe", reason=request.reason, applied=bool(finding.answer),
            detail=f"看{request.region}：{request.question} → {(finding.answer or '看不清')[:60]}",
        ))
        if finding.answer:
            findings.append(finding)
    return findings, used


def _verdict_rank(report: Any) -> int:
    return VERDICT_RANK.get(str(getattr(report, "verdict", "ALL_GREEN")), 1)


def run_review(
    ir: dict[str, Any],
    report: InferenceReport,
    critic: PromptCritic,
    library: PhraseLibrary,
    *,
    rounds: int = 1,
    probe: VisionProbe | None = None,
    video_path: str = "",
    probe_budget: int = 3,
    budget_sec: float = 0.0,
    on_progress: Callable[[str, float], None] | None = None,
) -> tuple[dict[str, Any], tuple[ReviewNote, ...], tuple[str, ...]]:
    """跑自检-重写回路。返回 ``(新的 IR, 记账, 降级说明)``。

    每轮是"**自检 →（没把握就）取证 → 再自检 → 改**"：
    模型可以在第一轮里要求"再看一眼某个时间点"，拿到那几帧之后再决定怎么改。
    这是整个链路里唯一"看不懂就去查"的地方。

    **时间预算**（用户 2026-10-07 报"自检与重写卡很久"）：``budget_sec > 0`` 时，
    每轮与每次取证前都查一次是否超时 —— 超了就**停下并保留当前版本**，
    如实写一条降级说明。推理模型一次调用可能想很久，没有预算这一阶段会像卡死。

    ``on_progress`` 让内部步骤也上报（第几轮、取证到第几条）——
    以前这一阶段只有一个静态的 0.97，用户分不清它在动还是死了。

    任何异常都不该拖垮整条反推：出错就原样返回 + 记一条降级说明。
    """
    import time

    notes: list[ReviewNote] = []
    degradations: list[str] = []
    if rounds <= 0 or critic is None:
        return ir, tuple(notes), tuple(degradations)
    deadline = (time.monotonic() + budget_sec) if budget_sec > 0 else None

    def expired() -> bool:
        return deadline is not None and time.monotonic() > deadline

    def timing_out() -> None:
        degradations.append(
            f"自检-重写超时（预算 {budget_sec:g} 秒），已保留未复核版本；"
            f"可在设置里调 REVIEW_BUDGET_SEC"
        )

    doc = assembler.assemble(ir, library)
    try:
        current_verdict = _verdict_rank(gates.run_gates(ir, doc, library))
    except Exception:  # noqa: BLE001 - 闸门坏了不该阻断反推
        current_verdict = VERDICT_RANK["PASS_WITH_WARNINGS"]

    open_issues: tuple[str, ...] = ()
    spent_probes = 0
    for round_index in range(1, rounds + 1):
        if expired():
            timing_out()
            break
        if on_progress:
            on_progress(f"自检与重写（第{round_index}轮）", 0.97)
        try:
            outcome = _as_outcome(critic.critique(
                prompt=doc.full_text(),
                evidence=review_core.critique_evidence(ir, report, open_issues),
            ))
        except Exception as exc:  # noqa: BLE001 - 模型/网络失败就停，不拖垮反推
            degradations.append(f"自检回路第 {round_index} 轮失败，已保留上一版：{exc}")
            break

        # ---- 取证：模型自己点名要看哪一段，我们抽帧（可放大）问它 ----
        if (
            outcome.probes
            and probe is not None
            and video_path
            and spent_probes < probe_budget
            and not expired()
        ):
            if on_progress:
                on_progress("自检取证中", 0.98)
            findings, used = _run_probes(
                probe, video_path, outcome.probes,
                budget=probe_budget - spent_probes, notes=notes,
                deadline=deadline,
            )
            spent_probes += used
            if findings and not expired():
                if on_progress:
                    on_progress(f"自检与重写（第{round_index}轮 · 取证后复检）", 0.98)
                try:
                    outcome = _as_outcome(critic.critique(
                        prompt=doc.full_text(),
                        evidence=review_core.critique_evidence(
                            ir, report, open_issues, findings
                        ),
                    ))
                except Exception as exc:  # noqa: BLE001
                    degradations.append(f"取证后重新自检失败，已保留上一版：{exc}")
                    break

        edits = list(outcome.edits)
        extensions = list(outcome.extensions)
        # ---- 结构化提问：真的两难就**问人**（最多 2 条，答案由结果页点选）----
        if outcome.questions:
            accepted, ask_notes = review_core.apply_questions(ir, outcome.questions)
            notes.extend(ask_notes)
            for question in accepted:
                notes.append(ReviewNote(
                    path=question.path, action="ask", reason=question.reason or "两难",
                    applied=False,
                    detail=f"问用户：{question.text}（选项：{'/'.join(question.options)}）",
                ))
            if accepted:
                # 问题挂在 report 上（不是 IR —— 它不是交付物内容）。
                # report 是同一个对象，调用方拿到的就是带问题的报告。
                report.questions = tuple(report.questions) + tuple(accepted)

        if not edits and not extensions:
            notes.append(ReviewNote(
                path="", action="none", reason="自检未发现问题", applied=False,
                detail=f"第 {round_index} 轮：模型没提出任何修改",
            ))
            break

        candidate, round_notes = review_core.apply_edits(ir, edits)
        # 扩展段与字段修改**同一套把关**：都由代码按契约校验，
        # 之后一起过闸门 —— 写坏了一样会整体回滚。
        if extensions:
            candidate, extension_notes = review_core.apply_extensions(candidate, extensions)
            round_notes = list(round_notes) + extension_notes
        applied = [note for note in round_notes if note.applied]
        if not applied:
            # 提了但一条都不合法：记下来（这是审计线索，不是静默丢弃）
            notes.extend(round_notes)
            open_issues = tuple(f"{n.path}：{n.detail or n.reason}" for n in round_notes[:5])
            break

        try:
            new_doc = assembler.assemble(candidate, library)
            new_verdict = _verdict_rank(gates.run_gates(candidate, new_doc, library))
        except Exception as exc:  # noqa: BLE001
            degradations.append(f"自检回路第 {round_index} 轮产出无法组装，已回滚：{exc}")
            break

        if new_verdict > current_verdict:
            # **改坏了 → 整体回滚**。宁可少改，不可改坏。
            notes.extend(
                ReviewNote(path=n.path, action=n.action, reason=n.reason, applied=False,
                           detail=f"{n.detail}；但本轮判定由 {current_verdict} 变差，已整体回滚")
                for n in applied
            )
            degradations.append(
                f"自检回路第 {round_index} 轮把交付判定改差了（闸门等级 {current_verdict}→"
                f"{new_verdict}），已整体回滚到上一版"
            )
            break

        ir, doc, current_verdict = candidate, new_doc, new_verdict
        notes.extend(round_notes)
        open_issues = ()

    return ir, tuple(notes), tuple(degradations)
