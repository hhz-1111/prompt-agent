"""致命维度多源裁决：**检测框与模型打架时，去取一次证，让票数说话**。

## 为什么必须有它

人数与站位是"致命/高"影响级维度，而这两路证据**各自都会错**（都有实测反例）：

- **检测框会多检**：演播厅单人示范片的背景大屏上印着 4 张人物照片，
  YOLO 把屏里的人也算成真人（检出 3）；
- **检测框会漏检**：双人片子里后排的人被前排挡住，4 个镜头里 3 个只检出 1 个；
- **模型也会看错**：它会把屏上图像当成真人（同一个例子里它一开始也数成 3）。

原来的处置是"以模型/以 CV 为准 + 标黄"，结果只是把分歧记下来，
**没有第三方来判断谁对**。这里补的就是第三方：针对那个镜头抽帧问一句
**只回答数字**的问题，把答案当第三票。

## 裁决规则（不猜、不静默改写）

| 情形 | 处置 |
|---|---|
| 取证与模型一致 | 记「已裁决：以模型为准（取证一致）」 |
| 取证与检测框一致 | 记「已裁决：以检测框为准」**并提醒用户**——正文的人数句取的是模型值，可能偏 |
| 三方各说各话 | 记「三源不一致」，如实说明 |
| 取证失败/答不出数字 | 保留原判据，记一条说明 |

**为什么不直接改写模型看到的人数**：那等于用一句话的自由文本去覆盖一次结构化感知，
而且取证同样可能看错。裁决要留下的是一条**可核对的三源记录**，不是偷偷改数据。
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

from engine.core.contracts import Conflict, ReviewNote, ShotBoundary

COUNT_FIELD = "creative.C2_beats[].people_count"

COUNT_QUESTION = (
    "画面里有几个**真人**？**不要把屏幕上、照片里、海报上、镜子倒影里的人算进去**。"
    "只回答一个阿拉伯数字，不要别的字。"
)

_DIGIT = re.compile(r"(\d+)")


def critical_conflicts(conflicts: Sequence[Conflict]) -> list[Conflict]:
    """值得花一次取证的冲突：人数（致命级），且两路结论确实不同。"""
    return [item for item in conflicts if item.field == COUNT_FIELD]


def parse_count(answer: str) -> int | None:
    """从回答里取人数。取不到就返回 None —— **不猜**。"""
    match = _DIGIT.search(str(answer or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - 正则保证了是数字
        return None


def _cv_count(conflict: Conflict) -> int | None:
    return parse_count(conflict.cv_value)


def _model_count(conflict: Conflict) -> int | None:
    return parse_count(conflict.model_value)


def arbitrate_conflicts(
    conflicts: Sequence[Conflict],
    shots: Sequence[ShotBoundary],
    probe: Any,
    video_path: str,
    *,
    budget: int,
    notes: list[ReviewNote],
    degradations: list[str],
) -> list[Conflict]:
    """对人数冲突做一次取证裁决，返回**更新过 resolution** 的冲突列表。"""
    targets = critical_conflicts(conflicts)
    if not targets or budget <= 0 or probe is None or not video_path:
        return list(conflicts)

    from engine.core.contracts import ProbeRequest

    by_index = {shot.index: shot for shot in shots}
    verdicts: dict[int, str] = {}
    spent = 0
    for conflict in targets:
        if spent >= budget:
            break
        shot = by_index.get(conflict.shot_index)
        if shot is None:
            continue
        spent += 1
        request = ProbeRequest(
            question=COUNT_QUESTION,
            time_start=shot.start_ms / 1000,
            time_end=shot.end_ms / 1000,
            region="full",
            reason=f"镜头{conflict.shot_index} 人数两路说法不一致，取第三票",
        )
        try:
            finding = probe.look(video_path, request)
        except Exception as exc:  # noqa: BLE001 - 取证失败不该拖垮反推
            degradations.append(
                f"镜头{conflict.shot_index} 人数仲裁取证失败：{str(exc)[:80]}"
            )
            continue
        answer = str(finding.answer or "").strip()
        count = parse_count(answer)
        notes.append(ReviewNote(
            path=f"creative.C2_beats[{conflict.shot_index - 1}].people_count",
            action="arbitrate", reason=request.reason,
            applied=count is not None,
            detail=f"检测框 {conflict.cv_value}｜模型 {conflict.model_value}"
                   f"｜取证：{answer[:40] or '答不出'}",
        ))
        if count is None:
            continue
        cv_count = _cv_count(conflict)
        model_count = _model_count(conflict)
        if count == model_count:
            verdicts[conflict.shot_index] = (
                f"已裁决：取证确认 {count} 位真人，与模型一致 → 以模型为准"
                f"（检测框 {cv_count} 位多算了屏上/照片里的人）"
            )
        elif count == cv_count:
            verdicts[conflict.shot_index] = (
                f"已裁决：取证确认 {count} 位真人，与检测框一致 → **以检测框为准**；"
                f"注意正文人数句取的是模型值（{model_count} 位），可能偏"
            )
            degradations.append(
                f"镜头{conflict.shot_index} 人数三方比对：取证与检测框都认为 {count} 位，"
                f"模型认为 {model_count} 位 —— 请核对正文的人数描述"
            )
        else:
            verdicts[conflict.shot_index] = (
                f"三源不一致：检测框 {cv_count}｜模型 {model_count}｜取证 {count}，"
                f"无法裁决，请人工核对"
            )
            degradations.append(
                f"镜头{conflict.shot_index} 人数三源各说各话（{cv_count}/{model_count}/{count}），"
                f"人数字段已按模型值保留，请人工核对"
            )

    if not verdicts:
        return list(conflicts)
    return [
        replace(item, resolution=verdicts[item.shot_index])
        if item.shot_index in verdicts and item.field == COUNT_FIELD else item
        for item in conflicts
    ]
