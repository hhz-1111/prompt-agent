"""感知覆盖度自检 + 定向补看：**发现"哪个镜头没看清"，然后自己去补**。

## 为什么必须有它（留出集实测暴露的）

拿一条**框架外**的片子（`杨淇换背景前贴`）跑，交付物长这样：

```
镜头1：全景，平视机位拍摄，固定机位拍摄，灰色墙面、灰色地毯的室内房间，…（完整）
镜头2：全景。
镜头3：全景。
镜头4：中全景。
```

模型只回了第 1 个镜头的描述，后 3 个镜头的场景/动作/表情**全空**。
而整条链路**没有任何地方发现这件事** —— 闸门不查这个，自检回路只能改
"已经写出来的东西"，写都没写的地方它看不见。

这就是用户说的"超出框架的视频你可能就推不出来"的**具体形态**：
不是推错，而是**信息在感知阶段就丢了，没人回头补**。

## 判据

一个镜头的描述由三块核心内容构成：**景别 / 场景 / 人物在做什么**。
任何一块空着，就算"感知残缺"，值得再抽一次帧、换一个问法补一次
（补看是**有预算的**，用完就如实留空）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from engine.core.contracts import (
    ReviewNote,
    ShotBoundary,
    ShotUnderstanding,
)

CORE_FIELDS: tuple[str, ...] = ("framing", "location", "blocking")
"""核心描述：**景别 / 场景 / 人物在做什么**。缺一即算残缺。"""

OPTIONAL_FIELDS: tuple[str, ...] = ("camera_angle", "camera", "expression", "emotion")
MIN_OPTIONAL = 1
"""可选描述至少要有 1 个：空镜可以没有表情，但不该一个词都没有。"""

_EMPTY_TOKENS = {"", "unknown", "none", "null", "n/a", "未知", "无", "看不清", "不可见"}


def filled(value: Any) -> bool:
    """这个字段算不算"有内容"。``unknown`` 是哨兵，不算。"""
    return str(value or "").strip().lower() not in _EMPTY_TOKENS


def missing_fields(understanding: ShotUnderstanding) -> list[str]:
    """这个镜头缺哪些字段。"""
    return [
        name for name in (*CORE_FIELDS, *OPTIONAL_FIELDS)
        if not filled(getattr(understanding, name, ""))
    ]


def is_thin(understanding: ShotUnderstanding) -> bool:
    """这个镜头是不是"感知残缺"。"""
    for name in CORE_FIELDS:
        if not filled(getattr(understanding, name, "")):
            return True
    have = sum(1 for name in OPTIONAL_FIELDS if filled(getattr(understanding, name, "")))
    return have < MIN_OPTIONAL


def perception_gaps(
    understandings: Sequence[ShotUnderstanding],
) -> list[ShotUnderstanding]:
    """按镜头号返回"感知残缺"的那些（顺序即镜头顺序）。"""
    return [item for item in understandings if is_thin(item)]


def merge_finding(
    understanding: ShotUnderstanding,
    values: dict[str, str],
) -> ShotUnderstanding:
    """把补看结果**只填进空字段**，已有内容一个字不改。

    只填空字段是有意的：补看是"换一个问法再看一次"，它同样可能看错，
    不该覆盖一次已经拿到的信息。哪边更可信我们无法判定，就不动它。
    """
    updates: dict[str, str] = {}
    for name, value in values.items():
        if name not in (*CORE_FIELDS, *OPTIONAL_FIELDS):
            continue
        if filled(getattr(understanding, name, "")):
            continue
        if filled(value):
            updates[name] = str(value).strip()
    return replace(understanding, **updates) if updates else understanding


def fill_perception_gaps(
    understandings: Sequence[ShotUnderstanding],
    shots: Sequence[ShotBoundary],
    probe: Any,
    video_path: str,
    *,
    budget: int,
    notes: list[ReviewNote],
    degradations: list[str],
) -> list[ShotUnderstanding]:
    """对残缺镜头做定向补看，返回补齐后的理解列表。

    - **只补不覆盖**（见 :func:`merge_finding`）；
    - **补看失败不致命**：单条异常吞掉，记一条降级说明继续；
    - 预算用完就停 —— 补看的成本是"抽帧 + 一次视觉调用"，必须可控。
    """
    if budget <= 0 or probe is None or not video_path:
        return list(understandings)
    by_index = {shot.index: shot for shot in shots}
    out: list[ShotUnderstanding] = []
    spent = 0
    for understanding in understandings:
        if spent >= budget or not is_thin(understanding):
            out.append(understanding)
            continue
        shot = by_index.get(understanding.shot_index)
        if shot is None:
            out.append(understanding)
            continue
        missing = missing_fields(understanding)
        spent += 1
        try:
            values = probe.describe_shot(video_path, shot, missing)
        except Exception as exc:  # noqa: BLE001 - 一次补看不到不该拖垮反推
            errors = str(exc)[:80]
            degradations.append(
                f"镜头{understanding.shot_index} 信息残缺（缺 {'/'.join(missing)}），"
                f"补看失败：{errors}"
            )
            notes.append(ReviewNote(
                path=f"creative.C2_beats[{understanding.shot_index - 1}]",
                action="probe", reason="感知残缺，定向补看", applied=False,
                detail=f"缺 {'/'.join(missing)}；补看失败：{errors}",
            ))
            out.append(understanding)
            continue
        merged = merge_finding(understanding, values or {})
        got = [name for name in missing if filled(getattr(merged, name, ""))]
        notes.append(ReviewNote(
            path=f"creative.C2_beats[{understanding.shot_index - 1}]",
            action="probe",
            reason=f"镜头{understanding.shot_index} 信息残缺（缺 {'/'.join(missing)}）",
            applied=bool(got),
            detail=(f"补到：{'/'.join(got)}" if got else "补看后仍然看不清，如实留空"),
        ))
        out.append(merged)
    return out


def gap_summary(understandings: Sequence[ShotUnderstanding]) -> str:
    """给降级说明用的一句话（还有人残缺时）。

    **必须写清"缺哪些字段"**（实测踩过）：只写「镜头 2、3、4 的信息仍然残缺」
    时，事后无法判断到底是没补上、还是补了但还差一项 ——
    看起来就像系统在说反话。写明字段名，这句话自己就能被核对。
    """
    thin = perception_gaps(understandings)
    if not thin:
        return ""
    detail = "、".join(
        f"镜头{item.shot_index}（缺 {'/'.join(missing_fields(item)[:3])}）" for item in thin[:4]
    )
    more = f" 等 {len(thin)} 个镜头" if len(thin) > 4 else ""
    return f"{detail}{more} 的画面信息仍然残缺，相关字段已如实留空"
