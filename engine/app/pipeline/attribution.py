"""台词归属覆盖度自检：**发现"这句不知道谁说的"，回去取证再判一次**。

## 为什么必须有它（留出集实测暴露的）

一条双人示范片子的交付物里：

```
镜头1（人物2）："腰难受，你就这样子拉伸；"
镜头2："背难受，就这样子拉伸；"      ← 没有说话人
镜头3："坐骨屁股难受，你就这样子拉伸；"  ← 没有说话人
镜头4："膝盖难受，你就这样子拉伸。"      ← 没有说话人
```

4 句台词只判出第 1 句的说话人，G6 因此判定不通过。
和"感知残缺"是同一类问题：**信息在某个环节丢了，而没人回头补**。

## 两层修法

1. **把答案递给它**（确定性）：`ShotDialogueContext` 现在带上 ``action``/``expression``
   —— 视觉理解经常直接写着「紫衣女子张口说话」，这就是最硬的证据。
2. **还是判不出就去取证**（本模块）：对说话人缺失的台词所在镜头抽帧，
   专门问一句「这个镜头里谁在说话」，把回答作为**追加证据**再判一次。
   仍然判不出就如实留空 —— 不许猜。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from engine.core.contracts import (
    DialogueTurn,
    ProbeRequest,
    ReviewNote,
    ShotBoundary,
    ShotDialogueContext,
)
from engine.core.types import ConfidenceBasis, compute_confidence

_EMPTY_TOKENS = {"", "unknown", "none", "null", "n/a", "未知", "无", "?"}


def fallback_turn(text: str, shot_index: int) -> DialogueTurn:
    """归属失败时的兜底：**把整段转写挂到第一个镜头、说话人留空**。

    为什么必须有（实测踩过）：收紧模型调用超时之后，归属调用可能超时
    （flash 单次实测 236 秒）—— 那时 ``attribute`` 返回空，交付物里
    **一句台词都没有**，真实存在的转写被静默扔掉了。
    转写本身是**真实数据**：宁可"说话人留空、请人工确认"，
    也不能把用户视频里真说过的话弄丢（红线 5 管的是"不编"，不是"不许留空"）。
    """
    return DialogueTurn(
        shot_index=int(shot_index),
        speaker="",
        text=str(text or "").strip(),
        tone="",
        confidence=compute_confidence(
            ConfidenceBasis.INFERRED, consistency="single", evidence="timecode_only"
        ),
    )


def has_speaker(turn: DialogueTurn) -> bool:
    return str(turn.speaker or "").strip().lower() not in _EMPTY_TOKENS


def unspoken_turns(turns: Sequence[DialogueTurn]) -> list[DialogueTurn]:
    """没有说话人的台词。"""
    return [turn for turn in turns if not has_speaker(turn)]


def speaker_question(context: ShotDialogueContext | None) -> str:
    """给取证模型的问法：只要"谁在说话"，不要它复述剧情。"""
    who = "、".join(context.people) if context and context.people else "画面里的人"
    return (
        f"这个镜头里正在说话的是谁？用画面里稳定的外观特征回答"
        f"（如「紫色上衣女子」）。画面里的人有：{who}。"
        f"如果看不出谁在说话，就回答「看不出」。"
    )


def collect_speaker_evidence(
    turns: Sequence[DialogueTurn],
    shots: Sequence[ShotBoundary],
    contexts: Sequence[ShotDialogueContext],
    probe: Any,
    video_path: str,
    *,
    budget: int,
    notes: list[ReviewNote],
    degradations: list[str],
) -> dict[int, str]:
    """对"说话人缺失"的镜头取证，返回 ``{镜头号: 看到的证据}``。

    - 同一个镜头只问一次（几句台词落在同一镜时，答案共用）；
    - 单次失败不致命，记一条说明继续；
    - 预算用完就停。
    """
    if budget <= 0 or probe is None or not video_path:
        return {}
    by_index = {shot.index: shot for shot in shots}
    context_by_index = {item.shot_index: item for item in contexts}
    evidence: dict[int, str] = {}
    asked: set[int] = set()
    spent = 0
    for turn in unspoken_turns(turns):
        index = turn.shot_index
        # **按"问过没"去重，不是按"拿到答案没"**（实测踩过：拿不到答案时
        # 同一镜头的第二句台词会把同一个问题再问一遍，白花一次取证预算）
        if index in asked or spent >= budget:
            continue
        shot = by_index.get(index)
        if shot is None:
            continue
        asked.add(index)
        spent += 1
        request = ProbeRequest(
            question=speaker_question(context_by_index.get(index)),
            time_start=shot.start_ms / 1000,
            time_end=shot.end_ms / 1000,
            region="full",
            reason=f"镜头{index} 的台词没有说话人，去看谁在说话",
        )
        try:
            finding = probe.look(video_path, request)
        except Exception as exc:  # noqa: BLE001
            degradations.append(f"镜头{index} 说话人取证失败：{str(exc)[:80]}")
            notes.append(ReviewNote(
                path=f"creative.C2_beats[{index - 1}].dialogue.speaker",
                action="probe", reason=request.reason, applied=False,
                detail=f"取证失败：{str(exc)[:80]}",
            ))
            continue
        answer = str(finding.answer or "").strip()
        notes.append(ReviewNote(
            path=f"creative.C2_beats[{index - 1}].dialogue.speaker",
            action="probe", reason=request.reason, applied=bool(answer),
            detail=f"画面回答：{answer[:70] or '看不出'}",
        ))
        if answer:
            evidence[index] = answer
    return evidence


def merge_speakers(
    turns: Sequence[DialogueTurn],
    resolved: dict[int, str],
    known: Sequence[str],
) -> list[DialogueTurn]:
    """把取证得到的结论落到台词上。**只填本来空着的说话人。**

    ``resolved`` 是 ``{镜头号: 结论}``，结论必须能对上一个人物名
    （``known`` 里的某一个）。**对不上就不填** —— 那一句就保持留空，
    让 G6 去报，而不是塞一个"看起来像"的名字。
    """
    out: list[DialogueTurn] = []
    for turn in turns:
        if has_speaker(turn):
            out.append(turn)
            continue
        answer = str(resolved.get(turn.shot_index) or "")
        matched = _match_person(answer, known)
        out.append(replace(turn, speaker=matched) if matched else turn)
    return out


def _match_person(answer: str, known: Sequence[str]) -> str:
    """从取证回答里认出是哪个人物。

    判据取得很窄（**只认包含关系**）：取证回答是自由文本，
    宽判据会把"看起来像"的名字塞进台词，那是猜。
    """
    text = str(answer or "").strip()
    if not text:
        return ""
    for name in known:
        if name and name in text:
            return name
    return ""


def attribution_summary(turns: Sequence[DialogueTurn]) -> str:
    """还有人没说话人时给用户的一句话（写清是哪几句）。"""
    missing = unspoken_turns(turns)
    if not missing:
        return ""
    preview = "、".join(f"「{turn.text[:12]}」" for turn in missing[:3])
    more = f" 等 {len(missing)} 句" if len(missing) > 3 else ""
    return f"{preview}{more} 仍然判不出说话人，已如实留空（请在【音色参考】里补）"
