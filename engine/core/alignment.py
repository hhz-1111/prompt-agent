"""内容对齐：用户内容 → 提示词位置（PRD §5 步骤 4 / F6）。

**这一步解决什么**：反推只能把台词按时间归到镜头（而且 MiMo 的 ASR 不给
词级时间戳，只能做整段归属）。用户真正需要的是把自己的台词稿
一次粘进来，系统自动分派到各个位置。

设计上有三条硬要求：

1. **保留双向坐标**（PRD F6.4 特别强调）：用户内容块同时记录
   「在原稿里的字符区间」与「绑定到的原片时间码」。
   用户既知道"我这句对应原片第 3 秒"，也能随时切回自己的原稿上下文。
2. **对齐要保序**：用动态规划求全局最优，不允许出现"第 3 句排到第 1 个位置"
   这种交叉。人写稿子是按顺序写的，保序约束能显著提升准确率。
3. **置信度分级处置**：<0.70 必须用户确认（PRD F6.3 写 0.75，此处对齐
   MULTIMODAL §4.3 的 0.70 分界；两者都落在"必须确认"区间，取更严的那个）。

**判据为什么是三层而不是一层**：单靠文本相似度，用户把"陈姐，好久不见"
改成"王姐，你这气色"就完全匹配不上；单靠时长，短句会乱配。
叠起来各自补位 —— 与反推里「CV 强判断 + 模型补位」是同一套思路。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .types import (
    CONFIDENCE_ACCEPT,
    CONFIDENCE_REVIEW,
    Span,
)

# 语速估算（AGENT §5.4）：中文常规 5 字/秒，慢 4、快 6。
DEFAULT_CHARS_PER_SEC = 5.0

# 各判据在总分里的权重。**时长权重最高**：一个 1.2 秒的位置塞 18 个字，
# 无论文字多像都不成立 —— 这是最常见的对齐事故。
WEIGHT_SIMILARITY = 0.45
WEIGHT_DURATION_FIT = 0.35
WEIGHT_SPEAKER = 0.20

# 相邻位置之间"跳过一个位置"的惩罚。用来避免把内容硬塞进错误的位置 ——
# 宁可留空让用户补，也不乱配（红线 7 的同一精神）。
GAP_PENALTY = 0.15

# 低于此分不绑定，标记为"需用户确认"。与 CONFIDENCE_ACCEPT 一致。
BIND_THRESHOLD = CONFIDENCE_ACCEPT

# 说话人明确不一致时的分配惩罚。
# 把台词放进别人的嘴里不是"匹配得不够好"，而是**明确错误** ——
# 所以不光要标黄，还要让它在全局分配里**竞争不过**别的位置，
# 从而更倾向于留空而不是错配。
SPEAKER_CONFLICT_PENALTY = 0.45


class BindingStatus(str, Enum):
    """一个位置的绑定状态。**门禁据此决定能否进入组装**。"""

    BOUND = "bound"        # 已绑定且置信度达标
    PENDING = "pending"    # 绑定了但置信度不足，必须用户确认
    UNMATCHED = "unmatched"  # 有位置但没有内容
    ORPHAN = "orphan"      # 有内容但没有位置可放


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """用户原稿里的一段内容。

    ``char_start`` / ``char_end`` 是它在**原稿里的字符区间**，
    这是 PRD F6.4 要求的"保留引用位置"的一半 ——
    用户点任意一条对齐结果都能跳回原稿的那一段。
    """

    index: int
    content: str
    char_start: int
    char_end: int
    speaker: str = ""
    timecode_hint_sec: float | None = None
    """用户若用 ``00:03 内容`` 形态写了时间码，记在这里。"""

    @property
    def char_span(self) -> Span:
        return Span(self.char_start, self.char_end)

    @property
    def char_count(self) -> int:
        return sum(1 for ch in self.content if not ch.isspace())

    def speech_seconds(self, chars_per_sec: float = DEFAULT_CHARS_PER_SEC) -> float:
        """按语速估算这段内容要念多久。"""
        if chars_per_sec <= 0:
            return 0.0
        return round(self.char_count / chars_per_sec, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "content": self.content,
            "speaker": self.speaker,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "estimated_seconds": self.speech_seconds(),
            "timecode_hint_sec": self.timecode_hint_sec,
        }


@dataclass(frozen=True, slots=True)
class AlignmentSlot:
    """提示词里一个可放台词的位置。

    ``reference_line`` 是原片在这个位置说的台词（来自 ASR），
    它是文本相似度判据的比对基准。
    """

    index: int
    path: str
    label: str
    span: Span
    reference_line: str = ""
    speaker: str = ""
    capacity_chars: int = 0

    @property
    def duration_sec(self) -> float:
        return self.span.duration_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "path": self.path,
            "label": self.label,
            "timecode": self.span.format_timecode(),
            "span": self.span.to_dict(),
            "reference_line": self.reference_line,
            "speaker": self.speaker,
            "capacity_chars": self.capacity_chars,
        }


@dataclass(frozen=True, slots=True)
class AlignmentScore:
    """一次匹配的分数构成。**拆开是为了让用户看懂为什么这么对齐。**"""

    total: float
    similarity: float
    duration_fit: float
    speaker_match: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "similarity": round(self.similarity, 4),
            "duration_fit": round(self.duration_fit, 4),
            "speaker_match": round(self.speaker_match, 4),
        }


@dataclass(slots=True)
class AlignmentBinding:
    """一条对齐结果。三列对照表的每一行就是它。"""

    slot: AlignmentSlot
    block: ContentBlock | None
    score: AlignmentScore
    status: BindingStatus
    reason: str = ""
    conflicts: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """综合置信度，0~1。

        **为什么不套用 ``compute_confidence`` 的乘法公式**：
        那一套是给"反推"用的 —— 基础分代表来源可靠性（CV 0.95 / 模型 0.70），
        再乘一致性系数与证据完整度。但对齐不是量测，它的分数本身就是
        三个判据的加权和，再乘一遍系数会**双重惩罚**：

            文字高度相似 + 时长贴合 + 说话人一致 → 原始分 0.94
            套公式：0.85 × 0.85 × 0.90 = 0.65 → 被判成"把握不足"

        实测就是这样：一个明显正确的对齐只拿到 0.54，逼用户确认本该确信的事。

        这里改成**直接映射**：``0.35 + 0.65 × score``。
        用意：没有证据时不给高分（下限 0.35），判据全中时给满分。
        标定后 —— 完全一致 ≈1.00、高度相似 ≈0.75、弱相似 ≈0.5。
        与 PRD 的 0.75 / MULTIMODAL 的 0.70 分界都兼容。
        """
        return round(min(1.0, 0.35 + 0.65 * self.score.total), 4)

    @property
    def requires_confirmation(self) -> bool:
        """低于阈值就必须用户确认（PRD F6.3）。"""
        return self.status is BindingStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot.to_dict(),
            "block": self.block.to_dict() if self.block else None,
            "score": self.score.to_dict(),
            "confidence": self.confidence,
            "status": self.status.value,
            "requires_confirmation": self.requires_confirmation,
            "reason": self.reason,
            "conflicts": list(self.conflicts),
        }


@dataclass(slots=True)
class AlignmentReport:
    """一次对齐的完整产物。"""

    bindings: list[AlignmentBinding] = field(default_factory=list)
    orphans: list[ContentBlock] = field(default_factory=list)
    """有内容但没位置可放 —— 通常是用户多写了几句，或位置太少。"""
    chars_per_sec: float = DEFAULT_CHARS_PER_SEC
    notes: list[str] = field(default_factory=list)

    @property
    def pending_count(self) -> int:
        return sum(1 for b in self.bindings if b.status is BindingStatus.PENDING)

    @property
    def unmatched_count(self) -> int:
        return sum(1 for b in self.bindings if b.status is BindingStatus.UNMATCHED)

    @property
    def bound_count(self) -> int:
        return sum(1 for b in self.bindings if b.status is BindingStatus.BOUND)

    @property
    def has_unresolved(self) -> bool:
        """存在未决状态 —— 用于 F6.6 的强制门禁。"""
        return bool(
            self.pending_count or self.unmatched_count or self.orphans
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bindings": [b.to_dict() for b in self.bindings],
            "orphans": [b.to_dict() for b in self.orphans],
            "chars_per_sec": self.chars_per_sec,
            "notes": list(self.notes),
            "summary": {
                "total_slots": len(self.bindings),
                "bound": self.bound_count,
                "pending": self.pending_count,
                "unmatched": self.unmatched_count,
                "orphans": len(self.orphans),
                "has_unresolved": self.has_unresolved,
            },
        }


# --------------------------------------------------------------------------
# 解析：三种输入形态（PRD F6.1）
# --------------------------------------------------------------------------
# 形态 1：纯脚本（一行一句）
# 形态 2：说话人：内容
# 形态 3：00:03 内容
_SPEAKER_RE = re.compile(r"^\s*([^：:\n]{1,12})\s*[：:]\s*(.+)$")
_TIMECODE_RE = re.compile(r"^\s*(?:\[)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?(?:\])?\s*[：:，,\s]?\s*(.+)$")
_TIME_RANGE_RE = re.compile(
    r"^\s*(?:\[)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\s*[-–~至]\s*(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?(?:\])?\s*[：:，,\s]?\s*(.+)$"
)


def parse_user_content(raw: str) -> list[ContentBlock]:
    """把用户粘贴的稿子解析成内容块。

    支持三种形态，可以混用：

        朋友都问我是不是做项目了          ← 纯脚本
        王姐：你这气色也太好了吧          ← 说话人：内容
        00:03 哪有啊，我就是在家练普拉提   ← 时间码 内容

    解析时**保留每块在原稿里的字符区间** —— 这是"双向引用位置"的一半，
    少了它用户就没法从对齐结果跳回自己的原稿。所以这里不能先 strip 再切，
    必须按原始字符串算偏移。
    """
    blocks: list[ContentBlock] = []
    cursor = 0
    index = 0

    for raw_line in raw.splitlines(keepends=True):
        line_start = cursor
        cursor += len(raw_line)
        stripped = raw_line.rstrip("\r\n")
        if not stripped.strip():
            continue
        # 前导空白也算进原稿偏移，但不进内容
        indent = len(stripped) - len(stripped.lstrip())
        body = stripped.strip()
        if not body:
            continue

        speaker = ""
        hint: float | None = None
        content = body
        content_offset = indent

        range_match = _TIME_RANGE_RE.match(body)
        if range_match:
            hint = _to_seconds(range_match.group(1), range_match.group(2), range_match.group(3))
            content = range_match.group(7).strip()
            content_offset = stripped.index(content) if content in stripped else indent
        else:
            time_match = _TIMECODE_RE.match(body)
            if time_match:
                hint = _to_seconds(
                    time_match.group(1), time_match.group(2), time_match.group(3)
                )
                content = time_match.group(4).strip()
                content_offset = stripped.index(content) if content in stripped else indent
            else:
                speaker_match = _SPEAKER_RE.match(body)
                if speaker_match:
                    candidate = speaker_match.group(1).strip()
                    rest = speaker_match.group(2).strip()
                    # 排除「他说：…」这类误判：说话人应当是短名词
                    if rest and not any(ch in candidate for ch in "，。！？、"):
                        speaker = candidate
                        content = rest
                        content_offset = stripped.index(rest) if rest in stripped else indent

        if not content:
            continue
        blocks.append(ContentBlock(
            index=index,
            content=content,
            char_start=line_start + content_offset,
            char_end=line_start + content_offset + len(content),
            speaker=speaker,
            timecode_hint_sec=hint,
        ))
        index += 1

    return blocks


def _to_seconds(minutes: str, seconds: str, millis: str | None) -> float:
    value = int(minutes) * 60 + int(seconds)
    if millis:
        value += int(millis.ljust(3, "0")) / 1000.0
    return round(value, 3)


# --------------------------------------------------------------------------
# 打分
# --------------------------------------------------------------------------
def text_similarity(a: str, b: str) -> float:
    """文本相似度。

    用 ``difflib`` 而不是引第三方：我们比较的是**中文短句**，
    字符级 LCS 已经够用，而引一个 embedding 服务会带来
    网络依赖、成本、以及不可复现的问题。
    """
    left = _normalize(a)
    right = _normalize(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return round(difflib.SequenceMatcher(None, left, right).ratio(), 4)


_PUNCT_RE = re.compile("[" + "\\s\\u3000" + "，。！？、；：" + "\u201c\u201d\u2018\u2019"
                       + "（）()\\[\\]…—\\-" + "]+")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text or "")


def duration_fit(block: ContentBlock, slot: AlignmentSlot, chars_per_sec: float) -> float:
    """内容量与位置时长的适配度，0~1。

    **最容易被忽略但最致命的一维**：1.2 秒的位置塞 18 个字，
    文字再像也不成立。PRD §5 步骤 5 给的正是这种提示。

    判据：估算时长落在位置时长的 [0.5, 1.25] 区间内算贴合；
    超出越多扣分越狠。
    """
    if slot.duration_sec <= 0:
        return 0.5
    needed = block.speech_seconds(chars_per_sec)
    if needed <= 0:
        return 0.5
    ratio = needed / slot.duration_sec
    if 0.5 <= ratio <= 1.25:
        return 1.0
    if ratio < 0.5:
        # 内容太少：位置会空着，不算错但要扣一点
        return round(0.6 + 0.8 * ratio, 4)
    # 内容超长：这是真问题，衰减要快
    return round(max(0.0, 1.0 - (ratio - 1.25) * 0.8), 4)


def speaker_match(block: ContentBlock, slot: AlignmentSlot) -> float:
    """说话人一致性。返回的是"判据是否可用 + 结论"。

    双方都有说话人时才真正参与计分；否则返回中性值 —— 但中性值取 0.4
    而不是 0.5，因为"没有说话人信息"**不该把总分抬起来**：
    实测两份毫无共同点的内容能靠这个中性分拿到 0.65 的虚高分。
    """
    b = (block.speaker or "").strip()
    s = (slot.speaker or "").strip()
    if not b or not s:
        return 0.4
    if b == s:
        return 1.0
    # 说话人名字可能被用户改过（"陈姐"→"王姐"），用相似度兜一下
    return round(text_similarity(b, s) * 0.6, 4)


def speakers_conflict(block: ContentBlock, slot: AlignmentSlot) -> bool:
    """双方都有说话人且明确不同 —— 这是**硬冲突**，必须用户确认。

    为什么单列：把台词放进别人的嘴里不是"匹配得不够好"，
    而是**明确的错误**。不能靠加权平均把它稀释掉。
    """
    b = (block.speaker or "").strip()
    s = (slot.speaker or "").strip()
    return bool(b and s and b != s)


def score_pair(
    block: ContentBlock,
    slot: AlignmentSlot,
    chars_per_sec: float,
) -> AlignmentScore:
    """给「内容块 → 位置」这一对打分。

    加权方式：说话人判据**可用时**才占 20%，不可用时把它的权重
    让给文本相似度 —— 否则"没有说话人信息"会变成加分的漏洞。
    """
    similarity = text_similarity(block.content, slot.reference_line)
    fit = duration_fit(block, slot, chars_per_sec)
    speaker = speaker_match(block, slot)

    speaker_usable = bool(
        (block.speaker or "").strip() and (slot.speaker or "").strip()
    )
    if speaker_usable:
        total = (
            WEIGHT_SIMILARITY * similarity
            + WEIGHT_DURATION_FIT * fit
            + WEIGHT_SPEAKER * speaker
        )
    else:
        # 把说话人的 20% 按比例分给另两维
        scale = 1.0 / (WEIGHT_SIMILARITY + WEIGHT_DURATION_FIT)
        total = (
            WEIGHT_SIMILARITY * scale * similarity
            + WEIGHT_DURATION_FIT * scale * fit
        )
    return AlignmentScore(
        total=round(total, 4),
        similarity=similarity,
        duration_fit=fit,
        speaker_match=speaker,
    )


# --------------------------------------------------------------------------
# 全局最优分配（保序动态规划）
# --------------------------------------------------------------------------
def align(
    blocks: Sequence[ContentBlock],
    slots: Sequence[AlignmentSlot],
    *,
    chars_per_sec: float = DEFAULT_CHARS_PER_SEC,
    bind_threshold: float = BIND_THRESHOLD,
) -> AlignmentReport:
    """把内容块分配到位置。

    用动态规划求**保序**全局最优：允许某个位置空着，也允许某个内容块
    没地方放（记为 orphan），但不允许交叉分配。

    为什么要保序：用户写稿是按顺序写的，稿子的第 3 句不可能对应
    视频的第 1 个位置（除非用户明确用时间码指定）。保序约束把
    搜索空间从 n! 降到 n²，既快又准。
    """
    report = AlignmentReport(chars_per_sec=chars_per_sec)
    if not slots:
        report.orphans = list(blocks)
        report.notes.append("提示词里没有可放置台词的位置")
        return report
    if not blocks:
        report.bindings = [
            AlignmentBinding(
                slot=slot, block=None,
                score=AlignmentScore(0.0, 0.0, 0.0, 0.0),
                status=BindingStatus.UNMATCHED,
                reason="用户未提供内容",
            )
            for slot in slots
        ]
        report.notes.append("未提供内容，全部位置保持原片结构")
        return report

    scores = [
        [score_pair(block, slot, chars_per_sec) for slot in slots]
        for block in blocks
    ]
    # 说话人明确冲突的对，在全局分配里降权 —— 让它竞争不过正确的位置。
    # 注意循环变量别叫 score，否则会遮蔽下面逐条用的那个。
    allocating = [
        [
            pair.total - (
                SPEAKER_CONFLICT_PENALTY if speakers_conflict(block, slot) else 0.0
            )
            for slot, pair in zip(slots, row)
        ]
        for block, row in zip(blocks, scores)
    ]
    assignment = _order_preserving_assignment(allocating)

    used_blocks: set[int] = set()
    for slot_index, slot in enumerate(slots):
        block_index = assignment.get(slot_index)
        if block_index is None:
            report.bindings.append(AlignmentBinding(
                slot=slot, block=None,
                score=AlignmentScore(0.0, 0.0, 0.0, 0.0),
                status=BindingStatus.UNMATCHED,
                reason="未分配内容（保留原片结构，台词清空）",
            ))
            continue
        block = blocks[block_index]
        used_blocks.add(block_index)
        score = scores[block_index][slot_index]
        binding = AlignmentBinding(slot=slot, block=block, score=score, status=BindingStatus.BOUND)
        binding.conflicts = _slot_conflicts(block, slot, chars_per_sec)

        if score.total < bind_threshold or binding.conflicts:
            binding.status = BindingStatus.PENDING
            parts: list[str] = []
            if score.total < bind_threshold:
                parts.append(f"匹配把握不足（{score.total:.2f} < {bind_threshold}）")
            if binding.conflicts:
                parts.append("；".join(binding.conflicts))
            binding.reason = "；".join(parts) + "。请确认或手动调整"
        else:
            binding.reason = _explain(score)
        report.bindings.append(binding)

    report.orphans = [b for i, b in enumerate(blocks) if i not in used_blocks]
    if report.orphans:
        report.notes.append(
            f"{len(report.orphans)} 段内容没有位置可以放置 —— "
            f"可能是位置不够，或内容比原片多。请合并或删减。"
        )
    if report.pending_count:
        report.notes.append(
            f"{report.pending_count} 处对齐把握不足，必须确认后才能进入组装"
        )
    return report


def _order_preserving_assignment(scores: list[list[float]]) -> dict[int, int]:
    """保序最大权分配。返回 ``{位置下标: 内容块下标}``。

    状态：``dp[i][j]`` = 前 i 个内容块、前 j 个位置的最优总分。
    转移三选一：跳过位置 / 匹配 / 跳过内容块。

    入参是**分配权重**而不是展示分数 —— 两者可以不同：
    展示分数要如实反映匹配质量，分配权重还要计入硬冲突的惩罚。
    """
    n = len(scores)
    m = len(scores[0]) if n else 0
    if not n or not m:
        return {}

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    choice = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i][j - 1]
            src = "skip_slot"
            if dp[i - 1][j] > best:
                best = dp[i - 1][j]
                src = "skip_block"
            matched = dp[i - 1][j - 1] + scores[i - 1][j - 1]
            if matched > best:
                best = matched
                src = "match"
            dp[i][j] = best
            choice[i][j] = src

    out: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        step = choice[i][j]
        if step == "match":
            out[j - 1] = i - 1
            i -= 1
            j -= 1
        elif step == "skip_block":
            i -= 1
        else:
            j -= 1
    return out


def _slot_conflicts(
    block: ContentBlock,
    slot: AlignmentSlot,
    chars_per_sec: float,
) -> list[str]:
    """这一对匹配有没有硬问题 —— 要直接告诉用户。"""
    conflicts: list[str] = []
    needed = block.speech_seconds(chars_per_sec)
    if slot.duration_sec > 0 and needed > slot.duration_sec * 1.4:
        limit = max(1, int(slot.duration_sec * chars_per_sec))
        conflicts.append(
            f"时长超限：{block.char_count} 字按语速约需 {needed:.1f} 秒，"
            f"但该位置只有 {slot.duration_sec:.1f} 秒（约 {limit} 字）"
        )
    if slot.speaker and block.speaker and slot.speaker != block.speaker:
        conflicts.append(
            f"说话人不一致：位置属于「{slot.speaker}」，内容是「{block.speaker}」"
        )
    return conflicts


def _explain(score: AlignmentScore) -> str:
    reasons: list[str] = []
    if score.similarity >= 0.6:
        reasons.append("与原片台词高度相似")
    elif score.similarity >= 0.3:
        reasons.append("与原片台词部分相似")
    if score.duration_fit >= 0.9:
        reasons.append("时长贴合")
    elif score.duration_fit < 0.6:
        reasons.append("时长偏紧")
    if score.speaker_match >= 1.0:
        reasons.append("说话人一致")
    return "；".join(reasons) or "按顺序分配"


def collect_slots(ir: Mapping[str, Any]) -> list[AlignmentSlot]:
    """从 IR 里抽出**可放置台词的位置**。

    位置来源：
    - ``creative.C2_beats[].dialogue.line`` —— 每个镜头一条台词位；
    - 时间区间取该镜头的起止（用于时长适配判据与三列对照表）；
    - ``reference_line`` 取当前值 —— 它通常就是反推来的原片台词，
      是文本相似度判据的比对基准。

    **不收集锁定位**：禁止项、站位那些地方放不了台词。
    """
    from . import framework as fw

    beats = ir.get("creative", {}).get("C2_beats", [])
    slots: list[AlignmentSlot] = []
    cursor_ms = 0
    for position, beat in enumerate(beats):
        duration = float(beat.get("duration_sec") or 0)
        span = Span(cursor_ms, cursor_ms + int(round(duration * 1000)))
        cursor_ms = span.end_ms

        dialogue = beat.get("dialogue") or {}
        path = f"creative.C2_beats[{position}].dialogue"
        if fw.spec_for(f"{path}.line") is None:
            continue
        # 说话人只取 beat 自己的。**不要**用"全片只有一个音色"去反推 ——
        # 那会给每个位置硬套同一个说话人，让 speaker_match 判据虚高，
        # 反而掩盖真正的不一致。
        slots.append(AlignmentSlot(
            index=len(slots),
            path=path,
            label=f"镜头{beat.get('shot_id') or position + 1}",
            span=span,
            reference_line=str(dialogue.get("line") or ""),
            speaker=str(dialogue.get("speaker") or ""),
            capacity_chars=max(1, int(span.duration_sec * DEFAULT_CHARS_PER_SEC)),
        ))
    return slots


def apply_alignment(ir: dict[str, Any], report: AlignmentReport) -> list[str]:
    """把对齐结果写进 IR。返回被写入的路径。

    **只写 BOUND 状态的绑定**。PENDING 的必须等用户确认 ——
    这正是 F6.3「置信度分级处置」与 F6.6「强制门禁」的落地：
    未经确认的猜测不许污染交付物。
    """
    from . import framework as fw

    written: list[str] = []
    for binding in report.bindings:
        if binding.status is not BindingStatus.BOUND or binding.block is None:
            continue
        for field_name, value in (
            ("line", binding.block.content),
            ("speaker", binding.block.speaker or binding.slot.speaker),
        ):
            path = f"{binding.slot.path}.{field_name}"
            if fw.spec_for(path) is None:
                continue
            fw.set_in(ir, path, value)
            written.append(path)
    return written
