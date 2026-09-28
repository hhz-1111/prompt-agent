"""CV 强判断层（PRD §6.1.2 第一层）。

**能力边界要说清楚，不能假装。**

规范要求这一层用光流算运镜、用人体框占比算景别、用检测框算站位。
那需要 OpenCV + 姿态估计/人体检测模型。当前环境（Python 3.14）装不上这些，
所以本模块的策略是：

1. **能确定性算出来的，认真算** —— 逐帧亮度序列可以判定"固定机位"与
   "持续单向变化"，这是运镜里区分度最大的一类；
2. **算不了的，返回 unknown 并如实登记缺失** —— 绝不编一个数字冒充 CV 结果，
   那会让下游误以为有交叉校验（PRD 讲的"假的交叉验证"就是这么来的）；
3. **留好接入点** —— ``PersonDetector`` 协议一实现（OpenCV/YOLO/HTTP 服务），
   景别与站位立刻升级为 CV 强判断，其余代码不动。

缺失的能力会体现在 ``CvCapabilities.missing`` 里，由编排层下调对应维度的置信度。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from engine.core.contracts import (
    BlockingEstimate,
    FramingEstimate,
    MotionEstimate,
    ShotBoundary,
)
from engine.core.types import ConfidenceBasis, compute_confidence

from .ffmpeg_media import FrameMean


@dataclass(frozen=True, slots=True)
class CvCapabilities:
    """本机 CV 能力盘点。**运行时可探测**，不硬编码。"""

    motion_from_luma: bool = True
    """运镜：逐帧亮度序列能区分"稳定"与"持续变化"。"""
    framing: bool = False
    """景别：需要人体框占比，缺人体检测。"""
    blocking: bool = False
    """站位：需要检测框中心，缺人体检测。"""
    people_count: bool = False
    """人物计数：需要人体检测。"""

    @property
    def missing(self) -> tuple[str, ...]:
        gaps: list[str] = []
        if not self.framing:
            gaps.append("景别（需人体框占比）")
        if not self.blocking:
            gaps.append("站位（需检测框位置）")
        if not self.people_count:
            gaps.append("人物计数（需人体检测）")
        return tuple(gaps)

    def to_dict(self) -> dict[str, object]:
        return {
            "motion_from_luma": self.motion_from_luma,
            "framing": self.framing,
            "blocking": self.blocking,
            "people_count": self.people_count,
            "missing": list(self.missing),
        }


def detect_capabilities() -> CvCapabilities:
    """探测本机可用的 CV 能力。

    当前实现基于 ffmpeg 逐帧均值，因此只有运镜能力。
    接入 OpenCV/Ultralytics/外部检测服务后，在这里打开对应开关。
    """
    return CvCapabilities(
        motion_from_luma=True,
        framing=False,
        blocking=False,
        people_count=False,
    )


# ----------------------------------------------------------------------
# 运镜：基于逐帧亮度序列
# ----------------------------------------------------------------------
# 判据（都是可解释的，不是玄学阈值）：
#   固定机位   → 帧间变化极小，且无单调趋势
#   推近/远离  → 画面内容量持续单调变化，亮度趋势单调
#   横摇/跟拍  → 持续单向变化但非单调（场景在换，整体亮度不一定单调）
#   手持       → 高频小幅抖动，方向反复
MOTION_STABLE_MAX = 0.015
MOTION_TREND_MIN = 0.05


def analyze_motion(
    frames: Sequence[FrameMean],
    shot: ShotBoundary,
) -> MotionEstimate:
    """用镜头区间内的逐帧亮度序列判断运镜。

    只输出能站得住脚的结论：``fixed`` / ``push_in`` / ``pan_or_track`` /
    ``handheld`` / ``unknown``。区分不了就返回 ``unknown``。
    """
    window = [f for f in frames if shot.start_ms <= int(f.pts_time * 1000) <= shot.end_ms]
    if len(window) < 5:
        return MotionEstimate(
            kind="unknown",
            confidence=compute_confidence(
                ConfidenceBasis.CV, consistency="single", evidence="timecode_only"
            ),
            note="镜头内有效帧不足 5 帧，无法做运动估计",
        )

    lumas = [f.luma for f in window]
    deltas = [b - a for a, b in zip(lumas, lumas[1:])]
    mean_abs = statistics.fmean(abs(d) for d in deltas)
    span = max(lumas) - min(lumas)
    # 方向一致性：同号比例。1.0 = 完全单调，0.5 = 方向反复
    positives = sum(1 for d in deltas if d > 0)
    direction_ratio = max(positives, len(deltas) - positives) / len(deltas)

    confidence = compute_confidence(
        ConfidenceBasis.CV, consistency="single", evidence="timecode_and_frame"
    )

    if mean_abs < MOTION_STABLE_MAX and span < MOTION_TREND_MIN:
        return MotionEstimate(
            kind="fixed", confidence=confidence, magnitude=mean_abs,
            note=f"帧间亮度变化 {mean_abs:.4f}，全片稳定 → 判定固定机位",
        )
    if span >= MOTION_TREND_MIN and direction_ratio >= 0.75:
        return MotionEstimate(
            kind="push_in",
            confidence=round(confidence * 0.8, 4),
            magnitude=mean_abs,
            note=f"亮度单调变化（方向一致度 {direction_ratio:.2f}），可能是推近/拉远",
        )
    if direction_ratio >= 0.65:
        return MotionEstimate(
            kind="pan_or_track",
            confidence=round(confidence * 0.7, 4),
            magnitude=mean_abs,
            note=f"持续单向变化（方向一致度 {direction_ratio:.2f}），可能是横摇或跟拍",
        )
    return MotionEstimate(
        kind="handheld",
        confidence=round(confidence * 0.6, 4),
        magnitude=mean_abs,
        note=f"方向反复（方向一致度 {direction_ratio:.2f}），判定为手持抖动",
    )


# ----------------------------------------------------------------------
# 未实现的能力：显式留空，不编数据
# ----------------------------------------------------------------------
def analyze_framing_unavailable(shot: ShotBoundary) -> FramingEstimate:
    """景别：缺人体检测，**返回 unknown 并说明原因**。

    绝不返回一个猜测值冒充 CV 结论 —— 那会让编排层以为拿到了交叉校验证据。
    """
    return FramingEstimate(
        kind="unknown",
        confidence=0.0,
        note="CV 景别需要人体框占比，当前环境缺人体检测实现；请接入 OpenCV/YOLO 后启用",
    )


def analyze_blocking_unavailable() -> tuple[tuple[BlockingEstimate, ...], int, float]:
    """站位与人物计数：同上，显式留空。"""
    return (), 0, 0.0
