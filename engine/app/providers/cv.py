"""CV 强判断层（PRD §6.1.2 第一层）。

**能力边界要说清楚，不能假装。**

规范要求这一层用光流算运镜、用人体框占比算景别、用检测框算站位。
本模块的策略是：

1. **能确定性算出来的，认真算** —— 逐帧亮度序列判定运镜；
   接入 YOLO 后，人体框高度占比算景别、框中心算站位、去重计数算人数；
2. **算不了的，返回 unknown 并如实登记缺失** —— 绝不编一个数字冒充 CV 结果，
   那会让下游误以为有交叉校验（PRD 讲的"假的交叉验证"就是这么来的）；
3. **能力运行时可探测** —— ``detect_capabilities()`` 报告本机有什么、缺什么，
   缺失项对应的维度置信度会自动从 ``CV``(0.95) 降到 ``VLM_ALONE``(0.70)。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
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
    """景别：需要人体框高度占比。"""
    blocking: bool = False
    """站位：需要检测框中心 x。"""
    people_count: bool = False
    """人物计数：需要人体检测。"""
    person_detector: str = ""
    """人体检测实现名（如 ``yolov8n``）。空串表示没有。"""

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
            "person_detector": self.person_detector,
            "missing": list(self.missing),
        }


def detect_capabilities() -> CvCapabilities:
    """探测本机可用的 CV 能力。

    人体检测是**可选增强**：装了 ultralytics 就打开景别/站位/计数，
    没装就退回"模型单独判断"。两条路都必须能跑通。
    """
    name = _probe_person_detector()
    return CvCapabilities(
        motion_from_luma=True,
        framing=bool(name),
        blocking=bool(name),
        people_count=bool(name),
        person_detector=name,
    )


def _probe_person_detector() -> str:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return ""
    return "yolov8n"


# ----------------------------------------------------------------------
# 人体检测（可选增强）
# ----------------------------------------------------------------------
MIN_PERSON_CONF = 0.35
"""人体检测置信度下限。实测背景人物常在 0.4~0.7，虚警多在 0.3 以下。"""

BACKGROUND_PERSON_HEIGHT = 0.25
"""高度占画面比例低于此值的人视为背景人物。

菜市场、教室这类场景背景里全是路人。它们不该进"人物清单"
（会给每个路人建多视图、浪费参考图配额），但要参与站位判断。
"""


@dataclass(frozen=True, slots=True)
class PersonBox:
    """归一化到 0~1 的人体框。"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def height_ratio(self) -> float:
        return self.y2 - self.y1

    @property
    def area_ratio(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def side(self) -> str:
        """按中心 x 判定左/中/右。三分法，不做模糊地带。"""
        if self.center_x < 0.4:
            return "left"
        if self.center_x > 0.6:
            return "right"
        return "center"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "center_x": round(self.center_x, 3),
            "height_ratio": round(self.height_ratio, 3),
            "area_ratio": round(self.area_ratio, 3),
            "side": self.side,
            "confidence": round(self.confidence, 3),
        }


class YoloPersonDetector:
    """``ObjectDetector`` 端口的 ultralytics YOLO 实现。

    只用 COCO 的 person 类（class 0），不做姿态估计 ——
    PRD 要的景别与站位只需要人体框，姿态点是多余的算力。

    模型权重首次使用会自动下载（约 6MB）。
    """

    def __init__(self, model_name: str = "yolov8n", workspace: Path | None = None) -> None:
        self.model_name = model_name
        self.workspace = workspace
        self._model = None

    @property
    def available(self) -> bool:
        return bool(_probe_person_detector())

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            weights = f"{self.model_name}.pt"
            if self.workspace is not None:
                # 权重落到工作区，避免污染仓库根目录
                self.workspace.mkdir(parents=True, exist_ok=True)
                local = self.workspace / weights
                weights = str(local) if local.exists() else weights
            self._model = YOLO(weights)
        return self._model

    def detect(self, frame_path: str) -> list[PersonBox]:
        """返回该帧里的人体框（归一化坐标）。"""
        model = self._load()
        result = model(frame_path, verbose=False)[0]
        height, width = result.orig_shape
        if not height or not width:
            return []
        boxes: list[PersonBox] = []
        for box in result.boxes:
            if int(box.cls[0]) != 0:  # COCO class 0 = person
                continue
            confidence = float(box.conf[0])
            if confidence < MIN_PERSON_CONF:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            boxes.append(PersonBox(
                x1=max(0.0, x1 / width), y1=max(0.0, y1 / height),
                x2=min(1.0, x2 / width), y2=min(1.0, y2 / height),
                confidence=confidence,
            ))
        boxes.sort(key=lambda b: -b.area_ratio)
        return boxes


def framing_from_person(box: PersonBox) -> FramingEstimate:
    """人体框 → 景别。

    用**高度占比**而不是面积占比：竖屏 9:16 下人体是细长的，
    面积占比会被宽度稀释，而高度占比直接对应"人物占画面多大"。
    实测：菜市场采访那条主人物高度占 0.92，正好是近景；背景路人只占 0.09。
    """
    ratio = box.height_ratio
    confidence = compute_confidence(
        ConfidenceBasis.CV, consistency="single", evidence="timecode_and_frame"
    )
    if ratio >= 1.05:
        kind = "脸部特写"
    elif ratio >= 0.75:
        kind = "近景"
    elif ratio >= 0.55:
        kind = "中近景"
    elif ratio >= 0.35:
        kind = "中景"
    elif ratio >= 0.18:
        kind = "中全景"
    else:
        kind = "全景"
    return FramingEstimate(
        kind=kind,
        confidence=confidence,
        person_ratio=round(ratio, 3),
        note=f"人体框高度占画面 {ratio:.2f}",
    )


def blocking_from_boxes(
    boxes: Sequence[PersonBox],
) -> tuple[tuple[BlockingEstimate, ...], int, float]:
    """人体框 → 站位 + 人数。

    返回 ``(站位列表, 出镜人数, 置信度)``。只有**前景人物**进站位列表 ——
    背景路人参与不了"谁左谁右"的空间关系，混进来只会制造噪声。
    """
    if not boxes:
        return (), 0, 0.0

    foreground = [b for b in boxes if b.height_ratio >= BACKGROUND_PERSON_HEIGHT]
    confidence = compute_confidence(
        ConfidenceBasis.CV, consistency="single", evidence="timecode_and_frame"
    )
    estimates = tuple(
        BlockingEstimate(
            subject=f"人物{i}",
            side=box.side,
            x_ratio=round(box.center_x, 3),
            confidence=round(confidence * min(1.0, box.confidence / 0.7), 4),
        )
        for i, box in enumerate(foreground, start=1)
    )
    return estimates, len(foreground), confidence


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
        note="CV 景别需要人体框占比，当前环境缺人体检测实现；请接入 ultralytics 后启用",
    )


def analyze_blocking_unavailable() -> tuple[tuple[BlockingEstimate, ...], int, float]:
    """站位与人物计数：同上，显式留空。"""
    return (), 0, 0.0
