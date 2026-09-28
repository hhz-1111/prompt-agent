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
# **能力边界要说透，这里是本模块最弱的一环。**
#
# 全帧亮度均值**无法区分「镜头在动」和「画面里的人在动」**。
# 实测踩过：一条固定机位教学视频里人物做深蹲、大面积移位，
# 亮度自然变化，我的算法把被摄体运动误判成了镜头运动 ——
# 判出 5 处 pan_or_track/push_in/handheld，而模型（正确地）判「固定」。
#
# 更早还犯过一次同样的错：阈值按合成色块校准到 0.015，
# 真实内容永远超不过它。**合成素材上校准的参数套真实内容必然失准。**
#
# 所以现在的策略是**保守**：
#   - 只有变化极小才敢判 ``fixed``（这是最有把握的一类）；
#   - 明显单调且幅度大才判 ``push_in``；
#   - 其余一律 ``unknown`` —— 交给视觉模型判断，并如实标注能力边界。
#
# 正确做法是**光流分析**（OpenCV 的 ``calcOpticalFlowFarneback`` 或
# ``DISOpticalFlow``）：光流场能区分"整场一致位移"（镜头摇）与
# "局部不一致位移"（被摄体动）。那需要安装 opencv-python。
# 在装上之前，宁可不给结论，也不给错结论（红线 7 的同一精神）。
MOTION_STABLE_SPAN = 4.0
"""全片亮度跨度低于此值 → 判定固定机位。

实测：真实固定机位内容跨度 0.6~2.9（人物走动导致）；合成纯色块接近 0。
阈值取 4.0 留出余量，宁漏判不错判。
"""
MOTION_STABLE_MAD = 0.6
"""帧间亮度平均绝对差低于此值 → 判定固定机位。"""
MOTION_PUSH_SPAN = 25.0
"""亮度跨度超过此值且高度单调 → 才敢判推近/拉远。

真实推近会让画面整体亮度大幅单调变化，量级远高于人物走动。
"""
MOTION_PUSH_DIRECTION = 0.80
"""判定推近所需的单调性（方向一致度）。"""


def analyze_motion(
    frames: Sequence[FrameMean],
    shot: ShotBoundary,
) -> MotionEstimate:
    """用镜头区间内的逐帧亮度序列判断运镜。

    **只在有把握时才给结论**，其余返回 ``unknown``：

    - ``fixed``：变化极小，最有把握；
    - ``push_in``：跨度大且高度单调；
    - 其余：``unknown``，由视觉模型判断。

    不做 ``handheld`` / ``pan_or_track`` 的判定 —— 亮度序列区分不了它们，
    硬判就是编数据。
    """
    window = [f for f in frames if shot.start_ms <= int(f.pts_time * 1000) <= shot.end_ms]
    if len(window) < 8:
        return MotionEstimate(
            kind="unknown",
            confidence=0.0,
            note=f"镜头内仅 {len(window)} 帧，不足 8 帧，不做运动估计",
        )

    lumas = [f.luma for f in window]
    deltas = [b - a for a, b in zip(lumas, lumas[1:])]
    mean_abs = statistics.fmean(abs(d) for d in deltas)
    span = max(lumas) - min(lumas)
    positives = sum(1 for d in deltas if d > 0)
    direction_ratio = max(positives, len(deltas) - positives) / len(deltas)

    confidence = compute_confidence(
        ConfidenceBasis.CV, consistency="single", evidence="timecode_and_frame"
    )

    if span < MOTION_STABLE_SPAN and mean_abs < MOTION_STABLE_MAD:
        return MotionEstimate(
            kind="fixed",
            confidence=round(confidence * 0.9, 4),
            magnitude=mean_abs,
            note=f"帧间亮度变化 {mean_abs:.3f}、跨度 {span:.2f}，判定固定机位",
        )

    if span >= MOTION_PUSH_SPAN and direction_ratio >= MOTION_PUSH_DIRECTION:
        return MotionEstimate(
            kind="push_in",
            confidence=round(confidence * 0.7, 4),
            magnitude=mean_abs,
            note=f"亮度大幅单调变化（跨度 {span:.1f}、方向一致度 {direction_ratio:.2f}），"
                 f"判定推近/拉远",
        )

    # 其余交给视觉模型。亮度序列分不清"镜头动"与"被摄体动"，
    # 硬判会得出错误的运镜档位，而运镜是**高影响级**维度。
    return MotionEstimate(
        kind="unknown",
        confidence=0.0,
        magnitude=mean_abs,
        note=(
            f"亮度序列无法区分镜头运动与被摄体运动"
            f"（跨度 {span:.2f}、方向一致度 {direction_ratio:.2f}）；"
            f"改由视觉模型判断。装 opencv-python 可启用光流分析解决这一限制"
        ),
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
