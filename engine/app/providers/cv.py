"""CV 强判断层（PRD §6.1.2 第一层）。

**能力边界要说清楚，不能假装。**

规范要求这一层用光流算运镜、用人体框占比算景别、用检测框算站位。
本模块的策略是：

1. **能确定性算出来的，认真算** —— 人体框高度占比算景别、框中心算站位、
   去重计数算人数；运镜的正路（光流）在 ``optical_flow.py``，
   本模块的亮度序列只是**装不了 opencv 时的退路**；
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
    """运镜（退路）：逐帧亮度序列能区分"稳定"与"持续变化"。

    这个退路**分不清镜头动与被摄体动**，只能保守地给 fixed/unknown，
    所以它不算运镜维度真的被 CV 覆盖了。
    """
    motion_optical_flow: bool = False
    """运镜（正路）：光流能分开"镜头在动"与"画面里的人在动"。需 opencv。"""
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
        if not self.motion_optical_flow:
            gaps.append("运镜（缺光流，仅亮度退路，分不清镜头动与被摄体动）")
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
            "motion_optical_flow": self.motion_optical_flow,
            "framing": self.framing,
            "blocking": self.blocking,
            "people_count": self.people_count,
            "person_detector": self.person_detector,
            "missing": list(self.missing),
        }


def detect_capabilities() -> CvCapabilities:
    """探测本机可用的 CV 能力。

    人体检测与光流都是**可选增强**：装了就用，没装就退回退路，
    并且 ``missing`` 会如实报出来（``false`` 和"悄悄降级"是两回事）。
    """
    name = _probe_person_detector()
    return CvCapabilities(
        motion_from_luma=True,
        motion_optical_flow=_probe_optical_flow(),
        framing=bool(name),
        blocking=bool(name),
        people_count=bool(name),
        person_detector=name,
    )


def _probe_optical_flow() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


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

**这个阈值分不开"路人"与"远景真人"**：`IP演播厅带练-丹` 里远机位的
主要人物框高只有 0.23，会被当成背景。用 `scripts/verify_person_boxes.py`
（看该区域的帧间运动）验证过那几个框**确实在动**，是真人 —— 所以
远景镜头下这个小尺寸人物会被漏掉，是当前人体检测的**已知边界**。
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
        """按中心 x 判定左/中/右。判据与 ``side_of`` 共用一处，不重复定义。"""
        return side_of(self.center_x)

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


def composition_side(
    boxes: Sequence[PersonBox],
) -> tuple[str, float, int]:
    """画面**构图**偏向哪一侧。返回 ``(偏侧, 人物 x 跨度中点, 前景人数)``。

    **为什么不能取"最主要那个人"的左右**：双人镜头里两个人分别在 x=0.30 与
    0.69，取最靠左的那个会判成 ``left`` —— 可人工标注认为这种"一左一右"
    就是构图居中。所以取最左与最右的连线中点：一左一右 → 中点回到 0.5 附近。

    前景人数按 ``BACKGROUND_PERSON_HEIGHT`` 过滤（只比"这是个人"更严一点），
    这样既能滤掉背景路人，又不会把双人镜头里稍矮的那个当路人丢掉 ——
    实测用"相对最高者的比例"过滤会误伤（0.84 与 0.50 的两个人本来是居中，
    滤掉一个就变偏左了）。
    """
    foreground = [b for b in boxes if b.height_ratio >= BACKGROUND_PERSON_HEIGHT]
    if not foreground:
        return "", 0.0, 0
    left = min(b.center_x for b in foreground)
    right = max(b.center_x for b in foreground)
    midpoint = (left + right) / 2.0
    return side_of(midpoint), midpoint, len(foreground)


def side_of(x_ratio: float) -> str:
    """画面横向位置 → 左/中/右。中间带取 ±``SIDE_CENTER_BAND``。"""
    if x_ratio < 0.5 - SIDE_CENTER_BAND:
        return "left"
    if x_ratio > 0.5 + SIDE_CENTER_BAND:
        return "right"
    return "center"


SIDE_CENTER_BAND = 0.10
"""判定「居中」的容差带宽度（即 0.40~0.60 算中间）。

取自实测：单人居中镜头的框中心落在 0.50~0.54，双人镜头跨度中点在 0.42~0.51；
明确偏侧的镜头在 0.16~0.36。这条带把两者分开，两侧都留有余量。
"""

# ----------------------------------------------------------------------
# 景别分档阈值：人体框高度占画面的比例
# ----------------------------------------------------------------------
# 全部来自 16 个已标注镜头的穷举搜索（scripts/fit_framing_bands.py），
# 切点放在**相邻标注档位之间的空档**里，而不是贴着样本切。
# 标注样本量小，这些数字是当前最佳估计而非常量 —— 标注多了要重算。
FRAMING_MEDIUM_WIDE = 0.52
"""全景 / 中全景 分界。

标注"全景"最大 0.57、"中景"最小 0.74 —— 切点取 0.52 是为了让
0.57 那条落进中全景（它离"中景"的 0.74 更近），同时保住 0.47 以下的全景。
"""
FRAMING_MEDIUM = 0.66
"""中全景 / 中景 分界。放在 0.57~0.74 那段**完全没有样本**的空档中间。"""
FRAMING_MEDIUM_CLOSE = 0.88
"""中景 / 中近景 分界。标注在这附近自相矛盾（0.87 同时被标成中景与近景），
所以这一小段必然有损耗。"""
FRAMING_CLOSE = 0.90
"""中近景 / 近景 分界。

取 0.90 而不是更高：实测"近景"的最小样本是 0.92，而 0.87 被标注者自己
标成了两档。中近景因此只有 0.88~0.90 这一段 —— 窄，但那是标注数据的
真实分辨率决定的，不是刻意压窄。"""
FRAMING_FACE = 1.05
"""近景 / 脸部特写 分界。人体框高度超过画面高度即视为只剩脸部特写。

**这一档在标注里没有任何样本**（实测最大 0.92），属于按常识外推，
不是量测结论。
"""


def framing_from_person(box: PersonBox) -> FramingEstimate:
    """人体框 → 景别。

    用**高度占比**而不是面积占比：竖屏 9:16 下人体是细长的，
    面积占比会被宽度稀释，而高度占比直接对应"人物占画面多大"。

    **分档阈值来自人工标注的实测，不是拍的**（`scripts/fit_framing_bands.py`
    在 16 个已标注镜头上穷举切点）：

        标注"全景"的样本  高占比 0.32~0.57
        标注"中景"的样本  高占比 0.74~0.87
        标注"近景"的样本  高占比 0.87~0.92

    全景与中景之间有一大段**空的**（0.57~0.74），切点放这段中间最稳；
    中景与近景的标注本身就有重叠（0.87 同时被两位镜头标成两档），
    切点只能放在这团重叠里，所以这两档天生分不干净。

    **样本量只有 16 个，别把这里的数字当成定理。**标注多了要回来重算。
    """
    ratio = box.height_ratio
    confidence = compute_confidence(
        ConfidenceBasis.CV, consistency="single", evidence="timecode_and_frame"
    )
    if ratio >= FRAMING_FACE:
        kind = "脸部特写"
    elif ratio >= FRAMING_CLOSE:
        kind = "近景"
    elif ratio >= FRAMING_MEDIUM_CLOSE:
        kind = "中近景"
    elif ratio >= FRAMING_MEDIUM:
        kind = "中景"
    elif ratio >= FRAMING_MEDIUM_WIDE:
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

    .. deprecated:: 光流已可用
       这是**退路**，不是正路。装不了 opencv 时才用它。
       正路是 ``optical_flow.OpticalFlowMotionAnalyzer`` —— 它能分开
       "镜头在动"与"画面里的人在动"，而亮度序列做不到（见下）。

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
