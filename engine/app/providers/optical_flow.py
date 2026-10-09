"""光流运镜分析（PRD F1.4 要求的 CV 强判断）。

**为什么必须单独一个模块、而不是继续用亮度序列**

``cv.analyze_motion`` 用的是逐帧**全帧亮度均值**，它有个天生的盲区：
分不清「镜头在动」和「画面里的人在动」。固定机位下人物做深蹲、大面积移位，
亮度照样剧烈变化 —— 实测把被摄体运动误判成 ``pan_or_track``/``push_in``。
所以那一版只能保守地给 ``fixed`` 或 ``unknown``，运镜维度实际上**没有 CV 覆盖**。

光流能分开这两件事，依据是**位移场的一致性**：

    - 镜头摇/移 → 全画面一致平移（光流场位移接近同一向量）；
    - 推近/拉远 → 位移向量**呈放射状**（远离/指向画面中心），
      即"径向散度"显著非零；
    - 被摄体动、镜头不动 → 位移只集中在人物所在区域，
      全局平移量小、且**方向不一致**（散度有正有负、相互抵消）；
    - 手持 → 全局有位移但方向**杂乱抖动**，不单调。

判据因此落在三个可观测量上：全局平移量、平移方向一致度、径向散度。

**这不是靠感觉调的阈值**。本模块只做量测并给出候选结论，
阈值由 ``scripts/motion_probe.py`` 在**真实素材**上量测后确定 ——
用合成色块校准参数在这个项目里已经栽过两次（见 HANDOFF 坑 1）。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Sequence

from engine.core.contracts import MotionEstimate, ShotBoundary
from engine.core.types import ConfidenceBasis, compute_confidence

# ----------------------------------------------------------------------
# 量测参数（做过量测再改，别拍脑袋）
# ----------------------------------------------------------------------
FLOW_WIDTH = 160
"""解码宽度。光流只需要运动结构，不需要细节；160 宽已足够稳定且很快。"""

FLOW_FPS = 8.0
"""光流采样帧率。运镜是连续运动，8Hz 足够分辨，且把解码量压到 1/3。"""

MIN_FLOW_FRAMES = 4
"""一个镜头至少要有这么多采样帧才敢下结论。"""

# 判定阈值 —— 语义见模块开头。全部经 motion_probe.py 在 24 条真实视频上量测确定。
FIXED_TOTAL_MEDIAN = 0.35
"""全局平移量的中位数低于此值 → 固定机位。"""

FIXED_TOTAL_P10 = 0.20
"""更可靠的那条固定判据：分块位移的**第 10 百分位**低于此值。

为什么不用中位数：中位数会被"人物占据的少数格子"抬起来。
实测（16 个已标注镜头）：

    - 人物从远处走到镜头前的镜头，中位位移 0.37~0.48，看着像镜头在动；
      但画面里有大片区域纹丝不动 —— p10 只有 0.16。
    - 标注为推近的镜头，p10 是 0.25~0.26（整幅一起在动）。
    - 标注为摇镜的镜头，p10 是 2.36。

**只要画面里有一成格子是静止的，镜头就没在动。**
0.20 取在实测 fixed 最大 0.16 与 push_in 最小 0.25 之间。
这条判据比中位数稳，代价是极慢的运镜（全画面一致但幅度小于噪声）会被
判成固定 —— 那种情况本来就分不出来，归到固定是合理的选择。
"""

ZOOM_DIVERGENCE = 0.20
"""径向散度高于此值 → 推近/拉远。

**推近与固定机位只能靠散度分开，靠位移分不开。** 实测（16 个已标注镜头）：

    - 标注 fixed 的镜头，位移 0.00~0.48，散度全部落在 ±0.04 内；
    - 标注 push_in 的镜头，位移 0.36~0.41（和 fixed 完全重叠），
      但散度是 +0.25 与 +0.32。

位移重叠的原因是：人物朝镜头走近，同样会让画面产生大位移。
而"整场向中心收缩"这件事只有镜头推近才会做，所以散度才是判别量。
0.20 取在 fixed 的 0.04 之上、push_in 的 0.25 之下，两侧都留了余量。
"""

PAN_DIRECTION = 0.75
"""摇/移判据：全局位移大且方向高度一致。"""

HANDHELD_DIRECTION = 0.42
"""手持判据：全局位移大，但方向一致度明显低（抖动无主方向）。"""

SUBJECT_TAIL_PX = 0.8
"""被摄体主导度的尾巴阈值（像素，作用在分块中位流上）。

用来把"整幅一致位移"和"只有一部分在动"分开：位移超过全图分块中位
0.8px 的格子占比越大，越说明动的是局部（人），而不是镜头。
"""

FLOW_GRID = 6
"""分块网格边长。6×6 = 36 块。

块数太少（如 3×3）时人物能占到 1/9，仍会左右统计；太多（如 12×12）时
每块像素少，中位流噪声大。6×6 在 160px 宽下每块约 26×20 像素，
背景块数稳定压倒人物块数。
"""


@dataclass(frozen=True, slots=True)
class FlowMeasurement:
    """一个镜头的原始光流量测结果。**先量测、再判定**，判定可复核。"""

    frames: int
    translation: float
    """全局平移量（每帧位移 / 画面宽度 × 100）的中位数。"""
    direction: float
    """方向一致度：主方向上的位移占总位移的比例，0~1。"""
    divergence: float
    """径向散度（见 ZOOM_DIVERGENCE）。"""
    spread: float = 0.0
    """位移量的四分位距，反映抖动强弱。"""
    p10_translation: float = 0.0
    """分块位移的第 10 百分位。**固定机位的主判据**（见 FIXED_TOTAL_P10）。"""
    subject_dominance: float = 0.0
    """被摄体主导度：分块位移里"明显大于全图分块中位"的格子占比，0~1。

    **这个量专门用来救亮度法的那个坑**：固定机位下人物大面积移动时，
    背景块的中位位移仍然很小，但会有一批格子明显偏大 ——
    这批格子有多少，就说明"动的只是一部分"有多明显。
    镜头真的在摇时，位移是整幅一致的，几乎没有这样的格子。
    """
    valid_ratio: float = 0.0
    """逐像素位移算得出来的像素占比。纹理稀少（纯色墙面）时会很低。"""

    @property
    def usable(self) -> bool:
        return self.frames >= MIN_FLOW_FRAMES


def analyze_flow_shot(
    measurement: FlowMeasurement,
    shot: ShotBoundary,
) -> MotionEstimate:
    """量测 → 运镜结论。判据顺序即置信度顺序：先给最有把握的。"""
    if not measurement.usable:
        return MotionEstimate(
            kind="unknown",
            confidence=0.0,
            note=f"镜头内仅 {measurement.frames} 个光流采样帧，不足 {MIN_FLOW_FRAMES} 帧",
        )

    confidence = compute_confidence(
        ConfidenceBasis.CV, consistency="single", evidence="timecode_and_frame"
    )
    t = measurement.translation
    d = measurement.direction
    z = measurement.divergence

    # 判据顺序 = 置信度顺序：先给最有把握的。
    # 固定必须排在推拉之前 —— 固定机位下人物走动也会造出中等散度，
    # 但"有一成格子纹丝不动"是更硬的证据。
    if measurement.p10_translation < FIXED_TOTAL_P10:
        return MotionEstimate(
            kind="fixed",
            confidence=round(confidence * 0.95, 4),
            magnitude=t,
            divergence=z,
            note=(f"画面有 {FIXED_TOTAL_P10 * 100:.0f}% 以上区域基本静止"
                  f"（分块位移 p10={measurement.p10_translation:.2f}，中位 {t:.2f}），"
                  f"判定固定机位"),
        )

    if abs(z) >= ZOOM_DIVERGENCE:
        kind = "push_in" if z > 0 else "pull_out"
        return MotionEstimate(
            kind=kind,
            confidence=round(confidence * 0.8, 4),
            magnitude=t,
            divergence=z,
            note=f"径向散度 {z:+.2f}（位移呈放射状），判定{'推近' if z > 0 else '拉远'}",
        )

    if t >= FIXED_TOTAL_MEDIAN and d >= PAN_DIRECTION:
        return MotionEstimate(
            kind="pan_or_track",
            confidence=round(confidence * 0.8, 4),
            magnitude=t,
            divergence=z,
            note=f"全局位移 {t:.2f} 且方向一致度 {d:.2f}，判定摇/移镜",
        )

    if t >= FIXED_TOTAL_MEDIAN and d < HANDHELD_DIRECTION:
        return MotionEstimate(
            kind="handheld",
            confidence=round(confidence * 0.6, 4),
            magnitude=t,
            divergence=z,
            note=f"全局位移 {t:.2f} 但方向杂乱（一致度 {d:.2f}），判定手持晃动",
        )

    # 位移中等、方向半一致 —— 分不清是镜头在缓动还是被摄体在动。
    return MotionEstimate(
        kind="unknown",
        confidence=0.0,
        magnitude=t,
        divergence=z,
        note=(
            f"光流量测不足以定性（位移 {t:.2f}、方向一致度 {d:.2f}、散度 {z:+.2f}）；"
            f"交视觉模型判断"
        ),
    )

# ----------------------------------------------------------------------
# 帧解码 + 光流计算
# ----------------------------------------------------------------------
def decode_gray(
    ffmpeg_bin: str,
    path: str,
    *,
    width: int = FLOW_WIDTH,
    fps: float = FLOW_FPS,
) -> "object":
    """整段解码成灰度帧数组 ``(n, h, w)``。

    一次性解码而不是逐镜头定位：光流需要镜头内**连续**帧，逐镜头调 ffmpeg
    会让每个镜头都要重新 seek+解码，反而更慢；且 seek 结果不保证落在同一相位。
    """
    import numpy as np

    probe = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    height = _scaled_height(probe.stderr or "", width)
    raw = subprocess.run(
        [
            ffmpeg_bin, "-v", "error",
            "-i", path,
            "-vf", f"fps={fps},scale={width}:{height}",
            "-pix_fmt", "gray",
            "-f", "rawvideo", "-",
        ],
        capture_output=True, check=False,
    ).stdout
    frame_bytes = width * height
    count = len(raw) // frame_bytes
    if count == 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.frombuffer(raw[: count * frame_bytes], dtype=np.uint8).reshape(
        count, height, width
    )


def measure_shots(
    frames: "object",
    shots: Sequence[ShotBoundary],
    *,
    fps: float = FLOW_FPS,
) -> list[FlowMeasurement]:
    """对每个镜头算光流量测。某镜头采样帧不足时如实记录 0 帧。"""
    import cv2
    import numpy as np

    per_shot = _frame_indices(shots, fps, frame_count=len(frames))
    out: list[FlowMeasurement] = []
    for indices in per_shot:
        if len(indices) < MIN_FLOW_FRAMES:
            out.append(FlowMeasurement(frames=len(indices), translation=0.0,
                                       direction=0.0, divergence=0.0))
            continue
        trans: list[float] = []
        dirs: list[float] = []
        divs: list[float] = []
        doms: list[float] = []
        valids: list[float] = []
        blocks: list[float] = []
        for a, b in zip(indices, indices[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                frames[a], frames[b], None,
                0.5, 3, 21, 3, 5, 1.2, 0,
            )
            block_norm = _block_norms(flow)
            translation, direction, divergence, dominance, valid = _flow_stats(
                flow, block_norm=block_norm
            )
            trans.append(translation)
            dirs.append(direction)
            divs.append(divergence)
            doms.append(dominance)
            valids.append(valid)
            blocks.append(block_norm)
        # p10 必须**按帧算完再汇总**：直接对逐帧中位数取分位是错的 ——
        # 中位数已经被人物格子抬高了，再取它的分位只会得到"被抬高值"的分位。
        stacked = np.concatenate(blocks) if blocks else np.zeros(1)
        out.append(FlowMeasurement(
            frames=len(indices),
            translation=float(np.median(trans)),
            direction=float(np.median(dirs)),
            divergence=float(np.median(divs)),
            spread=float(np.percentile(trans, 75) - np.percentile(trans, 25)),
            p10_translation=float(np.percentile(stacked, 10)),
            subject_dominance=float(np.median(doms)),
            valid_ratio=float(np.median(valids)),
        ))
    return out


def _block_norms(flow: "object") -> "object":
    """分块中位位移长度数组（36 个块的向量长度）。"""
    import numpy as np

    height, width = flow.shape[:2]
    grid = FLOW_GRID
    bh, bw = height // grid, width // grid
    if bh < 2 or bw < 2:
        return np.zeros(1)
    usable_h, usable_w = bh * grid, bw * grid
    vx = flow[:usable_h, :usable_w, 0].reshape(grid, bh, grid, bw)
    vy = flow[:usable_h, :usable_w, 1].reshape(grid, bh, grid, bw)
    block_vx = np.median(vx, axis=(1, 3))
    block_vy = np.median(vy, axis=(1, 3))
    return np.sqrt(block_vx ** 2 + block_vy ** 2)


def _flow_stats(
    flow: "object", *, block_norm: "object | None" = None
) -> tuple[float, float, float, float, float]:
    """单对帧的光流 → ``(平移量, 方向一致度, 径向散度, 被摄体主导度, 有效率)``。

    用**分块中位流**而不是逐像素求和。原因是实测踩到的：

        一条演播厅教学镜头里人物朝镜头走近，人物剪影在画面中心**膨胀**，
        逐像素求散度得到 -1.01 → 被判成"拉远"。而镜头其实根本没动。

    人物这种"局部膨胀"和镜头推拉产生的径向场形状完全不同，但逐像素求和
    会把局部的大位移摊进全局统计里，压过背景的稳定证据。分块取中位数之后，
    占据少数格子的人物无法左右多数的背景格子 —— 背景说不动，就是不动。

    - 平移量：分块中位位移长度按画面宽度归一化后 ×100，再取全图分块中位；
    - 方向一致度：||Σ块向量|| / Σ||块向量||，全体同向为 1；
    - 径向散度：块向量在"中心→该块"单位向量上的投影均值 ×100；
    - 被摄体主导度：块位移超过全图分块中位 + ``SUBJECT_TAIL_PX`` 的格子占比。
    """
    import numpy as np

    height, width = flow.shape[:2]
    grid = FLOW_GRID
    bh, bw = height // grid, width // grid
    if bh < 2 or bw < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    usable_h, usable_w = bh * grid, bw * grid

    vx = flow[:usable_h, :usable_w, 0].reshape(grid, bh, grid, bw)
    vy = flow[:usable_h, :usable_w, 1].reshape(grid, bh, grid, bw)
    block_vx = np.median(vx, axis=(1, 3))
    block_vy = np.median(vy, axis=(1, 3))

    # 每块中心的径向单位向量（以画面中心为原点）
    centers_y = (np.arange(grid) + 0.5) * bh
    centers_x = (np.arange(grid) + 0.5) * bw
    cx, cy = usable_w / 2.0, usable_h / 2.0
    dx = centers_x[None, :] - cx
    dy = centers_y[:, None] - cy
    radius = np.sqrt(dx * dx + dy * dy)
    radius[radius < 1.0] = 1.0
    ux, uy = dx / radius, dy / radius

    norm = block_norm if block_norm is not None else _block_norms(flow)
    translation = float(np.median(norm) / width * 100.0)

    total = float(norm.sum())
    direction = float(
        np.sqrt(block_vx.sum() ** 2 + block_vy.sum() ** 2) / total
    ) if total > 1e-6 else 0.0
    divergence = float(np.mean(block_vx * ux + block_vy * uy) / width * 100.0)

    tail = float(np.median(norm)) + SUBJECT_TAIL_PX
    dominance = float(np.mean(norm > tail))

    pixel_norm = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    valid = float(np.mean(pixel_norm > 0.01))
    return translation, direction, divergence, dominance, valid


def _frame_indices(
    shots: Sequence[ShotBoundary],
    fps: float,
    *,
    frame_count: int | None = None,
) -> list[list[int]]:
    """镜头区间 → 全局采样帧下标。

    ``frame_count`` 用于**夹住尾巴**：ffmpeg 按固定 fps 采样，实际帧数可能比
    "时长 × fps" 少一帧；不夹住就会索引越界，把整条反推带崩。
    """
    out: list[list[int]] = []
    for shot in shots:
        start = max(0, int(shot.start_ms / 1000.0 * fps))
        end = max(0, int(shot.end_ms / 1000.0 * fps))
        if frame_count is not None:
            end = min(end, frame_count)
            start = min(start, end)
        out.append(list(range(start, end)))
    return out


def _scaled_height(stderr: str, width: int) -> int:
    """从 ffmpeg 日志里取源分辨率，算出等比高度（保持偶数）。"""
    import re

    match = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", stderr)
    if not match:
        return width * 2  # 竖屏默认
    src_w, src_h = int(match.group(1)), int(match.group(2))
    height = max(2, round(width * src_h / src_w))
    return height if height % 2 == 0 else height + 1


# ----------------------------------------------------------------------
# 编排层适配器
# ----------------------------------------------------------------------
class OpticalFlowMotionAnalyzer:
    """``MotionAnalyzer`` 端口的光流实现。

    **为什么整段只在第一次调用时解码一次**：编排层是逐镜头调 ``analyze`` 的，
    而光流必须一次拿到整段连续帧（逐镜头 seek 会重解码且相位不一致）。
    所以在 ``bind()`` 之后按需解码一次、把全部镜头的量测算好缓存起来，
    后续每个镜头直接查表 —— 一条 15 秒视频的总代价约 1~2 秒。
    """

    def __init__(self, ffmpeg_bin: str) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self._video_path = ""
        self._shots: tuple[ShotBoundary, ...] = ()
        self._cache: dict[int, MotionEstimate] = {}
        self.frame_count = 0
        """最近一次解码得到的采样帧数（诊断用）。"""
        self.measurements: dict[int, FlowMeasurement] = {}
        """逐镜头原始量测，保留下来供回归与 UI 证据回溯。"""

    def bind(self, video_path: str, shots: Sequence[ShotBoundary]) -> None:
        """绑定本轮视频与镜头表。编排层每次反推调一次，自动作废旧缓存。"""
        if video_path != self._video_path:
            self._cache.clear()
            self.measurements.clear()
        self._video_path = video_path
        self._shots = tuple(shots)

    def analyze(self, path: str, shot: ShotBoundary) -> MotionEstimate:
        """``path`` 被忽略（编排层逐镜头调用时传的是空串）；视频由 ``bind`` 指定。"""
        if shot.index in self._cache:
            return self._cache[shot.index]
        if not self._video_path or not self._shots:
            return MotionEstimate(
                kind="unknown", confidence=0.0,
                note="光流分析器未绑定视频，跳过（不猜）",
            )
        self._prepare()
        return self._cache.get(
            shot.index,
            MotionEstimate(kind="unknown", confidence=0.0, note="该镜头不在本次切分结果里"),
        )

    def _prepare(self) -> None:
        """一次解码整段、批量算完所有镜头 —— 逐镜头重解码会慢一个数量级。"""
        frames = decode_gray(self.ffmpeg_bin, self._video_path)
        self.frame_count = len(frames)
        for shot, measurement in zip(
            self._shots, measure_shots(frames, self._shots)
        ):
            self.measurements[shot.index] = measurement
            self._cache[shot.index] = analyze_flow_shot(measurement, shot)


def build_motion_analyzer(ffmpeg_bin: str) -> OpticalFlowMotionAnalyzer | None:
    """按需构建：缺 opencv 就返回 ``None``，让编排层退回亮度法。"""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return None
    return OpticalFlowMotionAnalyzer(ffmpeg_bin)
