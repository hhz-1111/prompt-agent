"""镜头切分：PySceneDetect 适配器。

**为什么换成这个库**：自研实现用「全帧 RGB 均值色差」判定切点，
而全帧均值只有**一个数**。两个镜头整体色调相近但内容完全不同时
（同一间屋换了个人、同一货架换了商品），这个数几乎不变 —— 必然漏切。

实测对比（24 条真实爆款视频）：

    自研（全帧均值）      平均 1.21 个镜头，20/24 只切出 1 个
    PySceneDetect 内容   平均 1.71 个镜头，17/24 只切出 1 个
    PySceneDetect 自适应  平均 1.96 个镜头，15/24 只切出 1 个

最典型的一条：`爆款视频AI复刻刘畊宏` 9 秒内有 5 个明显切镜，
自研实现检出 **0** 个。

合成视频逐点校验（4 个近同色灰镜头，真相切点 [2.0, 4.0, 6.0]）：

    ContentDetector threshold=27   → []            默认阈值，完全漏检
    ContentDetector threshold=3    → [2.0, 4.0, 6.0]
    AdaptiveDetector 低阈值        → [2.0, 4.0, 6.0]   ✓ 全对

所以默认走自适应检测。PySceneDetect 的核心是**逐像素 HSV 三通道加权差分**
（delta_hue / delta_sat / delta_lum），能捕捉"整体色调不变但画面内容变了"。
这正是自研实现缺的那一维。
"""

from __future__ import annotations

from typing import Callable, Sequence

from engine.core.contracts import ShotBoundary, VideoInfo

from ..config import Settings

ProgressFn = Callable[[str, float], None]

# 检测器参数。**在真实素材上校准，不是拍脑袋定的**。
#
# 校准轨迹（值得记下来，因为第一版栽过跟头）：
#
#   自适应 1.0 / min_content 2.0 / min_scene 0.3s
#       → 合成低对比度视频三个切点全对，但真实视频被切碎成 22~30 段。
#         "镜头"只有 0.3~0.5 秒 —— 那是把正常镜头运动当成了切镜。
#   自适应 1.5 / min_content 4.0 / min_scene 0.6s  → 真实视频仍 14 段，偏碎
#   自适应 2.0 / min_content 8.0 / min_scene 1.2s  → 刘畊宏 5、户外教学 3、菜市场 1 ✓
#   自适应 3.0 / min_content 15.0 / min_scene 1.5s → 刘畊宏 3，开始漏切
#
# 教训：**合成视频上校准出来的激进参数，搬到真实内容上一定过度敏感**。
# 真实镜头有运动、有曝光变化，而合成色块没有。
ADAPTIVE_THRESHOLD = 2.0
MIN_CONTENT_VAL = 8.0
MIN_SCENE_SECONDS = 1.2

# 第二遍：高阈值内容检测，专门抓高置信度的硬切。
# 阈值取 PySceneDetect 的默认值 27，它的最小镜头长度可以放宽些 ——
# 能过 27 阈值的基本都是真硬切。
CONTENT_THRESHOLD = 27.0
CONTENT_MIN_SCENE_SECONDS = 0.8


class SceneDetectShots:
    """``ShotDetector`` 端口的 PySceneDetect 实现。

    用两遍检测取并集：

    - **自适应**（低阈值）：负责灵敏度，抓低对比度切镜；
    - **内容检测**（默认阈值 27）：负责准确度，抓高置信度硬切。

    两遍取并集后按最小镜头长度合并 —— 宁可多给一个切点让用户删，
    也不要漏掉一个真切镜，因为切点是**致命级维度**（节拍会全错位）。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        """PySceneDetect 是否可用。装不上时由调用方退回 ffmpeg 实现。"""
        try:
            import scenedetect  # noqa: F401
        except ImportError:
            return False
        return True

    def detect(
        self,
        path: str,
        info: VideoInfo,
        *,
        on_progress: ProgressFn | None = None,
    ) -> list[ShotBoundary]:
        from scenedetect import AdaptiveDetector, ContentDetector, detect

        fps = info.fps or 25.0

        if on_progress:
            on_progress("镜头切分（自适应）", 0.15)
        adaptive = detect(
            path,
            AdaptiveDetector(
                adaptive_threshold=ADAPTIVE_THRESHOLD,
                min_content_val=MIN_CONTENT_VAL,
                min_scene_len=max(1, int(fps * MIN_SCENE_SECONDS)),
            ),
            show_progress=False,
        )
        cuts_ms = [start.seconds * 1000 for start, _ in adaptive[1:]]

        if on_progress:
            on_progress("镜头切分（硬切校验）", 0.22)
        try:
            content = detect(
                path,
                ContentDetector(
                    threshold=CONTENT_THRESHOLD,
                    min_scene_len=max(1, int(fps * CONTENT_MIN_SCENE_SECONDS)),
                ),
                show_progress=False,
            )
            cuts_ms.extend(start.seconds * 1000 for start, _ in content[1:])
        except Exception:
            # 第二遍失败不影响主结果：自适应那遍已经给出切点了
            pass

        return build_boundaries(
            sorted(set(round(c) for c in cuts_ms)),
            info.duration_sec,
            min_gap_ms=int(MIN_SCENE_SECONDS * 1000 * 0.8),
        )


# 视频末尾常被切出一个几百毫秒的"残留镜头"（最后的转场/黑帧），
# 它装不下任何内容，而且抽帧时 `-ss` 落到实际帧位置之外会让 ffmpeg 报错。
# 实测踩过：一条视频切出 180ms 的尾巴，导致整条反推失败。
MIN_TAIL_MS = 250


def build_boundaries(
    cut_ms: Sequence[int],
    duration_sec: float,
    *,
    min_gap_ms: int = 300,
) -> list[ShotBoundary]:
    """把切点整理成连续的镜头区间。

    做三件事：

    1. 丢弃贴边切点（编码抖动，不是真切镜）；
    2. 合并挨得太近的切点（一段渐变会被判成多次切点）；
    3. **把过短的碎片镜头并入前一个镜头** —— 碎片装不下内容，
       还会让抽帧的 ``-ss`` 落到帧位置之外。
    """
    total_ms = int(round(duration_sec * 1000))
    edges = [0]
    for cut in sorted(cut_ms):
        if cut <= 60 or cut >= total_ms - 60:
            continue
        if edges and cut - edges[-1] < min_gap_ms:
            continue
        edges.append(int(cut))

    # 末尾碎片并入前一镜，而不是单独成一个 0.2 秒的"镜头"
    while len(edges) > 1 and total_ms - edges[-1] < MIN_TAIL_MS:
        edges.pop()
    edges.append(total_ms)

    boundaries: list[ShotBoundary] = []
    for index in range(len(edges) - 1):
        start, end = edges[index], edges[index + 1]
        if end <= start:
            continue
        boundaries.append(ShotBoundary(index=len(boundaries) + 1, start_ms=start, end_ms=end))
    return boundaries
