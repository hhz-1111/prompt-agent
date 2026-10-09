"""光流运镜分析的单元测试。

**这里只用合成数据验证「算法逻辑对不对」**，绝不在这里校准阈值 ——
阈值在真实素材上量测（``scripts/motion_probe.py``）。这条分界是 HANDOFF 坑 1
的教训：合成色块上校准出来的参数，搬到真实视频必然失准。
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.app.providers import optical_flow as of
from engine.core.contracts import ShotBoundary

SHOT = ShotBoundary(index=1, start_ms=0, end_ms=8000)


def _translate(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """整幅平移 —— 模拟摇镜/移镜（全局一致位移）。"""
    return np.roll(np.roll(image, dx, axis=1), dy, axis=0)


def _texture(height: int = 120, width: int = 160, seed: int = 7) -> np.ndarray:
    """有纹理的底图。纯色图的 Farneback 光流退化不出有效向量。"""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    # 放大成块状纹理，让光流有稳定的可追踪结构
    return np.kron(base[: height // 8, : width // 8], np.ones((8, 8), dtype=np.uint8))


# ----------------------------------------------------------------------
# _flow_stats：三个可观测量本身对不对
# ----------------------------------------------------------------------
def test_flow_stats_pure_translation_has_no_divergence() -> None:
    """纯平移：位移大、方向一致度高、径向散度接近 0。"""
    import cv2

    image = _texture()
    shifted = _translate(image, 6, 0)
    flow = cv2.calcOpticalFlowFarneback(image, shifted, None, 0.5, 3, 21, 3, 5, 1.2, 0)

    translation, direction, divergence, _, _ = of._flow_stats(flow)

    assert translation > 1.0, f"平移量应当显著，实得 {translation}"
    assert direction > 0.9, f"整幅同向移动，方向一致度应接近 1，实得 {direction}"
    assert abs(divergence) < 0.5, f"纯平移不该有径向散度，实得 {divergence}"


def test_flow_stats_zoom_has_divergence() -> None:
    """推近：位移呈放射状 —— 径向散度显著为正。"""
    import cv2

    image = _texture()
    height, width = image.shape
    # 以中心为原点放大 6%：等价于镜头推近
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1.06)
    zoomed = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    flow = cv2.calcOpticalFlowFarneback(image, zoomed, None, 0.5, 3, 21, 3, 5, 1.2, 0)

    _, _, divergence, _, _ = of._flow_stats(flow)

    assert divergence > 0.5, f"推近应当产生正散度，实得 {divergence}"


def test_flow_stats_local_motion_keeps_median_translation_low() -> None:
    """只有一块区域在动（被摄体动、镜头不动）：全局平移量仍在噪声量级。

    这条是"亮度法为什么会误判"的量化证据：亮度均值会剧变，
    但分块中位位移几乎不动 —— 因为动的只是少数格子。
    """
    import cv2

    height, width = 120, 160
    background = np.full((height, width), 30, dtype=np.uint8)
    # 一个高对比度方块（人物替身），在两帧之间横移
    first = background.copy()
    first[40:100, 30:70] = 230
    second = background.copy()
    second[40:100, 80:120] = 230
    flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 21, 3, 5, 1.2, 0)

    translation, _, divergence, _, _ = of._flow_stats(flow)

    assert translation < 1.0, f"局部运动不该抬高中位位移，实得 {translation}"
    assert abs(divergence) < 0.5, f"局部平移不产生径向散度，实得 {divergence}"


def test_advancing_subject_is_not_mistaken_for_zoom_out() -> None:
    """**实测踩过的坑**：人物朝镜头走近，会让逐像素径向散度变成负值。

    真实案例：一条演播厅教学镜头（兴趣岛增长中台_2）里人物从远处走到镜头前，
    逐像素求散度得到 -1.01，被判成"拉远"；而镜头根本没动。
    分块取中位后，占据少数格子的人物不能再左右全局结论。
    """
    import cv2

    height, width = 120, 160
    background = np.full((height, width), 40, dtype=np.uint8)
    # 一个亮块代表人物，从远处（小）走到近处（大），背景完全不动
    first = background.copy()
    first[50:70, 74:86] = 235
    second = background.copy()
    second[20:100, 55:105] = 235
    flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 21, 3, 5, 1.2, 0)

    _, _, divergence, _, _ = of._flow_stats(flow)

    estimate = of.analyze_flow_shot(
        of.FlowMeasurement(
            frames=20, translation=0.2, direction=0.5,
            divergence=divergence, subject_dominance=0.3,
        ),
        SHOT,
    )
    assert estimate.kind != "pull_out", (
        f"人物走近不是镜头拉远；散度={divergence:.2f}，判定={estimate.kind}"
    )


# ----------------------------------------------------------------------
# analyze_flow_shot：量测 → 结论的判据
# ----------------------------------------------------------------------
def _m(**kwargs) -> of.FlowMeasurement:
    base = dict(frames=40, translation=0.0, direction=0.0, divergence=0.0)
    base.update(kwargs)
    return of.FlowMeasurement(**base)  # type: ignore[arg-type]


def test_fixed_needs_a_quiet_patch_not_just_a_low_median() -> None:
    """固定的判据是「画面里有一成区域基本静止」，不是「位移中位数小」。

    实测数据（16 个已标注镜头）：人物朝镜头走的镜头位移中位数 0.37~0.48，
    看着像镜头在动；但背景大片静止，p10 只有 0.06 → 固定。
    """
    # 中位不低，但有一成格子是静止的 → 固定
    assert of.analyze_flow_shot(
        _m(translation=0.40, p10_translation=0.06, direction=0.61, divergence=-0.01),
        SHOT,
    ).kind == "fixed"
    # 反过来：中位不高但整幅都在动（p10 也高）→ 不能判固定
    assert of.analyze_flow_shot(
        _m(translation=0.40, p10_translation=0.35, direction=0.61, divergence=-0.01),
        SHOT,
    ).kind == "unknown"


def test_fixed_confidence_beats_model_alone() -> None:
    estimate = of.analyze_flow_shot(
        _m(translation=0.05, p10_translation=0.02, direction=0.99), SHOT
    )
    assert estimate.kind == "fixed"
    # 唯一来源的一致性系数是 0.85，所以上限是 0.95×0.85×0.95 ≈ 0.77；
    # 关键是它必须**高于**模型单独判断的 0.70，否则等于没有 CV 加成
    assert estimate.confidence > 0.75


def test_pan_needs_both_translation_and_direction() -> None:
    # 镜头摇的时候几乎所有格子一起动，所以 p10 跟中位同量级
    assert of.analyze_flow_shot(
        _m(translation=2.0, p10_translation=1.8, direction=0.9, divergence=0.05), SHOT
    ).kind == "pan_or_track"
    # 位移大但方向散 → 不是摇镜
    assert of.analyze_flow_shot(
        _m(translation=2.0, p10_translation=1.8, direction=0.5, divergence=0.05), SHOT
    ).kind == "unknown"


def test_zoom_beats_pan_and_has_direction_sign() -> None:
    pushed = of.analyze_flow_shot(
        _m(translation=1.5, p10_translation=1.3, direction=0.9, divergence=0.8), SHOT
    )
    assert pushed.kind == "push_in"

    pulled = of.analyze_flow_shot(
        _m(translation=1.5, p10_translation=1.3, direction=0.9, divergence=-0.8), SHOT
    )
    assert pulled.kind == "pull_out"


def test_push_in_and_fixed_separated_by_divergence_not_translation() -> None:
    """**这条来自真实标注数据**：fixed 与 push_in 的位移完全重叠。

    标注为 fixed 的镜头位移 0.00~0.48、散度全在 ±0.04；
    标注为 push_in 的镜头位移 0.36~0.41、散度 +0.25/+0.32。
    所以判别量只能是散度 —— 靠位移分不开。
    """
    assert of.analyze_flow_shot(
        _m(translation=0.41, p10_translation=0.06, direction=0.51, divergence=0.25),
        SHOT,
    ).kind == "fixed"          # 背景有静止区域 → 固定优先
    assert of.analyze_flow_shot(
        _m(translation=0.41, p10_translation=0.30, direction=0.51, divergence=0.25),
        SHOT,
    ).kind == "push_in"        # 整幅一起动且径向收缩 → 推近


def test_handheld_only_when_direction_is_scattered() -> None:
    assert of.analyze_flow_shot(
        _m(translation=1.2, p10_translation=1.0, direction=0.3, divergence=0.1), SHOT
    ).kind == "handheld"


def test_insufficient_frames_says_unknown_not_a_guess() -> None:
    """采样帧不够就明说，绝不硬判（PRD 红线 7）。"""
    estimate = of.analyze_flow_shot(
        _m(frames=2, translation=9.9, direction=0.99, divergence=9.9), SHOT
    )
    assert estimate.kind == "unknown"
    assert estimate.confidence == 0.0
    assert "不足" in estimate.note


# ----------------------------------------------------------------------
# 编排层适配器：解码一次、逐镜头查表
# ----------------------------------------------------------------------
def test_analyzer_bind_decodes_once_and_answers_every_shot(monkeypatch) -> None:
    calls: list[str] = []

    def fake_decode(ffmpeg_bin, path, **kwargs):
        calls.append(path)
        return np.zeros((64, 120, 160), dtype=np.uint8)

    monkeypatch.setattr(of, "decode_gray", fake_decode)

    analyzer = of.OpticalFlowMotionAnalyzer("ffmpeg")
    shots = [
        ShotBoundary(index=1, start_ms=0, end_ms=2000),
        ShotBoundary(index=2, start_ms=2000, end_ms=4000),
    ]
    analyzer.bind("clip.mp4", shots)

    for shot in shots * 3:  # 编排层是逐镜头反复调的
        analyzer.analyze("", shot)

    assert calls == ["clip.mp4"], f"整段只该解码一次，实得 {calls}"
    assert set(analyzer.measurements) == {1, 2}


def test_analyzer_without_bind_refuses_to_guess() -> None:
    analyzer = of.OpticalFlowMotionAnalyzer("ffmpeg")
    estimate = analyzer.analyze("", SHOT)
    assert estimate.kind == "unknown"
    assert estimate.confidence == 0.0


def test_bind_clears_cache_when_video_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        of, "decode_gray", lambda *a, **k: np.zeros((32, 120, 160), dtype=np.uint8)
    )
    analyzer = of.OpticalFlowMotionAnalyzer("ffmpeg")
    analyzer.bind("a.mp4", [SHOT])
    analyzer.analyze("", SHOT)
    assert analyzer.measurements

    analyzer.bind("b.mp4", [SHOT])
    assert analyzer.measurements == {}, "换视频必须作废旧量测，否则会串数据"


def test_build_motion_analyzer_returns_none_without_opencv(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert of.build_motion_analyzer("ffmpeg") is None


# ----------------------------------------------------------------------
# 帧下标映射
# ----------------------------------------------------------------------
def test_frame_indices_follow_shot_boundaries() -> None:
    shots = [
        ShotBoundary(index=1, start_ms=0, end_ms=1000),
        ShotBoundary(index=2, start_ms=1000, end_ms=2500),
    ]
    indices = of._frame_indices(shots, fps=8.0)
    assert indices[0] == list(range(0, 8))
    assert indices[1] == list(range(8, 20))


@pytest.mark.parametrize(
    "stderr,expected",
    [
        # 160 宽的等比高度：1280/720×160 ≈ 284、1080/1920×160 = 90
        ("Stream #0:0: Video: h264, yuv420p, 720x1280, 25 fps", 284),
        ("Stream #0:0: Video: h264, 1920x1080", 90),
        ("no video line at all", 320),  # 认不出来时按竖屏 2:1 兜底
    ],
)
def test_scaled_height_reads_source_resolution(stderr: str, expected: int) -> None:
    assert of._scaled_height(stderr, 160) == expected


def test_frame_indices_are_clamped_to_available_frames() -> None:
    """ffmpeg 实际采样帧数可能少一帧，越界会把整条反推带崩。"""
    shots = [ShotBoundary(index=1, start_ms=0, end_ms=4000)]
    indices = of._frame_indices(shots, fps=8.0, frame_count=30)
    assert indices[0] == list(range(0, 30))
