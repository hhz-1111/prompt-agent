"""Provider 层与反推编排的测试。

这里用**确定性替身**替代真实模型调用：

- 网络调用不可复现，测它等于测网络；
- 但要测的是**编排逻辑**（CV 优先、冲突处置、置信度分档、不猜原则），
  这些完全可以用替身精确验证。

真实调用的连通性由 ``scripts/smoke_providers.py`` 负责，不进单测。
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from engine.app.config import Settings
from engine.app.pipeline.infer import (
    Inferencer,
    _camera_agrees,
    build_inferencer,
    reconcile_shot_mode,
)
from engine.app.pipeline import to_ir as to_ir_module
from engine.app.providers import cv as cv_layer
from engine.app.providers.ffmpeg_media import (
    FrameMean,
    _build_boundaries,
    detect_cuts,
    parse_frame_means,
)
from engine.app.providers.mimo import _segments_from_asr, _try_parse
from engine.core.contracts import (
    FramingEstimate,
    MotionEstimate,
    SemanticSummary,
    ShotBoundary,
    ShotSignals,
    ShotUnderstanding,
    TranscriptSegment,
    VideoInfo,
)
from engine.core.types import (
    BlinkBeat,
    ConfidenceBasis,
    Genre,
    QualityPreset,
    ShotMode,
    Span,
    compute_confidence,
)


# ==========================================================================
# ffmpeg 解析层（纯函数，可直接测）
# ==========================================================================
class TestFrameParsing:
    def test_parse_field_mean_lines(self):
        stderr = (
            "[Parsed_showinfo_1 @ 0x1] n:   0 pts:      0 pts_time:0       "
            "fmt:yuv420p s:64x64 mean:[80 105 196] stdev:[0.0 0.0 0.0]\n"
            "[Parsed_showinfo_1 @ 0x1] n:   1 pts:    512 pts_time:0.04    "
            "fmt:yuv420p s:64x64 mean:[91 180 86] stdev:[0.0 0.0 0.0]\n"
        )
        frames = parse_frame_means(stderr)
        assert len(frames) == 2
        assert frames[0].mean == (80.0, 105.0, 196.0)
        assert frames[1].pts_time == pytest.approx(0.04)

    def test_parse_ignores_non_frame_lines(self):
        assert parse_frame_means("Input #0, mov,mp4\nStream mapping:\n") == []

    def test_perceptual_distance_orders_correctly(self):
        """感知加权距离：硬切要显著大于同场景内的轻微色变。"""
        a = FrameMean(0, 0.0, (183.0, 28.0, 28.0))   # 深红
        b = FrameMean(1, 0.04, (21.0, 101.0, 192.0))  # 蓝
        c = FrameMean(2, 3.0, (69.0, 90.0, 100.0))    # 冷灰
        d = FrameMean(3, 3.04, (141.0, 110.0, 99.0))  # 暖棕
        assert a.distance(b) > 0.35, "硬切距离应显著"
        assert c.distance(d) > a.distance(b) * 0.2

    def test_detect_cuts_respects_min_gap(self):
        frames = [
            FrameMean(0, 0.00, (0.0, 0.0, 0.0)),
            FrameMean(1, 0.04, (255.0, 255.0, 255.0)),   # 切点
            FrameMean(2, 0.08, (0.0, 0.0, 0.0)),         # 紧跟的抖动，应被合并
            FrameMean(3, 2.00, (255.0, 255.0, 255.0)),   # 另一个切点
        ]
        cuts = detect_cuts(frames, threshold=0.10, min_gap_ms=300)
        assert cuts == [40, 2000]

    def test_build_boundaries_covers_whole_clip(self):
        shots = _build_boundaries([3000, 6000], 9.0)
        assert [(s.start_ms, s.end_ms) for s in shots] == [
            (0, 3000), (3000, 6000), (6000, 9000),
        ]
        assert [s.index for s in shots] == [1, 2, 3]

    def test_build_boundaries_drops_edge_jitter(self):
        """贴边 30ms 的"切点"是编码抖动，不是真切镜。"""
        shots = _build_boundaries([30, 5000], 9.0)
        assert len(shots) == 2
        assert shots[0].start_ms == 0


# ==========================================================================
# 置信度公式（MULTIMODAL §4.2）
# ==========================================================================
class TestConfidence:
    def test_base_ranking_matches_spec(self):
        # ffmpeg 作为唯一来源时也要乘一致性系数 0.85：
        # 一致/冲突两个系数都得有两个独立来源才谈得上
        ffmpeg = compute_confidence(ConfidenceBasis.FFMPEG, consistency="single",
                                    evidence="timecode_and_frame")
        assert ffmpeg == pytest.approx(1.0 * 0.85 * 1.0)

        cv = compute_confidence(ConfidenceBasis.CV, consistency="single",
                                evidence="timecode_and_frame")
        vlm = compute_confidence(ConfidenceBasis.VLM_ALONE, consistency="single",
                                 evidence="timecode_and_frame")
        conflict = compute_confidence(ConfidenceBasis.VLM_CONFLICT, consistency="conflict",
                                      evidence="timecode_and_frame")
        assert cv > vlm > conflict

    def test_two_sources_agree_beats_single_source(self):
        single = compute_confidence(ConfidenceBasis.VLM_CROSSED, consistency="single",
                                    evidence="timecode_and_frame")
        agree = compute_confidence(ConfidenceBasis.VLM_CROSSED, consistency="agree",
                                   evidence="timecode_and_frame")
        assert agree > single == pytest.approx(0.90 * 0.85 * 1.0)

    def test_multiplicative_coefficients(self):
        """基础分 × 一致性系数 × 证据完整度。"""
        value = compute_confidence(
            ConfidenceBasis.CV, consistency="agree", evidence="timecode_and_frame"
        )
        assert value == pytest.approx(0.95 * 1.0 * 1.0)

        degraded = compute_confidence(
            ConfidenceBasis.CV, consistency="single", evidence="none"
        )
        assert degraded == pytest.approx(0.95 * 0.85 * 0.70, abs=1e-4)

    def test_structured_inference_discount(self):
        plain = compute_confidence(ConfidenceBasis.VLM_CROSSED, consistency="agree",
                                   evidence="timecode_only")
        discounted = compute_confidence(
            ConfidenceBasis.VLM_CROSSED, consistency="agree", evidence="timecode_only",
            is_structured_inference=True,
        )
        assert discounted == pytest.approx(plain * 0.85)

    def test_disposition_thresholds(self):
        from engine.core.types import Provenance

        def make(c: float) -> Provenance:
            return Provenance(confidence=c, basis=ConfidenceBasis.CV)

        assert make(0.90).disposition == "certain"
        assert make(0.75).disposition == "accept"
        assert make(0.60).disposition == "review"
        assert make(0.40).disposition == "blank"
        # 红线 7：<0.70 必须用户确认
        assert make(0.60).must_confirm


# ==========================================================================
# ASR 结果解析（不编时间戳）
# ==========================================================================
class TestAsrParsing:
    def test_parses_segments_with_timestamps(self):
        raw = {"segments": [
            {"text": "你好", "start": 0.0, "end": 1.2},
            {"text": "再见", "start": 1.2, "end": 2.4},
        ]}
        segments = _segments_from_asr(raw)
        assert [s.text for s in segments] == ["你好", "再见"]
        assert segments[0].span == Span(0, 1200)

    def test_whole_text_uses_known_duration(self):
        """只有整段文本时，用已知音频时长给区间 —— 但置信度按"只有时间码"计。"""
        segments = _segments_from_asr({"text": "整段话"}, expected_span_ms=5440)
        assert segments[0].span == Span(0, 5440)
        assert segments[0].confidence == pytest.approx(0.90 * 0.85 * 0.90)

    def test_unknown_duration_lowers_confidence(self):
        """时长也拿不到时，明确降档，绝不假装有词级对齐。"""
        with_duration = _segments_from_asr({"text": "x"}, expected_span_ms=5000)
        without = _segments_from_asr({"text": "x"})
        assert without[0].confidence < with_duration[0].confidence
        assert without[0].span == Span(0, 0)

    def test_reads_chat_completions_shape(self):
        """MiMo 的 ASR 走 chat/completions，结果在 message.content。"""
        raw = {"choices": [{"message": {"content": "你这气色也太好了吧"}}]}
        segments = _segments_from_asr(raw, expected_span_ms=5440)
        assert segments[0].text == "你这气色也太好了吧"

    def test_empty_returns_nothing(self):
        assert _segments_from_asr({}) == []
        assert _segments_from_asr({"text": "   "}) == []


class TestJsonRepair:
    def test_parses_plain_and_fenced(self):
        assert _try_parse('{"a": 1}') == {"a": 1}
        assert _try_parse('```json\n{"a": 1}\n```') == {"a": 1}

    def test_recovers_from_leading_prose(self):
        """模型偶尔会串入思考文本，要能从噪声里捞出 JSON。"""
        assert _try_parse('Here is the result:\n{"a": 1}') == {"a": 1}

    def test_returns_none_on_garbage(self):
        assert _try_parse("完全不是 JSON") is None
        assert _try_parse("") is None


# ==========================================================================
# CV 层：能算的算，算不了的如实留空
# ==========================================================================
class TestCvLayer:
    @staticmethod
    def _frames(values: list[float], step: float = 0.04) -> list[FrameMean]:
        return [
            FrameMean(index=i, pts_time=i * step, mean=(v, v, v))
            for i, v in enumerate(values)
        ]

    def test_stable_sequence_is_fixed(self):
        shot = ShotBoundary(1, 0, 2000)
        frames = self._frames([100.0] * 60)
        result = cv_layer.analyze_motion(frames, shot)
        assert result.kind == "fixed"
        assert "固定机位" in result.note

    def test_monotonic_large_swing_is_push_in(self):
        shot = ShotBoundary(1, 0, 2000)
        frames = self._frames([100.0 + i * 3 for i in range(60)])
        result = cv_layer.analyze_motion(frames, shot)
        assert result.kind == "push_in"

    def test_alternating_sequence_is_unknown_not_a_guess(self):
        """亮度序列**分不清手持与横摇**，所以不能硬判。

        这条测试此前断言 handheld —— 那是在测试一个我后来撤掉的能力。
        实测发现：固定机位视频里人物做深蹲、大面积移位时，
        亮度照样剧烈变化，算法把**被摄体运动**误判成了镜头运动。
        现在只在有把握时给 fixed / push_in，其余交给视觉模型。
        """
        shot = ShotBoundary(1, 0, 2000)
        values = [100.0 + (8 if i % 2 else -8) for i in range(60)]
        result = cv_layer.analyze_motion(frames=self._frames(values), shot=shot)
        assert result.kind == "unknown"
        assert "无法区分镜头运动与被摄体运动" in result.note
        assert result.confidence == 0.0, "给不出结论就不该给置信度"

    def test_moderate_swing_is_unknown(self):
        """中等幅度变化在真实内容里多为被摄体运动，不该判成运镜。"""
        shot = ShotBoundary(1, 0, 2000)
        frames = self._frames([100.0 + (i % 5) * 0.6 for i in range(60)])
        assert cv_layer.analyze_motion(frames, shot).kind == "unknown"

    def test_too_few_frames_is_unknown_not_a_guess(self):
        """帧不够就说 unknown，绝不编一个运镜档位。"""
        result = cv_layer.analyze_motion(self._frames([100.0, 101.0]),
                                         ShotBoundary(1, 0, 100))
        assert result.kind == "unknown"
        assert "不足" in result.note

    def test_missing_capabilities_are_declared(self):
        caps = cv_layer.CvCapabilities(framing=False, blocking=False, people_count=False)
        assert "景别（需人体框占比）" in caps.missing
        assert "站位（需检测框位置）" in caps.missing

    def test_unavailable_framing_returns_zero_confidence(self):
        """未实现的能力必须返回 0 置信度，否则下游会以为拿到了 CV 证据。"""
        result = cv_layer.analyze_framing_unavailable(ShotBoundary(1, 0, 1000))
        assert result.kind == "unknown"
        assert result.confidence == 0.0

    def test_camera_agreement_table(self):
        assert _camera_agrees("平视", "fixed")
        assert _camera_agrees("推近", "push_in")
        assert not _camera_agrees("推近", "fixed")
        assert not _camera_agrees("手持", "push_in")


class TestCompositionSide:
    """构图偏侧 = 整个人物跨度的中点，**不是**最显眼那个人的左右。

    实测踩过：双人镜头里两人在 x=0.30 与 0.69，取最靠左的人会判 left，
    但人工标注一致认为这种"一左一右"就是居中。
    """

    @staticmethod
    def _box(center_x: float, height: float = 0.5):
        half = 0.05
        return cv_layer.PersonBox(
            x1=center_x - half, y1=0.1, x2=center_x + half, y2=0.1 + height,
            confidence=0.9,
        )

    def test_two_people_either_side_is_center(self):
        side, midpoint, count = cv_layer.composition_side([self._box(0.30), self._box(0.69)])
        assert side == "center", f"一左一右应判居中，实得 {side}"
        assert midpoint == pytest.approx(0.495, abs=0.01)
        assert count == 2

    def test_both_people_on_the_left_is_left(self):
        side, midpoint, _ = cv_layer.composition_side([self._box(0.16), self._box(0.34)])
        assert side == "left"
        assert midpoint == pytest.approx(0.25, abs=0.01)

    def test_single_person_uses_their_own_position(self):
        assert cv_layer.composition_side([self._box(0.50)])[0] == "center"
        assert cv_layer.composition_side([self._box(0.80)])[0] == "right"
        assert cv_layer.composition_side([self._box(0.12)])[0] == "left"

    def test_background_people_do_not_drag_the_composition(self):
        """菜市场那类场景背景全是路人，不该把构图拽走。"""
        side, _, count = cv_layer.composition_side([
            self._box(0.50, height=0.92),
            self._box(0.70, height=0.12),
            self._box(0.94, height=0.07),
        ])
        assert side == "center"
        assert count == 1, "背景路人（高度 0.12/0.07）不该进构图计算"

    def test_no_boxes_gives_empty_not_a_guess(self):
        assert cv_layer.composition_side([]) == ("", 0.0, 0)

    def test_side_of_bands(self):
        assert cv_layer.side_of(0.39) == "left"
        assert cv_layer.side_of(0.40) == "center"
        assert cv_layer.side_of(0.60) == "center"
        assert cv_layer.side_of(0.61) == "right"


class TestFramingBands:
    """景别分档阈值来自 16 个已标注镜头（scripts/fit_framing_bands.py）。

    这组断言把**实测校准出来的档位边界**钉住，防止日后被无意改回拍脑袋的数。
    """

    @staticmethod
    def _box(height: float):
        return cv_layer.PersonBox(x1=0.4, y1=0.0, x2=0.6, y2=height, confidence=0.9)

    @pytest.mark.parametrize(
        "height,expected",
        [
            (0.30, "全景"),
            (0.47, "全景"),      # 标注"全景"的实测最大段
            (0.57, "中全景"),    # 标注在这里给的是全景，是已知的边界损耗
            (0.74, "中景"),      # 标注"中景"的实测最小段
            (0.85, "中景"),
            (0.92, "近景"),      # 标注"近景"的实测下限
            (1.10, "脸部特写"),
        ],
    )
    def test_bands_match_annotation(self, height: float, expected: str):
        assert cv_layer.framing_from_person(self._box(height)).kind == expected

    def test_unknown_ratio_band_is_honest(self):
        """0.87 在标注里同时被标成中景与近景 —— 代码只能给一个答案，取中景。"""
        assert cv_layer.framing_from_person(self._box(0.87)).kind == "中景"


class TestVideoLimits:
    """PRD F1.1：≤3min / ≤500MB。超限要**早失败**，不是跑完再报错。"""

    def test_over_duration_is_rejected_before_analysis(self):
        inferencer = _inferencer(probe=_StubProbe(_info(240.0)))  # 4 分钟
        with pytest.raises(ValueError, match="超过 3 分钟上限"):
            inferencer.infer("fake.mp4")

    def test_zero_duration_is_rejected(self):
        inferencer = _inferencer(probe=_StubProbe(_info(0.0)))
        with pytest.raises(ValueError, match="时长为 0"):
            inferencer.infer("fake.mp4")

    def test_three_minutes_is_allowed(self):
        inferencer = _inferencer(probe=_StubProbe(_info(180.0)))
        report = inferencer.infer("fake.mp4")
        assert report.video.duration_sec == 180.0


class TestHookGateHonesty:
    """G11 在无基准时不能假称"已校验保留"。"""

    def test_no_baseline_says_so(self):
        from engine.app.pipeline.infer import Inferencer
        from engine.core.gates import g11_hook_preserved

        ir = Inferencer.to_ir(_inferencer().infer("fake.mp4"))
        result = g11_hook_preserved(ir, None)
        assert result.status == "ok"
        assert "无原片基准可比对类型" in result.detail

    def test_with_baseline_compares_counts(self):
        from engine.app.pipeline.infer import Inferencer
        from engine.core.gates import g11_hook_preserved

        baseline = Inferencer.to_ir(_inferencer().infer("fake.mp4"))
        modified = Inferencer.to_ir(_inferencer().infer("fake.mp4"))
        for beat in modified["creative"]["C2_beats"]:
            beat["beat_role"] = "demo"
        result = g11_hook_preserved(modified, baseline)
        assert result.blocking


class TestShotBoundaryBuilding:
    """PySceneDetect 切点 → 连续镜头区间。"""

    def test_covers_whole_clip(self):
        from engine.app.providers.scene_detect import build_boundaries

        shots = build_boundaries([3000, 6000], 9.0)
        assert [(s.start_ms, s.end_ms) for s in shots] == [
            (0, 3000), (3000, 6000), (6000, 9000),
        ]
        assert [s.index for s in shots] == [1, 2, 3]

    def test_drops_edge_jitter(self):
        from engine.app.providers.scene_detect import build_boundaries

        shots = build_boundaries([30, 5000], 9.0)
        assert len(shots) == 2
        assert shots[0].start_ms == 0

    def test_merges_cuts_too_close_together(self):
        """一段渐变会被判成多次切点，必须按最小间距合并。"""
        from engine.app.providers.scene_detect import build_boundaries

        shots = build_boundaries([3000, 3100, 3200, 6000], 9.0, min_gap_ms=300)
        assert [(s.start_ms, s.end_ms) for s in shots] == [
            (0, 3000), (3000, 6000), (6000, 9000),
        ]

    def test_dedupes_and_sorts(self):
        from engine.app.providers.scene_detect import build_boundaries

        shots = build_boundaries([6000, 3000, 6000, 3000], 9.0)
        assert len(shots) == 3

    def test_empty_cuts_gives_single_shot(self):
        from engine.app.providers.scene_detect import build_boundaries

        shots = build_boundaries([], 12.5)
        assert len(shots) == 1
        assert shots[0].end_ms == 12500

    def test_tail_sliver_merges_into_previous_shot(self):
        """视频末尾常被切出几百毫秒的残留镜头。

        实测踩过：一条视频切出 11917-12097ms（180ms）的尾巴，
        抽帧时 `-ss 12.007s` 落到实际帧位置之外，让 ffmpeg 报错、
        整条反推失败。碎片镜头必须并入前一镜。
        """
        from engine.app.providers.scene_detect import build_boundaries

        shots = build_boundaries([6417, 10417, 11917], 12.097)
        assert [(s.start_ms, s.end_ms) for s in shots] == [
            (0, 6417), (6417, 10417), (10417, 12097),
        ]
        assert all(s.duration_sec >= 0.25 for s in shots[1:])

    def test_all_shots_are_non_degenerate(self):
        from engine.app.providers.scene_detect import build_boundaries

        shots = build_boundaries([100, 200, 5000, 11950], 12.0, min_gap_ms=300)
        assert all(s.end_ms > s.start_ms for s in shots)
        assert [s.index for s in shots] == list(range(1, len(shots) + 1))


class TestSceneDetectAdapter:
    def test_availability_probe_does_not_raise(self):
        from engine.app.providers.scene_detect import SceneDetectShots

        assert isinstance(SceneDetectShots(Settings()).available, bool)

    def test_build_inferencer_prefers_scenedetect(self):
        """装了 PySceneDetect 就该用它的检测器，而不是自研的均值实现。"""
        from engine.app.providers.scene_detect import SceneDetectShots

        inferencer = build_inferencer(Settings(mimo_api_key=""))
        if SceneDetectShots(Settings()).available:
            assert type(inferencer.detector).__name__ == "SceneDetectShots"
        else:
            assert type(inferencer.detector).__name__ == "FFmpegMedia"


class TestShotModeReconciliation:
    """切分数是 CV 算出来的确定事实，模型没有推翻它的依据。

    实测出现过"1 个镜头却判 multi_shot"的自相矛盾 —— 会直接产出错误的
    拆条策略与提示词条数。
    """

    def test_single_shot_forces_one_take(self):
        corrected = reconcile_shot_mode(_semantic(shot_mode=ShotMode.MULTI_SHOT), shot_count=1)
        assert corrected.shot_mode is ShotMode.ONE_TAKE
        assert "校正" in corrected.note

    def test_multi_shot_forces_multi_shot(self):
        corrected = reconcile_shot_mode(_semantic(shot_mode=ShotMode.ONE_TAKE), shot_count=3)
        assert corrected.shot_mode is ShotMode.MULTI_SHOT

    def test_consistent_mode_untouched(self):
        original = _semantic(shot_mode=ShotMode.ONE_TAKE)
        assert reconcile_shot_mode(original, shot_count=1) is original

    def test_same_angle_segments_survives_single_shot(self):
        """同机位分段本质就是一个连续镜头，不该被改成 multi_shot。"""
        corrected = reconcile_shot_mode(
            _semantic(shot_mode=ShotMode.SAME_ANGLE_SEGMENTS), shot_count=1
        )
        assert corrected.shot_mode is ShotMode.SAME_ANGLE_SEGMENTS

    def test_pipeline_reconciles_end_to_end(self):
        report = _inferencer(semantic=_StubSemantic(
            _semantic(shot_mode=ShotMode.MULTI_SHOT)
        )).infer("fake.mp4")
        # _StubDetector 给 2 个镜头，所以 multi_shot 是自洽的
        assert report.semantic.shot_mode is ShotMode.MULTI_SHOT

        single = _inferencer(
            detector=_StubDetector([ShotBoundary(1, 0, 9000)]),
            semantic=_StubSemantic(_semantic(shot_mode=ShotMode.MULTI_SHOT)),
        ).infer("fake.mp4")
        assert single.semantic.shot_mode is ShotMode.ONE_TAKE


class TestPersonExtraction:
    def test_group_labels_are_filtered(self):
        """群体不是个体：给「台下观众」建多视图既浪费配额又违反 D2。

        这里让 ASR 只说"有人说话"但不给说话人，从而只考察视觉标签这一路。
        """
        visual = _StubVisual([
            ShotUnderstanding(shot_index=1, people=("魏巍老师", "台下观众", "8名学员")),
            ShotUnderstanding(shot_index=2, people=("魏巍老师",)),
        ])
        inferencer = _inferencer(
            visual=visual,
            transcriber=_StubTranscriber([
                TranscriptSegment(text="大家好", span=Span(0, 2000), confidence=0.9),
            ]),
        )
        report = inferencer.infer("fake.mp4")
        ir = Inferencer.to_ir(report)
        names = [p["name"] for p in ir["locked"]["B2_person"]]
        assert names == ["魏巍老师"]

    def test_person_count_is_capped(self):
        """实测有一条带练视频抽出 10 个"人物"，生成 10 张多视图撞爆 9 张配额。"""
        many = tuple(f"人物{i}" for i in range(10))
        visual = _StubVisual([ShotUnderstanding(shot_index=1, people=many)])
        report = _inferencer(visual=visual).infer("fake.mp4")
        ir = Inferencer.to_ir(report)

        assert len(ir["locked"]["B2_person"]) <= 4
        image_refs = [r for r in ir["references"] if r["type"] == "multi_view"]
        assert len(image_refs) <= 4

    def test_overflow_is_reported_not_silent(self):
        from engine.app.pipeline.to_ir import person_overflow_note

        many = tuple(f"人物{i}" for i in range(10))
        report = _inferencer(
            visual=_StubVisual([ShotUnderstanding(shot_index=1, people=many)])
        ).infer("fake.mp4")
        note = person_overflow_note(report)
        assert note
        assert "已截取" in note

    def test_no_overflow_produces_no_note(self):
        from engine.app.pipeline.to_ir import person_overflow_note

        report = _inferencer().infer("fake.mp4")
        assert person_overflow_note(report) == ""


class TestUnidentifiedPersons:
    """**实测踩过的坑**：视觉模型只给位置词时，位置词变成了人物名。

    交付物里出现「左侧人物形象参考[左侧人物多视图]」——
    凭空让用户准备一张不存在的素材，还多出两个假角色。
    位置词不是身份：它随镜头变，也不是**人**的属性。
    """

    def test_pure_position_labels_are_not_identities(self):
        from engine.app.pipeline.to_ir import dedupe_person_labels

        assert dedupe_person_labels(["左侧人物", "中央人物", "右侧背影人物"]) == []

    def test_position_words_are_stripped_when_appearance_remains(self):
        from engine.app.pipeline.to_ir import dedupe_person_labels

        # 同一人换机位 → 同一个身份键 → 合成一个
        names = dedupe_person_labels(["左侧蓝衣男子", "右侧蓝衣男子", "后方蓝衣男子"])
        assert len(names) == 1, names
        assert "左" not in names[0] and "右" not in names[0] and "后" not in names[0]
        # 名字统一压成「颜色+服装+性别」形式（用户 2026-09-29 指定的命名规则）
        assert names == ["蓝色衣服男子"]

    def test_duplicate_names_after_stripping_fall_back_to_index(self):
        """剥掉位置词后撞名时退回「人物N」，而不是让两个角色同名。

        用「女子」与「画面女子」构造：前者靠"剥离后剩下的字"取身份键，
        后者靠"剥离后剩下的字"也一样 —— 两条不同身份键却会撞出同一个名字。
        """
        from engine.app.pipeline.to_ir import dedupe_person_labels

        names = dedupe_person_labels(["中央女子", "背影女子"])
        assert len({*names}) == len(names), f"不该有两个同名角色：{names}"
        assert all("中央" not in n and "背影" not in n for n in names), names

    def test_same_person_across_shots_still_collapses(self):
        """同一个人换机位要合并 —— 位置词不同不构成"两个人"。"""
        from engine.app.pipeline.to_ir import dedupe_person_labels

        assert dedupe_person_labels(["中央女子", "左侧女子"]) == []
        assert dedupe_person_labels(["中央紫衣女子", "画面左侧紫衣女子"]) == ["紫色衣服女子"]

    def test_person_variants_in_shot_text_are_normalized(self):
        """同一个人在同一片子里被写成两种名字时，逐镜头描述要收敛到规范名。

        实测出现过的原样数据：人物清单是「灰色运动外套女子」，
        而镜头 2/3 的 blocking 写的是「深灰运动外套女子」——
        不收敛的话正文会一边写「人物1」一边冒出另一个名字，用户对不上号。
        """
        visual = _StubVisual([
            ShotUnderstanding(shot_index=1, people=("米色开衫女子",),
                              blocking="米色开衫女子沿木地板走向前台"),
            ShotUnderstanding(shot_index=2, people=("灰色运动外套女子", "米色开衫女子"),
                              blocking="深灰运动外套女子站在柜台内侧讲解"),
        ])
        report = _inferencer(
            visual=visual,
            transcriber=_StubTranscriber([
                TranscriptSegment(text="你好", span=Span(0, 2000),
                                  speaker="米色开衫女子", confidence=0.9),
            ]),
        ).infer("fake.mp4")
        ir = Inferencer.to_ir(report)

        names = [p["name"] for p in ir["locked"]["B2_person"]]
        assert "灰色运动外套女子" in names, names
        blocking = " ".join(b["blocking"] for b in ir["creative"]["C2_beats"])
        assert "深灰运动外套女子" not in blocking, "同一身份的不同写法必须先收敛"
        assert "灰色运动外套女子" in blocking

    def test_alias_map_keeps_unknown_names(self):
        """认不出身份的写法**原样保留** —— 不硬塞给别人。"""
        from engine.app.pipeline.to_ir import canonical_person_mentions, person_alias_map

        people = ["灰色运动外套女子", "米色开衫女子"]
        aliases = person_alias_map(["深灰运动外套女子", "路过的快递员"], people)
        assert aliases == {"深灰运动外套女子": "灰色运动外套女子"}
        text = "路过的快递员递上包裹"
        assert canonical_person_mentions(text, aliases) == text

    def test_color_comes_from_the_same_garment(self):
        """一次写两件衣服时，颜色必须跟**它修饰的那件**配对，且优先上衣。

        实测踩过：模型给「红衣黑色短裤者」，独立取色取到「黑色」（短裤的），
        独立取服装按词表顺序取到「短裤」，拼成「黑色短裤人物」——
        这个人的红上衣在交付物里彻底消失，用户认不出是谁。
        """
        from engine.app.pipeline.to_ir import compact_person_name

        assert compact_person_name("红衣黑色短裤者") == "红色衣服人物"
        assert compact_person_name("浅紫色运动服扎发髻者") == "紫色运动服人物"
        # 单字「衣」= 上衣，不是"随便一件衣服"
        assert compact_person_name("蓝衣男子") == "蓝色衣服男子"

    def test_short_reference_maps_to_the_same_person(self):
        """同一段描述里先写全名、后面用简写（「红衣者」）时，两者必须认成一个人。"""
        from engine.app.pipeline.to_ir import (
            canonical_person_mentions,
            compact_person_name,
            dedupe_person_labels,
            person_alias_map,
        )

        people = [
            compact_person_name(name)
            for name in dedupe_person_labels(["红衣黑色短裤者", "浅紫色运动服扎发髻者"])
        ]
        aliases = person_alias_map(
            [], people, texts=["红衣者位于画面偏左，浅紫色运动服者位于其右后方"]
        )
        text = canonical_person_mentions("红衣者位于画面偏左", aliases)
        assert "红衣者" not in text, aliases
        assert text.startswith(tuple(people)), text

    def test_aggregate_blocking_follows_the_model_not_the_boxes(self):
        """聚合站位句也要用模型说的位置 —— 检测框会把屏幕/倒影里的人算进来。

        实测踩过：演播厅单人示范片的背景大屏上印着 4 张人物照片，
        YOLO 在右侧多检出 2 个框，交付物就写「有角色位于画面右侧（检测框中心
        x≈0.816）」，而画面里只有 1 个人、且站在左侧。
        """
        from types import SimpleNamespace

        from engine.app.pipeline import to_ir

        ir = {"locked": {"B2_person": [{"name": "白色上衣男子"}], "B3_blocking": {}}}
        report = SimpleNamespace(
            shots=[
                SimpleNamespace(
                    people_count=3,
                    blocking=[
                        SimpleNamespace(x_ratio=0.242, side="left"),
                        SimpleNamespace(x_ratio=0.816, side="right"),
                    ],
                    composition="center",
                )
            ],
            understandings=[
                SimpleNamespace(
                    people=["白紫拼色上衣男子"],
                    blocking="白紫拼色上衣男子站立于演播室地面中央偏左位置",
                )
            ],
        )
        to_ir._fill_blocking(ir, report)
        text = "、".join(ir["locked"]["B3_blocking"]["positions"])
        assert "有角色位于画面左侧" in text, text
        assert "画面右侧" not in text, f"屏幕上的照片不该出现在站位里：{text}"
        assert "检测框中心" not in text, f"有模型证据时不该再用检测框口径：{text}"

    def test_aggregate_blocking_falls_back_to_boxes(self):
        """模型一个位置都没说时，才退回检测框口径（带 x 值）。"""
        from types import SimpleNamespace

        from engine.app.pipeline import to_ir

        ir = {"locked": {"B2_person": [{"name": "米色开衫女子"}], "B3_blocking": {}}}
        report = SimpleNamespace(
            shots=[
                SimpleNamespace(
                    people_count=1,
                    blocking=[SimpleNamespace(x_ratio=0.294, side="left")],
                    composition="center",
                )
            ],
            understandings=[SimpleNamespace(people=["米色开衫女子"], blocking="米色开衫女子站着说话")],
        )
        to_ir._fill_blocking(ir, report)
        text = "、".join(ir["locked"]["B3_blocking"]["positions"])
        # 用户 2026-10-07：**不写「检测框中心 x≈0.294」这类内部量测标记** ——
        # 那是我们自己的坐标，生成平台看不懂。量测值留在 provenance 里可追溯。
        assert "有角色位于画面左侧" in text, text
        assert "x≈" not in text and "检测框" not in text, text

    def test_color_is_taken_from_the_leftmost_position(self):
        """一次写两个颜色时取**文本最左**那个，不是词表里最靠前的那个。"""
        from engine.app.pipeline.to_ir import _find_color, compact_person_name

        assert _find_color("紫白色上衣女子") == "紫色"
        assert _find_color("黑白色上衣女子") == "黑色"
        assert _find_color("白紫拼色上衣男子") == "白色"
        # 名字与身份键必须同一口径：否则同一个人算不出同一个键，别名表会失效
        name = compact_person_name("白紫拼色上衣男子")
        assert name == "白色上衣男子"
        from engine.app.pipeline.to_ir import _identity_key

        assert _identity_key("白紫拼色上衣男子") == _identity_key(name)

    def test_gray_is_a_base_color_not_empty(self):
        """「灰色」是基础色本身，不能归一成空。

        实测踩过：单字「灰」在强度修饰词表里，`_base_color("灰色")` 返回空串 ——
        于是**所有穿灰衣服的人身份键都成了空色**，与"压根没写颜色"的人同键，
        两个不同的人会被并成一个（一个人凭空消失）。
        """
        from engine.app.pipeline.to_ir import (
            _base_color,
            _identity_key,
            dedupe_person_labels,
        )

        assert _base_color("灰色") == "灰"
        assert _base_color("深灰色") == "灰"
        assert _base_color("灰米色") == "米"
        # 灰色与"没写颜色"必须算出不同的键
        assert _identity_key("灰色运动外套女子") != _identity_key("运动外套女子")
        # 同一个人的灰色两种写法仍要合并（别名链靠它工作）
        assert _identity_key("深灰运动外套女子") == _identity_key("灰色运动外套女子")
        assert len(dedupe_person_labels(["灰色运动外套女子", "运动外套女子"])) == 2

    def test_box_positions_refused_when_counts_disagree(self):
        """检测框人数与模型人数对不上时，**不给**逐人位置（宁可不编）。

        检测框会漏检被挡住的人，也会把屏幕/倒影里的人算成真人 ——
        这时按"序号 → 人物"配位置就是把位置安到错误的人头上。
        """
        from types import SimpleNamespace

        from engine.app.pipeline import to_ir

        def report(model_people: int, cv_count: int):
            return SimpleNamespace(
                shots=[
                    SimpleNamespace(
                        people_count=cv_count,
                        blocking=[SimpleNamespace(x_ratio=0.29, side="left")],
                    )
                ],
                understandings=[
                    SimpleNamespace(people=["x"] * model_people, blocking="没有任何位置描述")
                ],
            )

        people = ["甲人物", "乙人物"]
        # 人数一致 → 可以用框兜底
        assert to_ir._person_positions(report(1, 1), people[:1])[0] == "画面左侧"
        # 人数不一致 → 一个字都不给
        assert to_ir._person_positions(report(2, 1), people) == ["", ""]

    def test_rig_is_decided_by_majority_of_shots(self):
        """整片机位按**多数镜头**投票，不是"第一个能判的镜头"说了算。

        实测踩过：三镜头片子（镜1 跟拍 / 镜2 固定 / 镜3 固定）写成
        「斯坦尼康跟拍」—— 全片只占 1/3，还挂着镜3"判定固定机位"的证据。
        """
        from types import SimpleNamespace

        from engine.app.pipeline.infer import vote_rig

        def shot(index: int, kind: str | None):
            return SimpleNamespace(
                boundary=SimpleNamespace(index=index),
                motion=MotionEstimate(kind=kind, confidence=0.9, note="") if kind else None,
            )

        signals = [shot(1, "pan_or_track"), shot(2, "fixed"), shot(3, "fixed")]
        rig, source, index = vote_rig(signals, [])
        assert rig == "固定机位全程不变", rig
        assert index == 2, "代表性镜头应取多数票里最早的那一镜"
        assert source == "CV"
        # 一个镜头都给不出 → 空
        assert vote_rig([shot(1, None)], []) == ("", "", None)

    def test_compact_naming_drops_decoration_and_normalizes_color(self):
        """命名规则：颜色+服装+性别，丢掉发型/配饰，浅蓝归一成蓝色。"""
        from engine.app.pipeline.to_ir import compact_person_name

        assert compact_person_name("紫色连衣裙长卷发女子") == "紫色连衣裙女子"
        assert compact_person_name("紫色背心扎马尾女子") == "紫色背心女子"
        assert compact_person_name("白色衬衫女子（灰白色短卷发、玳瑁色墨镜）") == "白色衬衫女子"
        assert compact_person_name("浅蓝运动装女子") == "蓝色运动装女子"
        # 双服装时取第一件（上衣才是主体）
        assert compact_person_name("灰色上衣深色长裤女子") == "灰色上衣女子"
        # 真名不动
        assert compact_person_name("王姐") == "王姐"
        assert compact_person_name("魏巍老师") == "魏巍老师"

    def test_appearance_only_labels_still_work(self):
        from engine.app.pipeline.to_ir import dedupe_person_labels

        names = dedupe_person_labels(["紫色上衣女子", "红衣男子"])
        assert sorted(names) == sorted(["紫色上衣女子", "红色衣服男子"])

    def test_no_person_entry_when_only_position_labels(self):
        visual = _StubVisual([
            ShotUnderstanding(shot_index=1, people=("左侧人物", "中央人物")),
            ShotUnderstanding(shot_index=2, people=("左侧人物",)),
        ])
        # 隔离 ASR 那一路：只说"有人说话"，不给说话人
        report = _inferencer(
            visual=visual,
            transcriber=_StubTranscriber([
                TranscriptSegment(text="大家好", span=Span(0, 2000), confidence=0.9),
            ]),
        ).infer("fake.mp4")
        ir = Inferencer.to_ir(report)

        assert ir["locked"]["B2_person"] == [], "位置词不该变成人物条目"
        fake = [r for r in ir["references"] if "多视图" in r["placeholder"]]
        assert fake == [], f"不该为位置词派生多视图素材位：{fake}"

    def test_asr_speaker_still_wins(self):
        """ASR 说话人是"谁在说"的直接证据，不受位置词过滤影响。"""
        visual = _StubVisual([ShotUnderstanding(shot_index=1, people=("左侧人物",))])
        report = _inferencer(visual=visual).infer("fake.mp4")  # 默认 ASR 给「王姐/李姐」
        ir = Inferencer.to_ir(report)
        assert [p["name"] for p in ir["locked"]["B2_person"]] == ["王姐", "李姐"]

    def test_no_person_means_no_blink_or_skin_module(self):
        """连带后果要说清：眨眼节拍与皮肤质感句挂在人物条目下面，一起消失。"""
        from engine.app.pipeline.to_ir import unnamed_person_note

        visual = _StubVisual([ShotUnderstanding(shot_index=1, people=("左侧人物",))])
        report = _inferencer(
            visual=visual,
            transcriber=_StubTranscriber([
                TranscriptSegment(text="大家好", span=Span(0, 2000), confidence=0.9),
            ]),
        ).infer("fake.mp4")

        note = unnamed_person_note(report)
        assert note, "有人却认不出身份时必须给出说明，不能静默丢内容"
        assert "眨眼节拍" in note and "皮肤质感" in note

    def test_note_reaches_the_report_as_degradation(self):
        """说明必须真的出现在反推报告里，否则界面上看不到。"""
        visual = _StubVisual([ShotUnderstanding(shot_index=1, people=("左侧人物",))])
        _, report = _inferencer(
            visual=visual,
            transcriber=_StubTranscriber([
                TranscriptSegment(text="大家好", span=Span(0, 2000), confidence=0.9),
            ]),
        ).infer_to_ir("fake.mp4")

        joined = " ".join(report.degradations)
        assert "无法确定身份" in joined, f"降级说明里没有这条：{report.degradations}"

    def test_note_is_empty_when_nobody_seen(self):
        from engine.app.pipeline.to_ir import unnamed_person_note

        report = _inferencer().infer("fake.mp4")
        assert unnamed_person_note(report) == ""

    def test_note_is_empty_when_identities_resolved(self):
        from engine.app.pipeline.to_ir import unnamed_person_note

        visual = _StubVisual([ShotUnderstanding(shot_index=1, people=("紫衣女子",))])
        report = _inferencer(visual=visual).infer("fake.mp4")
        assert unnamed_person_note(report) == ""


class TestPersonFromVideoContent:
    """人物身份、逐人位置、画面人数都必须来自**这条视频的实际内容**。

    用户 2026-09-29 拿一条新套路的片子（演播厅双人普拉提示范）反馈：
    「明明是两个人物，男的在前，女的在后，却未能识别」。
    实测当时产出的是「人物1〔紫色衣服人物 · 画面左侧〕/人物2〔黑色短裤人物〕」
    和「画面中固定 1 位人物出镜」—— 位置安错人、人数还自相矛盾。
    """

    @staticmethod
    def _report():
        visual = _StubVisual([
            ShotUnderstanding(shot_index=1, framing="全景", camera="固定",
                              people=("红衣黑色短裤者", "浅紫色运动服扎发髻者"),
                              blocking="红衣黑色短裤者位于画面偏左，浅紫色运动服扎发髻者位于其右后方"),
            ShotUnderstanding(shot_index=2, framing="全景", camera="固定",
                              people=("红衣黑色短裤者", "浅紫色运动服扎发髻者"),
                              blocking="红衣者位于画面偏左，浅紫色运动服者位于其右后方"),
        ])
        # 隔离出"人物"这一条链路：静音（不跑 ASR）、不注入人体检测器
        return _inferencer(
            visual=visual,
            probe=_StubProbe(replace(_info(), mean_volume_db=-91.0)),
            transcriber=_StubTranscriber([]),
        ).infer("fake.mp4")

    def test_two_people_are_named_by_what_they_wear(self):
        ir = Inferencer.to_ir(self._report())
        names = [p["name"] for p in ir["locked"]["B2_person"]]
        assert len(names) == 2, names
        assert "红色衣服人物" in names, f"红上衣的人被写成了别的：{names}"
        assert "紫色运动服人物" in names, names

    def test_positions_come_from_what_the_model_saw(self):
        """逐人位置要用模型逐镜头说的左右/前后，而不是检测框的序号。"""
        ir = Inferencer.to_ir(self._report())
        by_name = {p["name"]: p["position"] for p in ir["locked"]["B2_person"]}
        assert by_name["红色衣服人物"].startswith("画面左"), by_name
        assert by_name["紫色运动服人物"].startswith("画面右"), by_name
        assert "后" in by_name["紫色运动服人物"], f"女士在后方，位置没说清：{by_name}"

    def test_arrangement_never_undercounts_the_people_list(self):
        """人数不能少于人物清单 —— 检测框漏检不是"画面里没人"。"""
        ir = Inferencer.to_ir(self._report())
        positions = "、".join(ir["locked"]["B3_blocking"]["positions"])
        assert "固定 2 位人物出镜" in positions, positions
        assert "固定 1 位人物出镜" not in positions, positions

    def test_arrangement_follows_the_model_over_the_detector(self):
        """模型说 2 人、检出 1 人 → 写 2；模型说 1 人、检出 3 人 → 写 1。

        两个方向都踩过：检出会**少检**（后排被挡住）也会**多检**
        （背景大屏上印的人物照片被当成真人）。
        """
        from types import SimpleNamespace

        from engine.app.pipeline import to_ir

        def report(cv: list[int], model: int):
            return SimpleNamespace(
                shots=[SimpleNamespace(people_count=n) for n in cv],
                understandings=[SimpleNamespace(people=["x"] * model, blocking="")] * len(cv),
            )

        assert to_ir._arrangement_sentence(
            report([1, 1, 2, 1], 2)
        ) == "画面中固定 2 位人物出镜，不做增减"
        assert to_ir._arrangement_sentence(
            report([3], 1)
        ) == "画面中固定 1 位人物出镜，不做增减"
        # 模型一个身份都给不出时才退回检出数
        assert to_ir._arrangement_sentence(
            report([2, 2], 0)
        ) == "画面中固定 2 位人物出镜，不做增减"


# ==========================================================================
# 编排：CV 优先与冲突处置
# ==========================================================================
class _StubProbe:
    def __init__(self, info: VideoInfo) -> None:
        self.info = info
        self.frames: list[str] = []

    def probe(self, path: str) -> VideoInfo:
        return self.info

    def extract_audio(self, path: str, out_path: str) -> str:
        return out_path

    def extract_keyframes(self, path, shots, out_dir) -> list[str]:
        self.frames = [f"{out_dir}/f{i}.jpg" for i in range(len(shots))]
        return self.frames


class _StubDetector:
    def __init__(self, shots: list[ShotBoundary]) -> None:
        self.shots = shots

    def detect(self, path, info, *, on_progress=None) -> list[ShotBoundary]:
        return self.shots


class TestNullJsonFields:
    """结构化输出里的字段可能是 ``null``，**不能因此崩掉整段理解**。

    实测踩过（留出集里一条新套路片子暴露的）：
    模型返回 ``{"shots": null}``，而代码写的是 ``raw.get("shots", [])`` ——
    ``None`` 被当成可迭代对象，TypeError 直接把**整段视觉理解**降级丢掉：
    「视觉理解失败，景别/机位降级为 CV 或留空：'NoneType' object is not iterable」。
    """

    def test_null_shots_does_not_raise(self):
        from engine.app.providers.mimo_analysis import MimoVisualDescriber
        from engine.core.contracts import ShotBoundary

        describer = MimoVisualDescriber.__new__(MimoVisualDescriber)

        class _Client:
            def complete_json(self, **_kwargs):
                return {"shots": None}

        describer.client = _Client()
        shots = [
            ShotBoundary(index=1, start_ms=0, end_ms=1000),
            ShotBoundary(index=2, start_ms=1000, end_ms=2000),
        ]
        out = describer.describe_shots(["a.jpg", "b.jpg"], shots)
        # 一个镜头一条空记录（后续走"视觉理解全空"的降级说明），而不是抛异常
        assert len(out) == 2
        assert all(not item.description for item in out)

    def test_null_edits_and_probes_do_not_raise(self):
        from engine.app.providers.mimo_analysis import MimoPromptCritic

        class _Client:
            def complete_json(self, **_kwargs):
                return {"edits": None, "probes": None}

        critic = MimoPromptCritic.__new__(MimoPromptCritic)
        critic.client = _Client()
        outcome = critic.critique(prompt="p", evidence={})
        assert outcome.edits == () and outcome.probes == ()


class _StubVisual:
    model_id = "stub-vision"

    def __init__(self, items: list[ShotUnderstanding]) -> None:
        self.items = items
        self.calls = 0

    def describe_shots(self, frames, shots, context="") -> list[ShotUnderstanding]:
        self.calls += 1
        return self.items


class _StubSemantic:
    model_id = "stub-text"

    def __init__(self, summary: SemanticSummary) -> None:
        self.summary = summary

    def induce(self, understandings, transcript=(), video=None) -> SemanticSummary:
        return self.summary


class _StubTranscriber:
    def __init__(self, segments: list[TranscriptSegment]) -> None:
        self.segments = segments

    def transcribe(self, audio_path, model=None) -> list[TranscriptSegment]:
        return self.segments


class _StubMotion:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def analyze(self, path, shot) -> MotionEstimate:
        return MotionEstimate(kind=self.kind, confidence=0.95, note="stub")


def _info(duration: float = 9.0) -> VideoInfo:
    return VideoInfo(duration_sec=duration, width=480, height=854, fps=25.0,
                     has_audio=True)


def _semantic(**overrides) -> SemanticSummary:
    base = dict(
        logline="菜市场里两位阿姨聊健身",
        scene_type="菜市场",
        genre=Genre.DIALOGUE,
        shot_mode=ShotMode.MULTI_SHOT,
        hook_index=1,
        quality_preset=QualityPreset.P1_PHONE,
        blink_beat=BlinkBeat.FIVE,
        confidence=0.85,
    )
    base.update(overrides)
    return SemanticSummary(**base)


def _inferencer(**kwargs) -> Inferencer:
    settings = Settings(mimo_api_key="test-key")
    shots = [ShotBoundary(1, 0, 4500), ShotBoundary(2, 4500, 9000)]
    defaults = dict(
        settings=settings,
        probe=_StubProbe(_info()),
        detector=_StubDetector(shots),
        visual=_StubVisual([
            ShotUnderstanding(shot_index=1, framing="中景", camera="平视",
                              location="菜市场", blocking="抬手", emotion="惊讶"),
            ShotUnderstanding(shot_index=2, framing="中近景", camera="过肩",
                              location="菜市场", blocking="摆手", emotion="温和"),
        ]),
        semantic=_StubSemantic(_semantic()),
        transcriber=_StubTranscriber([
            TranscriptSegment(text="你这气色也太好了吧", span=Span(200, 3000),
                              speaker="王姐", confidence=0.93),
            TranscriptSegment(text="哪有啊", span=Span(4600, 6000),
                              speaker="李姐", confidence=0.91),
        ]),
    )
    defaults.update(kwargs)
    return Inferencer(**defaults)


class TestSilentAudio:
    """静音音轨不能送 ASR，更不能把编出来的东西当台词（实测踩过）。"""

    @staticmethod
    def _transcriber(record: list[str]) -> _StubTranscriber:
        class _Recording(_StubTranscriber):
            def transcribe(self, audio_path, model=None):
                record.append(audio_path)
                return super().transcribe(audio_path, model)

        return _Recording([
            TranscriptSegment(text="0, 1, 2, 3, 4, 5", span=Span(0, 9000),
                              speaker="", confidence=0.689),
        ])

    def test_silent_track_skips_asr(self):
        """平均响度 −91 dBFS（数字静音）→ 一次都不该调 ASR。"""
        called: list[str] = []
        report = _inferencer(
            probe=_StubProbe(replace(_info(), mean_volume_db=-91.0)),
            transcriber=self._transcriber(called),
        ).infer("fake.mp4")
        assert called == [], "静音音轨仍然调了 ASR"
        assert not report.transcript
        assert any("静音" in d for d in report.degradations), report.degradations

    def test_loud_track_still_runs_asr(self):
        called: list[str] = []
        report = _inferencer(
            probe=_StubProbe(replace(_info(), mean_volume_db=-18.0)),
            transcriber=self._transcriber(called),
        ).infer("fake.mp4")
        assert called, "有声音的音轨反而没调 ASR"

    def test_number_only_transcript_is_dropped(self):
        """响度过了阈值、但转写里一个汉字一个字母都没有 → 按无语音丢弃。"""
        report = _inferencer(
            probe=_StubProbe(replace(_info(), mean_volume_db=-30.0)),
            transcriber=self._transcriber([]),
        ).infer("fake.mp4")
        assert not report.transcript, "只含数字/标点的「转写」被当成了台词"
        assert any("没有任何文字" in d for d in report.degradations), report.degradations

    def test_speech_flag_needs_a_measured_level(self):
        """量不到响度时不擅自判哑 —— 交给 ASR 与输出兜底。"""
        assert _info().has_speech is True
        assert replace(_info(), mean_volume_db=-91.0).has_speech is False
        assert replace(_info(), has_audio=False, mean_volume_db=None).has_speech is False

    def test_no_speech_means_no_voice_reference(self):
        """静音片子不该派生音色素材位 —— 没有声音可以克隆。"""
        report = _inferencer(
            probe=_StubProbe(replace(_info(), mean_volume_db=-91.0)),
            transcriber=self._transcriber([]),
        ).infer("fake.mp4")
        ir = Inferencer.to_ir(report)
        assert ir["locked"]["B6_audio"] == []
        audio_slots = [r for r in ir["references"] if r["type"] == "audio"]
        assert audio_slots == [], f"静音片子派生出了音色素材位：{audio_slots}"


class TestInferencer:
    def test_full_pipeline_produces_report(self):
        report = _inferencer().infer("fake.mp4")
        assert report.video.duration_sec == 9.0
        assert len(report.shots) == 2
        assert len(report.understandings) == 2
        assert report.semantic is not None
        assert len(report.transcript) == 2
        assert "vision=stub-vision" in report.model_id

    def test_ffmpeg_duration_has_full_confidence(self):
        report = _inferencer().infer("fake.mp4")
        prov = report.provenance["meta.total_duration_sec"]
        assert prov.confidence == 1.0
        assert prov.basis is ConfidenceBasis.FFMPEG

    def test_cv_conflict_marks_and_resolves(self):
        """冲突时以 CV 为准，并把置信度降到冲突档。"""
        inferencer = _inferencer(motion_analyzer=_StubMotion("push_in"))
        report = inferencer.infer("fake.mp4")
        # 模型说平视/过肩，CV 说推近 —— 应当冲突
        assert report.conflicts
        conflict = report.conflicts[0]
        assert conflict.field == "B4_camera.rig"
        assert "以 CV 为准" in conflict.resolution
        assert conflict.confidence == pytest.approx(
            compute_confidence(ConfidenceBasis.VLM_CONFLICT,
                               consistency="conflict", evidence="timecode_and_frame")
        )

    def test_no_conflict_when_cv_and_model_agree(self):
        visual = _StubVisual([
            ShotUnderstanding(shot_index=1, framing="中景", camera="推近"),
            ShotUnderstanding(shot_index=2, framing="中近景", camera="推近"),
        ])
        inferencer = _inferencer(visual=visual, motion_analyzer=_StubMotion("push_in"))
        assert inferencer.infer("fake.mp4").conflicts == ()

    def test_visual_failure_degrades_not_crashes(self):
        class _Broken:
            model_id = "broken"

            def describe_shots(self, frames, shots, context=""):
                raise RuntimeError("模型超时")

        report = _inferencer(visual=_Broken()).infer("fake.mp4")
        assert any("视觉理解失败" in d for d in report.degradations)
        assert report.understandings == ()

    def test_asr_failure_degrades_not_crashes(self):
        class _Broken:
            def transcribe(self, audio_path, model=None):
                raise RuntimeError("404")

        report = _inferencer(transcriber=_Broken()).infer("fake.mp4")
        assert any("语音识别失败" in d for d in report.degradations)

    def test_cv_gaps_are_declared_in_degradations(self):
        """没有人体检测时，缺口必须如实登记 —— 那是置信度降档的依据。"""
        from engine.app.providers.cv import CvCapabilities

        inferencer = _inferencer()
        # 显式把能力表压成"只有运镜"，模拟没装 ultralytics 的环境
        inferencer.capabilities = CvCapabilities(
            motion_from_luma=True, framing=False, blocking=False, people_count=False
        )
        report = inferencer.infer("fake.mp4")
        assert any("CV 能力缺口" in d for d in report.degradations)
        assert any("景别" in d for d in report.degradations)

    def test_person_detector_upgrades_framing_to_cv(self):
        """有人体检测时，景别/站位/人数走 CV，置信度升到 CV 档(0.95)。

        这是 PRD §6.1.2 的核心主张：能算的绝不让模型猜。
        """
        from engine.app.providers.cv import PersonBox, YoloPersonDetector

        class StubDetector(YoloPersonDetector):
            def detect(self, frame_path):  # type: ignore[override]
                return [
                    PersonBox(0.30, 0.05, 0.70, 0.97, 0.92),  # 前景主体，高度 0.92
                    PersonBox(0.80, 0.50, 0.90, 0.61, 0.55),  # 背景路人
                ]

        inferencer = _inferencer(person_detector=StubDetector("stub"))
        report = inferencer.infer("fake.mp4")

        # 帧数与镜头数一致时才会逐镜头检测；替身探针给 2 帧、2 镜头
        assert report.shots
        first = report.shots[0]
        assert first.framing is not None
        assert first.framing.kind == "近景", "高度 0.92 应判近景"
        assert first.framing.confidence == pytest.approx(0.95 * 0.85 * 1.0)
        assert first.people_count == 1, "背景路人不计入出镜人数"
        assert first.blocking and first.blocking[0].side == "center"

    def test_framing_falls_back_to_model_without_detector(self):
        """没有人体检测时退回模型判断，置信度按 VLM_ALONE 记。"""
        inferencer = _inferencer()
        inferencer.person_detector = None
        report = inferencer.infer("fake.mp4")
        framing = report.shots[0].framing
        assert framing is not None
        assert framing.kind == "中景"
        assert framing.note == "CV 未覆盖，由模型单独判断"
        assert framing.confidence == pytest.approx(0.70 * 0.85 * 1.0)


# ==========================================================================
# IR 映射：不猜原则
# ==========================================================================
class TestToIr:
    def test_low_confidence_is_left_blank_not_guessed(self):
        """红线 7：低置信度宁可留空。"""
        report = _inferencer(transcriber=_StubTranscriber([
            TranscriptSegment(text="听不清的台词", span=Span(200, 3000),
                              speaker="王姐", confidence=0.42),
        ])).infer("fake.mp4")
        ir = Inferencer.to_ir(report)
        assert ir["creative"]["C2_beats"][0]["dialogue"]["line"] == ""
        assert ir["creative"]["C2_beats"][0]["dialogue"]["speaker"] == ""

    def test_confident_transcript_is_used(self):
        report = _inferencer().infer("fake.mp4")
        ir = Inferencer.to_ir(report)
        lines = [b["dialogue"]["line"] for b in ir["creative"]["C2_beats"]]
        assert "你这气色也太好了吧" in lines

    def test_duration_comes_from_ffprobe(self):
        report = _inferencer(probe=_StubProbe(_info(12.5))).infer("fake.mp4")
        assert Inferencer.to_ir(report)["meta"]["total_duration_sec"] == 12.5

    def test_unknown_scene_does_not_produce_garbage_sentence(self):
        report = _inferencer(semantic=_StubSemantic(_semantic(scene_type=""))).infer("fake.mp4")
        anchor = Inferencer.to_ir(report)["locked"]["B1_scene"]["anchor"]
        assert "未知" not in anchor
        assert "【场景参考图】" in anchor

    def test_scene_anchor_carries_camera_texture(self):
        """T-SCENE-1 规范里的 `{拍摄质感}` 占位符必须被填上。

        这条曾经整块漏掉：规范写了占位符，代码里既没填也没报错，
        于是"苹果手机拍摄质感"这个高频固定模块在 24 条反推里 0 条写出。
        """
        ir = Inferencer.to_ir(_inferencer().infer("fake.mp4"))
        anchor = ir["locked"]["B1_scene"]["anchor"]
        assert "{" not in anchor, f"锚定句里有没填的占位符：{anchor}"
        assert "苹果手机拍摄质感" in anchor, anchor

    @pytest.mark.parametrize("preset,expected", [
        ("P1", "苹果手机拍摄质感"),
        ("P2", "苹果手机前置摄像头原生自拍质感"),
        ("P3", "ARRI原生色彩"),
    ])
    def test_camera_texture_follows_quality_preset(self, preset, expected):
        from engine.app.pipeline.to_ir import _camera_texture

        assert _camera_texture({"locked": {"B5_quality": {"preset": preset}}}) == expected

    def test_camera_texture_falls_back_when_preset_unknown(self):
        from engine.app.pipeline.to_ir import _camera_texture

        assert _camera_texture({"locked": {"B5_quality": {"preset": "P9"}}}) == "真实相机拍摄质感"

    def test_unknown_placeholder_strings_are_cleaned(self):
        """模型把 unknown 写成中文"未知"时必须当空处理，不能当内容。"""
        visual = _StubVisual([
            ShotUnderstanding(shot_index=1, framing="未知", camera="unknown",
                              location="未知", blocking="无", emotion="无"),
            ShotUnderstanding(shot_index=2, framing="未知", camera="unknown"),
        ])
        report = _inferencer(visual=visual).infer("fake.mp4")
        beat = Inferencer.to_ir(report)["creative"]["C2_beats"][0]
        assert beat["framing"] == "unknown"
        assert beat["camera"] == "unknown"
        assert beat["location"] == ""
        assert beat["blocking"] == ""

    def test_forbidden_is_copied_not_inferred(self):
        """框架 §8：D 区禁止项不反推，直接套 D0 全库。"""
        report = _inferencer().infer("fake.mp4")
        ir = Inferencer.to_ir(report)
        assert ir["forbidden"]
        assert any(item["group"] == "D0" for item in ir["forbidden"])

    def test_hook_role_is_preserved(self):
        report = _inferencer().infer("fake.mp4")
        beats = Inferencer.to_ir(report)["creative"]["C2_beats"]
        assert beats[0]["beat_role"] == "hook"

    def test_delivery_strategy_follows_shot_mode(self):
        report = _inferencer().infer("fake.mp4")
        assert Inferencer.to_ir(report)["delivery"]["strategy"] == "S2"

        # 单镜头 + one_take → S1
        single = _inferencer(
            detector=_StubDetector([ShotBoundary(1, 0, 9000)]),
            semantic=_StubSemantic(_semantic(shot_mode=ShotMode.ONE_TAKE)),
        ).infer("fake.mp4")
        assert Inferencer.to_ir(single)["delivery"]["strategy"] == "S1"

    def test_reference_manifest_lists_materials(self):
        """Agent 只指名需要哪些素材，不生成素材。"""
        report = _inferencer().infer("fake.mp4")
        manifest = to_ir_module.reference_manifest(Inferencer.to_ir(report))
        kinds = {item["type"] for item in manifest}
        assert "scene" in kinds
        assert all(item["asset_id"] is None for item in manifest), "素材由上游绑定"

    def test_every_ir_field_exists_in_contract(self):
        """反推产出的每个业务字段都必须在契约里 —— 否则位置标注会漏掉它。

        ``framework_version`` 是版本标记，不是可编辑位置，故排除。
        """
        from engine.core import framework as fw

        ir = Inferencer.to_ir(_inferencer().infer("fake.mp4"))
        non_position_keys = {"framework_version"}
        for path, _ in fw.flatten(ir):
            if path.split(".")[0] in non_position_keys:
                continue
            assert fw.spec_for(path) is not None, f"{path} 不在框架契约中"


# ==========================================================================
# 装配
# ==========================================================================
class TestWiring:
    def test_build_inferencer_without_key_degrades(self):
        """没有 Key 时不报错，降级为"无模型"模式。"""
        inferencer = build_inferencer(Settings(mimo_api_key=""), with_llm=True)
        assert inferencer.visual is None
        assert inferencer.semantic is None
        assert inferencer.probe is not None

    def test_capability_report_is_config_driven(self):
        report = Settings(mimo_api_key="k").capability_report()
        assert report["llm"]["configured"] is True
        assert "vision_model" in report["llm"]
