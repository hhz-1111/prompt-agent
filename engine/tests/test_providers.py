"""Provider 层与反推编排的测试。

这里用**确定性替身**替代真实模型调用：

- 网络调用不可复现，测它等于测网络；
- 但要测的是**编排逻辑**（CV 优先、冲突处置、置信度分档、不猜原则），
  这些完全可以用替身精确验证。

真实调用的连通性由 ``scripts/smoke_providers.py`` 负责，不进单测。
"""

from __future__ import annotations

from pathlib import Path

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
        frames = self._frames([100.0] * 50)
        result = cv_layer.analyze_motion(frames, shot)
        assert result.kind == "fixed"
        assert "固定机位" in result.note

    def test_monotonic_sequence_is_push_in(self):
        shot = ShotBoundary(1, 0, 2000)
        frames = self._frames([100.0 + i * 3 for i in range(50)])
        result = cv_layer.analyze_motion(frames, shot)
        assert result.kind == "push_in"

    def test_alternating_sequence_is_handheld(self):
        shot = ShotBoundary(1, 0, 2000)
        values = [100.0 + (8 if i % 2 else -8) for i in range(50)]
        result = cv_layer.analyze_motion(frames=self._frames(values), shot=shot)
        assert result.kind == "handheld"

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
