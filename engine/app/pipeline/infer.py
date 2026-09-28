"""反推编排：六步工作流（AGENT §3）。

    步骤1 技术解析 —— ffprobe，不靠模型
    步骤2 四路信号 —— 切分 / 运镜 / 景别站位 / ASR
    步骤3 视觉理解 —— 逐镜头抽帧送多模态
    步骤4 结构归纳 —— 场景类型、镜头模式、钩子位置
    步骤5 交叉校验 —— **CV 优先**，冲突降到 0.6 并标黄
    步骤6 置信度处置 —— <0.50 留空，绝不猜

每一步都单独可测、可替换 —— 这是 PRD §6.1.3「分层后每层可单独测量」
能落地的前提。本模块只做编排，不做算法。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

from engine.core.contracts import (
    Conflict,
    InferenceReport,
    SemanticSummary,
    ShotBoundary,
    ShotSignals,
    ShotUnderstanding,
    TranscriptSegment,
    VideoInfo,
)
from engine.core.ports import (
    MotionAnalyzer,
    SemanticInducer,
    ShotDetector,
    Transcriber,
    VideoProbe,
    VisualDescriber,
)
from engine.core.types import (
    ConfidenceBasis,
    Provenance,
    ShotMode,
    Span,
    compute_confidence,
)

from ..config import Settings
from ..providers import cv as cv_layer
from ..providers.ffmpeg_media import FFmpegMedia, FrameMean, parse_frame_means

ProgressFn = Callable[[str, float], None]

# 运镜：CV 结论 → 框架 rig 白名单
RIG_BY_MOTION: dict[str, str] = {
    "fixed": "固定机位全程不变",
    "push_in": "稳定推近",
    "pan_or_track": "斯坦尼康跟拍",
    "handheld": "手持自拍",
}


def _noop(stage: str, progress: float) -> None:  # pragma: no cover - 默认回调
    return None


@dataclass(slots=True)
class Inferencer:
    """反推编排器。所有依赖都可注入，便于回归测试与替换。"""

    settings: Settings
    probe: VideoProbe
    detector: ShotDetector
    visual: VisualDescriber | None = None
    semantic: SemanticInducer | None = None
    transcriber: Transcriber | None = None
    motion_analyzer: MotionAnalyzer | None = None

    # 中间产物：回归测试与 UI 诊断都要看
    _frame_means: list[FrameMean] = field(default_factory=list, init=False)
    capabilities: cv_layer.CvCapabilities = field(
        default_factory=cv_layer.detect_capabilities, init=False
    )

    def infer(
        self,
        video_path: str,
        *,
        workspace: Path | None = None,
        on_progress: ProgressFn | None = None,
    ) -> InferenceReport:
        progress = on_progress or _noop
        work = Path(workspace or self.settings.workspace_dir) / "infer"
        work.mkdir(parents=True, exist_ok=True)
        degradations: list[str] = []

        # ---- 步骤 1：技术解析 ----
        progress("技术解析", 0.05)
        info = self.probe.probe(video_path)

        # ---- 步骤 2：四路信号 ----
        progress("镜头切分", 0.15)
        shots = self.detector.detect(video_path, info)
        if not shots:
            shots = [ShotBoundary(index=1, start_ms=0, end_ms=int(info.duration_sec * 1000))]

        progress("运动与画面特征", 0.30)
        self._frame_means = self._collect_frame_means(video_path)

        progress("抽取关键帧", 0.40)
        frames = self.probe.extract_keyframes(video_path, shots, str(work / "frames"))

        transcript: list[TranscriptSegment] = []
        if info.has_audio and self.transcriber is not None:
            progress("语音识别", 0.50)
            try:
                audio = self.probe.extract_audio(video_path, str(work / "audio.wav"))
                transcript = list(self.transcriber.transcribe(audio))
            except Exception as exc:  # ASR 失败不该拖垮整条链路
                degradations.append(f"语音识别失败，台词留空待用户填写：{exc}")

        # ---- 步骤 3：视觉理解 ----
        understandings: list[ShotUnderstanding] = []
        if self.visual is not None and frames:
            progress("视觉理解", 0.60)
            try:
                understandings = list(self.visual.describe_shots(frames, shots))
            except Exception as exc:
                degradations.append(f"视觉理解失败，景别/机位降级为 CV 或留空：{exc}")
        elif self.visual is None:
            degradations.append("未配置视觉模型，景别/机位/情绪只能依赖 CV 与留空")

        # ---- 步骤 4：结构归纳 ----
        semantic: SemanticSummary | None = None
        if self.semantic is not None and understandings:
            progress("结构归纳", 0.75)
            try:
                semantic = self.semantic.induce(understandings, transcript, info)
                semantic = reconcile_shot_mode(semantic, len(shots))
            except Exception as exc:
                degradations.append(f"结构归纳失败，场景类型与钩子位置待用户确认：{exc}")

        # ---- 步骤 5：交叉校验（CV 优先） ----
        progress("交叉校验", 0.88)
        signals, conflicts = self._fuse(shots, understandings)

        # ---- 步骤 6：置信度分级已内联在 provenance ----
        progress("汇总", 0.95)
        provenance = self._build_provenance(info, signals, semantic, transcript, conflicts)

        if self.capabilities.missing:
            degradations.append(
                "CV 能力缺口（这些维度当前由模型单独判断，置信度已降档）："
                + "、".join(self.capabilities.missing)
            )

        return InferenceReport(
            video=info,
            shots=tuple(signals),
            understandings=tuple(understandings),
            semantic=semantic,
            transcript=tuple(transcript),
            provenance=provenance,
            conflicts=tuple(conflicts),
            degradations=tuple(degradations),
            model_id=self._model_id(),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def to_ir(report: InferenceReport) -> dict:
        """把反推结果映射成 Framework IR。**低置信度一律留空**（红线 7）。"""
        from .to_ir import build_ir

        return build_ir(report)

    def infer_to_ir(
        self,
        video_path: str,
        *,
        workspace: Path | None = None,
        on_progress: ProgressFn | None = None,
    ) -> tuple[dict, InferenceReport]:
        report = self.infer(video_path, workspace=workspace, on_progress=on_progress)
        return self.to_ir(report), report

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _collect_frame_means(self, video_path: str) -> list[FrameMean]:
        media = self.probe
        if not isinstance(media, FFmpegMedia):
            return []
        stderr = media._run_text([
            self.settings.ffmpeg_bin,
            "-hide_banner", "-nostats",
            "-i", video_path,
            "-vf", "scale=64:64,showinfo",
            "-an", "-f", "null", "-",
        ])
        return parse_frame_means(stderr)

    def _fuse(
        self,
        shots: Sequence[ShotBoundary],
        understandings: Sequence[ShotUnderstanding],
    ) -> tuple[list[ShotSignals], list[Conflict]]:
        """CV 与模型结论融合。**冲突时以 CV 为准并降置信度**。"""
        by_index = {u.shot_index: u for u in understandings}
        conflicts: list[Conflict] = []
        signals: list[ShotSignals] = []

        for shot in shots:
            understanding = by_index.get(shot.index)

            # 运镜：CV 有能力就一定用 CV
            motion = None
            if self.motion_analyzer is not None:
                motion = self.motion_analyzer.analyze("", shot)
            elif self._frame_means:
                motion = cv_layer.analyze_motion(self._frame_means, shot)

            if motion is not None and understanding is not None:
                model_rig = RIG_BY_MOTION.get(motion.kind, "")
                if model_rig and understanding.camera not in ("", "unknown"):
                    agreed = _camera_agrees(understanding.camera, motion.kind)
                    if not agreed:
                        conflicts.append(Conflict(
                            field="B4_camera.rig",
                            shot_index=shot.index,
                            cv_value=motion.kind,
                            model_value=understanding.camera,
                            resolution="以 CV 为准（光流是确定性判断）",
                            confidence=compute_confidence(
                                ConfidenceBasis.VLM_CONFLICT,
                                consistency="conflict",
                                evidence="timecode_and_frame",
                            ),
                        ))

            # 景别与站位：CV 未覆盖时用模型值，置信度按"模型单独判断"计
            framing = None
            if understanding is not None and understanding.framing not in ("", "unknown"):
                from engine.core.contracts import FramingEstimate

                basis = (
                    ConfidenceBasis.CV
                    if self.capabilities.framing
                    else ConfidenceBasis.VLM_ALONE
                )
                framing = FramingEstimate(
                    kind=understanding.framing,
                    confidence=compute_confidence(
                        basis, consistency="single", evidence="timecode_and_frame"
                    ),
                    note="CV 未覆盖，由模型单独判断" if not self.capabilities.framing else "",
                )

            signals.append(ShotSignals(
                boundary=shot,
                motion=motion,
                framing=framing,
                people_count=0,
                people_confidence=0.0,
                blocking=(),
            ))
        return signals, conflicts

    def _build_provenance(
        self,
        info: VideoInfo,
        signals: Sequence[ShotSignals],
        semantic: SemanticSummary | None,
        transcript: Sequence[TranscriptSegment],
        conflicts: Sequence[Conflict],
    ) -> dict[str, Provenance]:
        from engine.core import framework as fw

        full = Span(0, int(info.duration_sec * 1000))
        prov: dict[str, Provenance] = {
            "meta.total_duration_sec": Provenance(
                confidence=1.0,
                basis=ConfidenceBasis.FFMPEG,
                span=full,
                note="ffprobe 直接测得，唯一可信来源",
            ),
            "meta.aspect_ratio": Provenance(
                confidence=1.0, basis=ConfidenceBasis.FFMPEG, span=full,
                note="由分辨率推导",
            ),
        }
        conflict_shots = {c.shot_index for c in conflicts}
        for signals_item in signals:
            shot = signals_item.boundary
            prefix = f"creative.C2_beats[{shot.index - 1}]"
            has_conflict = shot.index in conflict_shots

            if signals_item.motion is not None:
                basis = (
                    ConfidenceBasis.CV
                    if self.capabilities.motion_from_luma
                    else ConfidenceBasis.VLM_ALONE
                )
                prov[f"locked.B4_camera.rig"] = Provenance(
                    confidence=signals_item.motion.confidence,
                    basis=basis,
                    span=shot.span,
                    note=signals_item.motion.note,
                    evidence=f"帧间运动分析 @{shot.span.format_timecode()}",
                )
            if signals_item.framing is not None:
                prov[f"{prefix}.framing"] = Provenance(
                    confidence=signals_item.framing.confidence,
                    basis=ConfidenceBasis.CV if self.capabilities.framing
                    else ConfidenceBasis.VLM_ALONE,
                    span=shot.span,
                    note=signals_item.framing.note,
                )
            if has_conflict:
                prov[f"{prefix}.camera"] = Provenance(
                    confidence=compute_confidence(
                        ConfidenceBasis.VLM_CONFLICT,
                        consistency="conflict",
                        evidence="timecode_and_frame",
                    ),
                    basis=ConfidenceBasis.VLM_CONFLICT,
                    span=shot.span,
                    note="CV 与模型对运镜判断不一致，已以 CV 为准并标黄",
                )

        if semantic is not None:
            sem_span = Span(0, int(info.duration_sec * 1000))
            for path, value, note in (
                ("creative.C1_logline", semantic.logline, "由全片归纳"),
                ("meta.genre", semantic.genre.value, "由场景与人数归纳"),
                ("meta.shot_mode", semantic.shot_mode.value, "由镜头切换情况归纳"),
                ("locked.B1_scene.type", semantic.scene_type, "由视觉与场景特征归纳"),
                ("locked.B5_quality.preset", semantic.quality_preset.value, "由质感特征判断"),
                ("locked.B2_person[].blink", semantic.blink_beat.value, "按内容类型选档"),
            ):
                if not value:
                    continue
                if fw.spec_for(path) is None:
                    continue
                prov[path] = Provenance(
                    confidence=semantic.confidence,
                    basis=ConfidenceBasis.INFERRED,
                    span=sem_span,
                    note=note,
                )

        for segment in transcript:
            prov[f"transcript.{segment.span.start_ms}"] = Provenance(
                confidence=segment.confidence,
                basis=ConfidenceBasis.ASR_CLEAR,
                span=segment.span,
                note="ASR 识别结果",
            )
        return prov

    def _model_id(self) -> str:
        parts = []
        if self.visual is not None:
            parts.append(f"vision={getattr(self.visual, 'model_id', self.settings.model_vision)}")
        if self.semantic is not None:
            parts.append(f"text={getattr(self.semantic, 'model_id', self.settings.model_text)}")
        if self.transcriber is not None:
            parts.append(f"asr={self.settings.model_asr}")
        parts.append(f"cv={','.join(k for k, v in self.capabilities.to_dict().items() if v is True)}")
        return "; ".join(parts)


def _camera_agrees(model_camera: str, cv_kind: str) -> bool:
    """模型机位表述与 CV 结论是否相容。"""
    if cv_kind == "fixed":
        return model_camera in ("固定", "平视", "正面", "过肩", "俯拍", "仰拍")
    if cv_kind == "push_in":
        return model_camera in ("推近",)
    if cv_kind == "pan_or_track":
        return model_camera in ("跟拍", "横摇")
    if cv_kind == "handheld":
        return model_camera in ("手持",)
    return True


def reconcile_shot_mode(semantic: SemanticSummary, shot_count: int) -> SemanticSummary:
    """让模型判定的镜头模式与 CV 实际切分数**对齐**。

    为什么必须做：切分数是画面差分算出来的**确定事实**，
    模型没有推翻它的依据。而 shot_mode 是致命级维度 —— 它决定拆条策略
    与提示词条数。实测出现过"1 个镜头却判 multi_shot"的自相矛盾，
    会直接产出错误的交付结构。

    修正方向只有一个：向 CV 结果收敛。
    """
    if shot_count <= 1:
        # 只切出一个镜头，就不可能是"多镜头切分"。
        # 但 same_angle_segments 本质就是一个连续镜头按时间推进切段，
        # 与 one_take 的区别在**拆条方式**而非切分数，所以要保留。
        if semantic.shot_mode is ShotMode.MULTI_SHOT:
            return replace(semantic, shot_mode=ShotMode.ONE_TAKE, note="已按 CV 切分数校正")
        return semantic
    # 多镜头时，"一镜到底"站不住
    if semantic.shot_mode is ShotMode.ONE_TAKE:
        return replace(semantic, shot_mode=ShotMode.MULTI_SHOT, note="已按 CV 切分数校正")
    return semantic


def build_inferencer(
    settings: Settings | None = None,
    *,
    with_llm: bool = True,
) -> Inferencer:
    """按配置组装默认编排器。缺 Key 时自动降级为"无模型"模式。"""
    from ..providers.mimo_analysis import (
        MimoSemanticInducer,
        MimoTranscriber,
        MimoVisualDescriber,
    )

    resolved = settings or Settings.load()
    media = FFmpegMedia(resolved)
    use_llm = with_llm and resolved.has_llm
    return Inferencer(
        settings=resolved,
        probe=media,
        detector=media,
        visual=MimoVisualDescriber(resolved) if use_llm else None,
        semantic=MimoSemanticInducer(resolved) if use_llm else None,
        transcriber=MimoTranscriber(resolved) if use_llm else None,
        motion_analyzer=None,
    )
