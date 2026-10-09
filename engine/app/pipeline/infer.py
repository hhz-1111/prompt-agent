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

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from engine.core.contracts import (
    BlockingEstimate,
    Conflict,
    DialogueTurn,
    FramingEstimate,
    InferenceReport,
    SemanticSummary,
    ShotBoundary,
    ShotDialogueContext,
    ShotSignals,
    ShotUnderstanding,
    TranscriptSegment,
    VideoInfo,
)
from engine.core.ports import (
    DialogueAttributor,
    MotionAnalyzer,
    PromptCritic,
    SemanticInducer,
    ShotDetector,
    Transcriber,
    VisionProbe,
    VideoProbe,
    VisualDescriber,
)
from engine.core.types import (
    UNKNOWN,
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

MAX_VIDEO_SECONDS = 180.0
"""素材时长上限（PRD F1.1：≤3min）。

为什么在反推阶段硬拦：超长视频不光是慢 —— 它意味着**用户拿错素材了**。
爆款复刻的目标素材通常 7~30 秒；3 分钟以上的东西多半是成片而非可复刻的爆款。
早失败比跑完 5 分钟再报错好。
"""

# 运镜：CV 结论 → 框架 rig 白名单
RIG_BY_MOTION: dict[str, str] = {
    "fixed": "固定机位全程不变",
    "push_in": "稳定推近",
    "pan_or_track": "斯坦尼康跟拍",
    "handheld": "手持自拍",
}

# 模型机位表述 → 框架 rig 白名单。CV 给不出结论时用它兜底。
RIG_BY_MODEL_CAMERA: dict[str, str] = {
    "固定": "固定机位全程不变",
    "推近": "稳定推近",
    "跟拍": "斯坦尼康跟拍",
    "横摇": "斯坦尼康跟拍",
    "手持": "手持自拍",
    "平视": "固定平视",
    "正面": "固定平视",
    "过肩": "过肩正反打",
    "俯拍": "固定平视",
    "仰拍": "固定平视",
}


def resolve_rig(
    motion: "MotionEstimate | None",
    understanding: "ShotUnderstanding | None",
) -> tuple[str, str]:
    """决定这条片子的机位档位，返回 ``(rig, 来源说明)``。

    优先级：**CV 的确定性结论 > 模型的语义判断 > 留空**。

    为什么要这个兜底：CV 的亮度序列分析**分不清"镜头动"与"被摄体动"**，
    所以在真实内容上它经常只能返回 ``unknown``（这是刻意的保守，
    见 ``providers/cv.py``）。此时不能让机位字段空着 —— 那是**高影响级**
    维度，模型给的"固定/推近/跟拍"虽然不如 CV 硬，但远好过留空。
    """
    if motion is not None:
        rig = RIG_BY_MOTION.get(motion.kind, "")
        if rig:
            return rig, "CV"
    if understanding is not None:
        rig = RIG_BY_MODEL_CAMERA.get(understanding.camera, "")
        if rig:
            return rig, "模型"
    return "", ""


def vote_rig(
    signals: Sequence[ShotSignals],
    understandings: Sequence[ShotUnderstanding] = (),
) -> tuple[str, str, int | None]:
    """整片机位档位：**按多数镜头投票**，不是取第一个镜头。

    返回 ``(rig, 来源, 代表性镜头序号)``；一个镜头都给不出就 ``("", "", None)``。

    为什么改成投票（实测踩过）：`locked.B4_camera.rig` 是**单值**字段，
    原来是"第一个能给出结论的镜头"说了算 —— 一条三镜头片子
    （镜1 跟拍 / 镜2 固定 / 镜3 固定）会写成「斯坦尼康跟拍」，
    而它在全片只占 1/3，还挂着镜3"判定固定机位"的证据，自相矛盾。
    多数票至少代表"这条片子主要怎么拍"，且与逐镜头节拍不冲突。
    """
    by_index = {u.shot_index: u for u in understandings}
    votes: dict[str, list[tuple[str, int]]] = {}
    for signal in signals:
        rig, source = resolve_rig(signal.motion, by_index.get(signal.boundary.index))
        if rig:
            votes.setdefault(rig, []).append((source, signal.boundary.index))
    if not votes:
        return "", "", None
    # 票多者胜；同票取**更早出现**的那个（保证同样输入给同样结果）
    ordered = sorted(votes.items(), key=lambda item: (-len(item[1]), min(i for _, i in item[1])))
    rig, hits = ordered[0]
    source, index = min(hits, key=lambda pair: pair[1])
    return rig, source, index


def _noop(stage: str, progress: float) -> None:  # pragma: no cover - 默认回调
    return None


_LETTER_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")
"""转写里至少要有一个汉字或字母，才算"它其实说了点什么"。"""


def _is_blank_understanding(understanding: ShotUnderstanding) -> bool:
    """这个镜头的视觉理解是不是"什么都没认出来"。"""
    if understanding.people:
        return False
    fields = (
        understanding.description, understanding.framing, understanding.camera_angle,
        understanding.camera, understanding.location, understanding.expression,
        understanding.emotion, understanding.blocking,
    )
    return all(not str(value or "").strip() or str(value).strip() == UNKNOWN for value in fields)


def _speech_only(
    segments: Iterable[TranscriptSegment],
    degradations: list[str],
) -> list[TranscriptSegment]:
    """丢掉**明显不是话**的转写结果。

    这是静音闸门之外的第二道防线：有些片子音轨不是数字静音（有底噪、有 BGM），
    ``volumedetect`` 量出来过得了阈值，但里面确实没人说话 —— ASR 仍会编东西出来。
    实测那次编的是「0, 1, 2, …, 100.」：**一个汉字一个字母都没有**。

    判据刻意取到最保守（连一个字母/汉字都没有才丢），
    所以不会误伤正常的短句、数字混排（「2026年」有汉字）或外语台词。
    """
    kept: list[TranscriptSegment] = []
    dropped = 0
    for segment in segments:
        if _LETTER_RE.search(segment.text or ""):
            kept.append(segment)
        else:
            dropped += 1
    if dropped:
        degradations.append(
            f"{dropped} 段语音识别结果里没有任何文字（只有数字/标点），"
            f"按无语音丢弃，避免把噪声当成台词"
        )
    return kept


@dataclass(slots=True)
class Inferencer:
    """反推编排器。所有依赖都可注入，便于回归测试与替换。"""

    settings: Settings
    probe: VideoProbe
    detector: ShotDetector
    visual: VisualDescriber | None = None
    semantic: SemanticInducer | None = None
    transcriber: Transcriber | None = None
    dialogue_attributor: DialogueAttributor | None = None
    """台词归属（可选但强烈建议）：ASR 不给说话人/时间戳时，靠它把整段切成逐句。"""
    motion_analyzer: MotionAnalyzer | None = None
    person_detector: cv_layer.YoloPersonDetector | None = None
    """人体检测（可选增强）。有它就把景别/站位/人数升级为 CV 强判断。"""
    critic: PromptCritic | None = None
    """自检回路（可选）。有它就会在交付前"回头看一遍自己写的东西"。

    开关在 ``Settings.review_rounds``（默认 1 轮，0 关闭）。没有它时链路退回
    **单向管线** —— 语义问题只能靠确定性闸门兜，实测兜不住。
    """
    vision_probe: VisionProbe | None = None
    """按需取证（可选）。有它时自检回路可以要求"再看一眼某个时间点"。

    开关在 ``Settings.probe_budget``（默认 3 次，0 关闭）。
    这是"看不懂就去查"的能力 —— 没有它，回路只能在"已看到的东西"里改。
    **注意与上面 ``probe``（ffprobe 技术解析端口）不是一回事。**
    """
    library: Any = None
    """自检回路要重新组装交付物，需要短语库；不注入就按默认资产装一次。"""

    # 中间产物：回归测试与 UI 诊断都要看
    _frame_means: list[FrameMean] = field(default_factory=list, init=False)
    _perception_notes: list[Any] = field(default_factory=list, init=False)
    """感知补看的记账（ReviewNote）。由 ``infer_to_ir`` 合并进 report.reviews，
    这样"Agent 补看过哪些镜头"和自检回路的记账一起显示在结果页。"""
    composition: dict[int, str] = field(default_factory=dict, init=False)
    """逐镜头的构图偏侧（left/center/right）。逐人站位见 ``ShotSignals.blocking``。"""
    capabilities: cv_layer.CvCapabilities = field(
        default_factory=cv_layer.detect_capabilities, init=False
    )

    def __post_init__(self) -> None:
        """能力盘点必须反映**实际注入的检测器**，而不是本机装了什么。

        否则：注入了替身检测器的测试会看到"环境有 YOLO"的能力表，
        而实际跑的却是替身 —— 能力和实现脱节，降级判断就会出错。
        """
        if self.library is None:
            from engine.core import phrases as _ph

            self.library = _ph.load_library()
        self.capabilities = replace(
            self.capabilities,
            framing=self.person_detector is not None or self.capabilities.framing,
            blocking=self.person_detector is not None or self.capabilities.blocking,
            people_count=self.person_detector is not None or self.capabilities.people_count,
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
        if info.duration_sec > MAX_VIDEO_SECONDS:
            raise ValueError(
                f"视频 {info.duration_sec:.1f} 秒超过 {MAX_VIDEO_SECONDS / 60:.0f} 分钟上限"
                f"（PRD F1.1）。爆款复刻的素材通常很短，超长视频请先裁剪。"
            )
        if info.duration_sec <= 0:
            raise ValueError("ffprobe 读到的时长为 0，文件可能损坏或不是有效视频")

        # ---- 步骤 2：四路信号 ----
        progress("镜头切分", 0.15)
        shots = self.detector.detect(video_path, info)
        if not shots:
            shots = [ShotBoundary(index=1, start_ms=0, end_ms=int(info.duration_sec * 1000))]

        self.composition = {}

        # 光流分析器一次解码整段、批量算完所有镜头，所以要先把视频与镜头表交给它
        binder = getattr(self.motion_analyzer, "bind", None)
        if callable(binder):
            binder(video_path, shots)

        progress("运动与画面特征", 0.30)
        self._frame_means = self._collect_frame_means(video_path)

        progress("抽取关键帧", 0.40)
        frames = self.probe.extract_keyframes(video_path, shots, str(work / "frames"))
        lost_frames = getattr(self.probe, "last_frame_failures", [])
        if lost_frames:
            degradations.append(
                f"{len(lost_frames)} 个镜头抽帧失败（多为过短镜头或定位点越界），"
                f"这些镜头将缺少视觉描述：{'；'.join(lost_frames[:3])}"
            )
        if len(frames) < len(shots):
            # 帧数与镜头数不一致时无法逐镜头对齐，如实降级而不是硬凑
            degradations.append(
                f"关键帧数 {len(frames)} 少于镜头数 {len(shots)}，"
                f"视觉理解将只覆盖抽到帧的镜头"
            )

        transcript: list[TranscriptSegment] = []
        if info.has_audio and not info.has_speech:
            # **静音音轨不能送 ASR**：网关会在静音上凭空编出一整段"转写"，
            # 实测拿回来的是「0, 1, 2, …, 100.」，然后被当作真实台词
            # 写进【对白与口型】并派生出【音色参考】。宁可不转写。
            degradations.append(
                f"音轨是静音的（平均响度 {info.mean_volume_db:g} dBFS），"
                f"已跳过语音识别；台词留空待用户填写"
            )
        elif info.has_speech and self.transcriber is not None:
            progress("语音识别", 0.50)
            try:
                audio = self.probe.extract_audio(video_path, str(work / "audio.wav"))
                transcript = _speech_only(self.transcriber.transcribe(audio), degradations)
            except Exception as exc:  # ASR 失败不该拖垮整条链路
                degradations.append(f"语音识别失败，台词留空待用户填写：{exc}")

        # ---- 步骤 3：视觉理解 ----
        # 帧数必须与镜头数一致才能逐镜头标注。不一致时**不硬凑** ——
        # 错位标注比没有标注更糟，会让用户以为某个镜头的景别就是它旁边那个的。
        understandings: list[ShotUnderstanding] = []
        if self.visual is not None and frames and len(frames) == len(shots):
            progress("视觉理解", 0.60)
            try:
                understandings = list(self.visual.describe_shots(frames, shots))
            except Exception as exc:
                degradations.append(f"视觉理解失败，景别/机位降级为 CV 或留空：{exc}")
        elif self.visual is None:
            degradations.append("未配置视觉模型，景别/机位/情绪只能依赖 CV 与留空")
        elif frames and len(frames) != len(shots):
            degradations.append(
                f"关键帧({len(frames)})与镜头({len(shots)})数量不一致，"
                f"为避免错位标注，本次不做视觉理解"
            )
        if understandings and all(_is_blank_understanding(u) for u in understandings):
            # 模型跑了、但什么都没认出来（实测：一条纯色块素材两个镜头全 unknown）。
            # 不写这条说明的话，用户只会看到交付物缺了【镜头节拍】与人物，
            # 分不清是"画面本身没内容"还是"系统坏了"。
            degradations.append(
                "视觉理解没有产出任何内容（镜头描述与人物均为空）——"
                "景别/机位/情绪/人物只能留空。请确认视频画面是否可辨认"
                "（纯色、黑屏、严重模糊的素材会这样）"
            )

        # ---- 步骤 3.5：感知覆盖度自检 + 定向补看 ----
        # 留出集实测（框架外的片子）：4 个镜头里有 3 个只拿到「全景。」，
        # 场景/动作/表情全空 —— 而且**没有任何环节发现这件事**。
        # 这一步就是"发现自己没看清谁，然后回去补"，补在语义归纳之前，
        # 这样剧情/场景类型的归纳也能用上补回来的信息。
        if understandings:
            from .perception import fill_perception_gaps, gap_summary, is_thin

            budget = int(getattr(self.settings, "perception_budget", 0) or 0)
            thin_before = sum(1 for u in understandings if is_thin(u))
            if thin_before and budget > 0 and self.vision_probe is not None:
                progress("定向补看", 0.70)
            understandings = fill_perception_gaps(
                understandings, shots, self.vision_probe, video_path,
                budget=budget, notes=self._perception_notes, degradations=degradations,
            )
            if thin_before > budget > 0:
                degradations.append(
                    f"有 {thin_before} 个镜头画面信息残缺，本次只补看了 {budget} 个"
                    f"（预算上限，可在设置里调 PERCEPTION_BUDGET）"
                )
            note = gap_summary(understandings)
            if note:
                degradations.append(note)

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
        signals, conflicts = self._fuse(shots, understandings, frames)

        # ---- 步骤 5.2：致命维度多源裁决 ----
        # 人数这种致命级维度上，检测框与模型**各自都会错**（屏上照片被算成真人 /
        # 后排被挡住的人漏检），原来只记一条"以某方为准"的分歧就完了。
        # 这里补第三票：对那个镜头抽帧，问一句**只回答数字**的问题。
        conflicts = self._arbitrate(conflicts, shots, video_path, degradations)

        # ---- 步骤 5.5：台词归属（切句 + 说话人 + 所属镜头） ----
        # 放在这里而不是更早：它需要**画面里实际有谁**（从 understandings 来），
        # 也需要已经切好的镜头表。ASR 不给说话人与时间戳时，
        # 没有这一步双人对话会被压成一句话、且全片只有最后一个镜头有台词。
        dialogue_turns: list[DialogueTurn] = []
        joined = " ".join(t.text for t in transcript if t.text.strip()).strip()
        if joined and self.dialogue_attributor is not None:
            progress("台词归属", 0.92)
            contexts = [
                ShotDialogueContext(
                    shot_index=signal.boundary.index,
                    duration_sec=signal.boundary.duration_sec,
                    framing=signal.framing.kind if signal.framing else "",
                    camera_angle=(
                        by_index_camera_angle(understandings, signal.boundary.index)
                    ),
                    location=location_of(understandings, signal.boundary.index),
                    people=people_of(understandings, signal.boundary.index),
                    action=action_of(understandings, signal.boundary.index),
                    expression=expression_of(understandings, signal.boundary.index),
                )
                for signal in signals
            ]
            try:
                dialogue_turns = list(self.dialogue_attributor.attribute(joined, contexts))
            except Exception as exc:  # noqa: BLE001
                degradations.append(
                    f"台词归属失败，台词将只落在重叠最多的那个镜头：{exc}"
                )

            # **归属失败/返回空，绝不能把转写整段丢掉**（实测踩过）：
            # 收紧超时之后，模型一次调用可能超时（flash 实测能跑到 236 秒），
            # 那时 ``attribute`` 返回空 → 交付物里一句台词都没有、
            # 真实存在的转写被静默扔掉，对齐测试因此挂掉。
            # 转写本身是**真实数据**，兜底成"整段挂在第一个镜头、说话人留空"，
            # 并如实说明（红线 5 管的是"不编"，不是"不许留空"）。
            if joined and not dialogue_turns and contexts:
                from .attribution import fallback_turn

                dialogue_turns = [fallback_turn(joined, contexts[0].shot_index)]
                degradations.append(
                    "台词归属没成功，已把整段转写挂在第一个镜头、说话人留空"
                    "（台词内容没丢，但『谁说的』需要人工确认）"
                )

            # ---- 归属覆盖度自检：判不出说话人的，回去取证再判一次 ----
            # 留出集实测：双人示范片 4 句台词只判出第 1 句的说话人，
            # 交付物里出现「镜头2（无说话人）」，G6 因此不通过。
            from .attribution import (
                attribution_summary,
                collect_speaker_evidence,
                merge_speakers,
                unspoken_turns,
            )

            if dialogue_turns and unspoken_turns(dialogue_turns):
                budget = int(getattr(self.settings, "attribution_budget", 0) or 0)
                voters = _known_speakers(contexts)
                evidence = collect_speaker_evidence(
                    dialogue_turns, shots, contexts, self.vision_probe, video_path,
                    budget=budget, notes=self._perception_notes, degradations=degradations,
                )
                if evidence:
                    dialogue_turns = merge_speakers(dialogue_turns, evidence, voters)
                note = attribution_summary(dialogue_turns)
                if note:
                    degradations.append(note)
        if joined and self.dialogue_attributor is None:
            degradations.append(
                "未配置台词归属模型：双人对话会被压成一段、说话人留空。"
            )

        # ---- 步骤 6：置信度分级已内联在 provenance ----
        progress("汇总", 0.95)
        provenance = self._build_provenance(
            info, signals, semantic, transcript, conflicts, understandings
        )

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
            dialogue_turns=tuple(dialogue_turns),
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
        # 感知补看的记账先落地（它们发生在 infer 里，但要和自检记账一起给用户看）
        if self._perception_notes:
            report.reviews = tuple(self._perception_notes)
        ir = self.to_ir(report)
        # 人物认不出身份时，交付物会连带丢掉眨眼节拍与皮肤质感两条固定模块
        # （它们挂在人物条目下面）。这是**反推阶段的降级**，归属地就是这里。
        if not ir.get("locked", {}).get("B2_person"):
            from .to_ir import unnamed_person_note

            note = unnamed_person_note(report)
            if note:
                report.degradations = tuple(report.degradations) + (note,)

        # 人物结构是"模型判断的"还是"代码推断的"：推断的要如实说
        # （用户 2026-10-08 追问"有没有自适应" —— 自适应靠模型判断，
        #  猜出来的就得让用户知道这是猜的，并且一句话能改）。
        from .to_ir import people_structure_note

        structure_note = people_structure_note(ir, report)
        if structure_note:
            report.degradations = tuple(report.degradations) + (structure_note,)

        # ---- 自检-重写回路（把管线变成 agent 的那一步）----
        # 放在这里而不是"组装时"：改的是 **IR**，所以结果会被持久化，
        # 之后每次查看/导出看到的都是同一版，不会出现"看一眼变一次"。
        rounds = int(getattr(self.settings, "review_rounds", 0) or 0)
        if rounds > 0 and self.critic is not None:
            if on_progress:
                on_progress("自检与重写", 0.97)
            from .review import run_review

            ir, notes, extra = run_review(
                ir, report, self.critic, self.library, rounds=rounds,
                probe=self.vision_probe, video_path=video_path,
                probe_budget=int(getattr(self.settings, "probe_budget", 0) or 0),
                # 时间预算 + 内部子进度：这一阶段以前只有一个静态的 0.97，
                # 模型想很久时用户看到的就是"卡死"（用户 2026-10-07 报的）。
                budget_sec=float(getattr(self.settings, "review_budget_sec", 0) or 0),
                on_progress=on_progress,
            )
            if notes:
                report.reviews = tuple(report.reviews) + notes
            if extra:
                report.degradations = tuple(report.degradations) + extra
        return ir, report

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

    def _arbitrate(
        self,
        conflicts: Sequence[Conflict],
        shots: Sequence[ShotBoundary],
        video_path: str,
        degradations: list[str],
    ) -> list[Conflict]:
        """对致命维度的分歧做一次取证裁决（没有取证能力时原样返回）。"""
        from .arbitration import arbitrate_conflicts, critical_conflicts

        budget = int(getattr(self.settings, "probe_budget", 0) or 0)
        if not critical_conflicts(conflicts) or budget <= 0 or self.vision_probe is None:
            return list(conflicts)
        return arbitrate_conflicts(
            conflicts, shots, self.vision_probe, video_path,
            budget=budget, notes=self._perception_notes, degradations=degradations,
        )

    def _fuse(
        self,
        shots: Sequence[ShotBoundary],
        understandings: Sequence[ShotUnderstanding],
        frames: Sequence[str] = (),
    ) -> tuple[list[ShotSignals], list[Conflict]]:
        """CV 与模型结论融合。**冲突时以 CV 为准并降置信度**。

        人体检测可用时，景别/站位/人数走 CV；不可用时退回模型判断，
        并按 ``VLM_ALONE``(0.70) 记置信度 —— 差距是实打实的，不粉饰。
        """
        by_index = {u.shot_index: u for u in understandings}
        # 关键帧与镜头一一对应时才能逐镜头检测；不一致就不做，避免错位
        frame_by_shot = (
            dict(zip((s.index for s in shots), frames))
            if len(frames) == len(shots) else {}
        )
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
                    if not _camera_agrees(understanding.camera, motion.kind):
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

            framing, blocking, people = self._analyze_person(shot, frame_by_shot, understanding)

            # 景别与人数**也要交叉校验**（原来只比运镜）。
            # 实测踩过：5 个真实项目的 conflicts 全是空的，而同一批素材里
            # 景别有 3/11 个镜头 CV 与模型判得不一样、人数出现 CV=3 而模型=1 ——
            # 分歧一条都没进报告，"降置信度并标黄"这两个维度上等于没做。
            if understanding is not None:
                model_framing = str(understanding.framing or "").strip()
                if (
                    framing is not None
                    and framing.kind not in ("", UNKNOWN)
                    and model_framing not in ("", UNKNOWN)
                    and framing.kind != model_framing
                ):
                    conflicts.append(Conflict(
                        field="creative.C2_beats[].framing",
                        shot_index=shot.index,
                        cv_value=framing.kind,
                        model_value=model_framing,
                        resolution="以 CV 为准（人体框是几何量测）",
                        confidence=compute_confidence(
                            ConfidenceBasis.VLM_CONFLICT,
                            consistency="conflict",
                            evidence="timecode_and_frame",
                        ),
                    ))
                if people > 0 and understanding.people and people != len(understanding.people):
                    conflicts.append(Conflict(
                        field="creative.C2_beats[].people_count",
                        shot_index=shot.index,
                        cv_value=f"{people} 人（人体框）",
                        model_value=f"{len(understanding.people)} 人（{'、'.join(understanding.people)}）",
                        resolution="以模型为准（检出会漏掉被遮挡的人，也会把屏幕上的人算进来）",
                        confidence=compute_confidence(
                            ConfidenceBasis.VLM_CONFLICT,
                            consistency="conflict",
                            evidence="timecode_and_frame",
                        ),
                    ))

            signals.append(ShotSignals(
                boundary=shot,
                motion=motion,
                framing=framing,
                people_count=people,
                people_confidence=(
                    compute_confidence(
                        ConfidenceBasis.CV, consistency="single",
                        evidence="timecode_and_frame",
                    ) if self.person_detector is not None else 0.0
                ),
                blocking=blocking,
                composition=self.composition.get(shot.index, ""),
            ))
        return signals, conflicts

    def _analyze_person(
        self,
        shot: ShotBoundary,
        frame_by_shot: Mapping[int, str],
        understanding: ShotUnderstanding | None,
    ) -> tuple[
        FramingEstimate | None,
        tuple[BlockingEstimate, ...],
        int,
    ]:
        """景别 / 站位 / 人数：优先 CV，退路是模型判断。

        ``CV > 模型`` 不是随便定的：人体框是几何量测，模型是语义判断。
        景别与站位在 PRD 里分别是"中"和"致命"影响级维度，
        有确定算法就必须用确定算法。

        副产物：把**构图偏侧**记到 ``self.composition``。
        逐人的左右（``blocking``）与"整个画面居中不居中"是两个问题，
        合并成一个数会让"一左一右的双人镜头"被判成偏侧。
        """
        frame = frame_by_shot.get(shot.index)
        if self.person_detector is not None and frame:
            try:
                boxes = self.person_detector.detect(frame)
            except Exception:
                boxes = []
            if boxes:
                blocking, count, _ = cv_layer.blocking_from_boxes(boxes)
                composition, _, _ = cv_layer.composition_side(boxes)
                self.composition[shot.index] = composition
                primary = max(boxes, key=lambda b: b.height_ratio)
                return cv_layer.framing_from_person(primary), blocking, count
            # 检测不到人也要区分"画面里真没人"与"检测失败"
            return None, (), 0

        if understanding is not None and understanding.framing not in ("", "unknown"):
            return (
                FramingEstimate(
                    kind=understanding.framing,
                    confidence=compute_confidence(
                        ConfidenceBasis.VLM_ALONE, consistency="single",
                        evidence="timecode_and_frame",
                    ),
                    note="CV 未覆盖，由模型单独判断",
                ),
                (),
                0,
            )
        return None, (), 0

    def _build_provenance(
        self,
        info: VideoInfo,
        signals: Sequence[ShotSignals],
        semantic: SemanticSummary | None,
        transcript: Sequence[TranscriptSegment],
        conflicts: Sequence[Conflict],
        understandings: Sequence[ShotUnderstanding] = (),
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
        # 冲突要**按字段**分开：运镜冲突写机位，景别冲突写景别，
        # 否则一条景别冲突会把机位的证据串改成"运镜不一致"。
        camera_conflicts = {c.shot_index for c in conflicts if c.field == "B4_camera.rig"}
        framing_conflicts = {c.shot_index for c in conflicts if c.field.endswith(".framing")}
        # 机位档位（``locked.B4_camera.rig``）是**单值**，取法与 to_ir._fill_camera 一致：
        # 多数镜头投票，证据挂**代表性镜头**（票数最多、且最早出现的那一镜）。
        # 原来在循环里往同一个键反复写，最终只剩最后一个镜头的证据 ——
        # 于是「斯坦尼康跟拍」这个结论挂着镜3"判定固定机位"的证据（实测踩到过）。
        rig_value, rig_source, rig_index = vote_rig(signals, understandings)
        rig_prov: Provenance | None = None
        by_index = {u.shot_index: u for u in understandings}
        for signals_item in signals:
            shot = signals_item.boundary
            prefix = f"creative.C2_beats[{shot.index - 1}]"
            camera_prov: Provenance | None = None

            if signals_item.motion is not None:
                basis = (
                    ConfidenceBasis.CV
                    if self.capabilities.motion_from_luma
                    else ConfidenceBasis.VLM_ALONE
                )
                note = signals_item.motion.note
                if shot.index == rig_index:
                    note = f"{note}（整片机位取值来源：{rig_source}）"
                camera_prov = Provenance(
                    confidence=signals_item.motion.confidence,
                    basis=basis,
                    span=shot.span,
                    note=note,
                    evidence=f"帧间运动分析 @{shot.span.format_timecode()}",
                )
                if shot.index == rig_index:
                    rig_prov = camera_prov
            if shot.index in camera_conflicts:
                camera_prov = Provenance(
                    confidence=compute_confidence(
                        ConfidenceBasis.VLM_CONFLICT,
                        consistency="conflict",
                        evidence="timecode_and_frame",
                    ),
                    basis=ConfidenceBasis.VLM_CONFLICT,
                    span=shot.span,
                    note="CV 与模型对运镜判断不一致，已以 CV 为准并标黄",
                )
            if camera_prov is not None:
                prov[f"{prefix}.camera"] = camera_prov

            if signals_item.framing is not None:
                prov[f"{prefix}.framing"] = Provenance(
                    confidence=signals_item.framing.confidence,
                    basis=ConfidenceBasis.CV if self.capabilities.framing
                    else ConfidenceBasis.VLM_ALONE,
                    span=shot.span,
                    note=signals_item.framing.note,
                )
                if shot.index in framing_conflicts:
                    prov[f"{prefix}.framing"] = Provenance(
                        confidence=compute_confidence(
                            ConfidenceBasis.VLM_CONFLICT,
                            consistency="conflict",
                            evidence="timecode_and_frame",
                        ),
                        basis=ConfidenceBasis.VLM_CONFLICT,
                        span=shot.span,
                        note="CV 与模型对景别判断不一致，已以 CV 为准并标黄",
                    )
        if rig_prov is not None:
            prov["locked.B4_camera.rig"] = rig_prov

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


def by_index_camera_angle(
    understandings: Sequence[ShotUnderstanding], shot_index: int
) -> str:
    for item in understandings:
        if item.shot_index == shot_index:
            return item.camera_angle
    return ""


def location_of(
    understandings: Sequence[ShotUnderstanding], shot_index: int
) -> str:
    for item in understandings:
        if item.shot_index == shot_index:
            return item.location
    return ""


def _known_speakers(contexts: Sequence[ShotDialogueContext]) -> list[str]:
    """候选说话人：画面里出现过的人（按出现顺序去重）。

    归属取证的回答是自由文本，要从里面认出"是哪个人物"，
    候选集合就是画面里实际出现过的那些标签 —— **只在这些里面认**，
    否则会认出一个画面里根本没有的人。
    """
    seen: list[str] = []
    for context in contexts:
        for name in context.people:
            text = str(name).strip()
            if text and text not in seen:
                seen.append(text)
    return seen


def action_of(
    understandings: Sequence[ShotUnderstanding], shot_index: int
) -> str:
    """这个镜头的动作描述 —— 台词归属要靠它认出「谁张口说话」。"""
    for item in understandings:
        if item.shot_index == shot_index:
            return item.blocking
    return ""


def expression_of(
    understandings: Sequence[ShotUnderstanding], shot_index: int
) -> str:
    for item in understandings:
        if item.shot_index == shot_index:
            return item.expression
    return ""


def people_of(
    understandings: Sequence[ShotUnderstanding], shot_index: int
) -> tuple[str, ...]:
    """这个镜头里**实际出现**的人（原始标签，未去重）。

    台词归属要的是"这个镜头里看得见谁"，所以给未去重的原始标签 ——
    去重是跨镜头合并用的，会把两个穿同色衣服的人并成一个。
    """
    for item in understandings:
        if item.shot_index == shot_index:
            return item.people
    return ()


def build_inferencer(
    settings: Settings | None = None,
    *,
    with_llm: bool = True,
) -> Inferencer:
    """按配置组装默认编排器。

    两处**能力降级**都在这里决定，不在业务代码里散落：

    - 缺 API Key → 无模型模式（仍能跑 L0 技术层）；
    - 缺 PySceneDetect → 退回 ffmpeg 逐帧均值切分（灵敏度会下降）。
    """
    from ..providers.mimo_analysis import (
        MimoDialogueAttributor,
        MimoPromptCritic,
        MimoSemanticInducer,
        MimoTranscriber,
        MimoVisionProbe,
        MimoVisualDescriber,
    )
    from ..providers.optical_flow import build_motion_analyzer
    from ..providers.scene_detect import SceneDetectShots

    resolved = settings or Settings.load()
    media = FFmpegMedia(resolved)

    detector: ShotDetector = media
    scene_shots = SceneDetectShots(resolved)
    if scene_shots.available:
        detector = scene_shots

    # 人体检测是可选增强：装了 ultralytics 就把景别/站位升为 CV 强判断
    person_detector = None
    capabilities = cv_layer.detect_capabilities()
    if capabilities.person_detector:
        person_detector = cv_layer.YoloPersonDetector(
            capabilities.person_detector, workspace=resolved.workspace_dir / "models"
        )

    # 运镜升级为光流强判断：装了 opencv 才启用，否则退回亮度法（保守判定）
    motion_analyzer = build_motion_analyzer(resolved.ffmpeg_bin)

    use_llm = with_llm and resolved.has_llm
    # 自检回路：有模型 + 轮数 > 0 才开。它是"agent 回头看自己"的那一步，
    # 关掉时链路退回单向管线（结果仍在，只是没人复核）。
    reviewing = use_llm and int(getattr(resolved, "review_rounds", 0) or 0) > 0
    critic = MimoPromptCritic(resolved) if reviewing else None
    # 按需取证：只在回路开着、且预算 > 0 时给 —— 它是"看不懂就去查"的入口，
    # 每次取证要抽帧 + 一次视觉调用，所以要能单独关掉。
    probe = (
        MimoVisionProbe(resolved, media)
        if reviewing and int(getattr(resolved, "probe_budget", 0) or 0) > 0
        else None
    )
    return Inferencer(
        settings=resolved,
        probe=media,
        detector=detector,
        visual=MimoVisualDescriber(resolved) if use_llm else None,
        semantic=MimoSemanticInducer(resolved) if use_llm else None,
        transcriber=MimoTranscriber(resolved) if use_llm else None,
        dialogue_attributor=MimoDialogueAttributor(resolved) if use_llm else None,
        motion_analyzer=motion_analyzer,
        person_detector=person_detector,
        critic=critic,
        vision_probe=probe,
    )
