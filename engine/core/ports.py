"""Provider 端口：领域层与外部能力之间的接口。

PRD §6.1.3 说"分层后每层可单独测量、替换、回归"。可替换的前提就是**接口稳定**。

    Port（本文件）       靠 Protocol 定义，只声明"能做什么"
    Adapter（app/providers）  具体实现：MiMo / ffmpeg / 本地 CV 算法 / 测试替身

领域层只依赖 Port。因此：
- 换模型只改 Adapter，不动架构（PRD F4.9）；
- 回归测试可以注入确定性替身，不受模型漂移影响。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import (
    BlockingEstimate,
    CritiqueEdit,
    CritiqueOutcome,
    DialogueTurn,
    FramingEstimate,
    MotionEstimate,
    ProbeFinding,
    ProbeRequest,
    RevisionOutcome,
    SemanticSummary,
    ShotBoundary,
    ShotDialogueContext,
    ShotUnderstanding,
    TranscriptSegment,
    VideoInfo,
)


@runtime_checkable
class VideoProbe(Protocol):
    """L0：技术解析。**不靠模型判断**（PRD §6.1.2 第一层）。"""

    def probe(self, path: str) -> VideoInfo: ...

    def extract_audio(self, path: str, out_path: str) -> str:
        """抽音轨，供 ASR 使用。"""
        ...

    def extract_keyframes(
        self,
        path: str,
        shots: Sequence[ShotBoundary],
        out_dir: str,
    ) -> list[str]:
        """逐镜头抽关键帧，返回图片路径列表。"""
        ...


@runtime_checkable
class ShotDetector(Protocol):
    """镜头切分。目标：切点误差 ≤±2 帧。"""

    def detect(
        self,
        path: str,
        info: VideoInfo,
        *,
        on_progress: Any = None,
    ) -> list[ShotBoundary]: ...


@runtime_checkable
class MotionAnalyzer(Protocol):
    """运镜判断（光流分析）。目标 ≥90%。"""

    def analyze(self, path: str, shot: ShotBoundary) -> MotionEstimate: ...


@runtime_checkable
class FramingAnalyzer(Protocol):
    """景别判断（人体框占画面比例）。目标 ≥85%。"""

    def analyze(self, path: str, shot: ShotBoundary) -> FramingEstimate: ...


@runtime_checkable
class BlockingAnalyzer(Protocol):
    """站位判断（检测框中心 x vs 画面中线）。目标 ≥95%。"""

    def analyze(
        self,
        path: str,
        shot: ShotBoundary,
    ) -> tuple[tuple[BlockingEstimate, ...], int, float]:
        """返回 (站位列表, 人数, 置信度)。"""
        ...


@runtime_checkable
class VisualDescriber(Protocol):
    """L1：多模态视觉理解。必须支持高密度抽帧与结构化输出。"""

    def describe_shots(
        self,
        frames: Sequence[str],
        shots: Sequence[ShotBoundary],
        context: str = "",
    ) -> list[ShotUnderstanding]: ...


@runtime_checkable
class SemanticInducer(Protocol):
    """L2：语义归纳（情绪、节拍功能、场景类型、结构族）。"""

    def induce(
        self,
        understandings: Sequence[ShotUnderstanding],
        transcript: Sequence[TranscriptSegment] = (),
        video: VideoInfo | None = None,
    ) -> SemanticSummary: ...


@runtime_checkable
class DialogueAttributor(Protocol):
    """台词归属：把整段转写切成逐句台词，并判定**每句是谁说的、在哪个镜头**。

    为什么必须有这一环（实测依据）：

        中台封装的 ASR 是**纯转写网关** —— 不接受提示词、不返回时间戳、
        不做声纹分离（三种尝试分别返回 400 / 500 / 无字段）。
        所以一条双人对话视频拿回来就是一整段没有说话人的文本。

        没有这一环的后果（都复现过）：
          - 双人对话被压成**一句话**，全片只有最后一个镜头有台词；
          - 说话人只能靠"全片只有一个角色"推导，双人时直接留空；
          - 用户看到「人物1」和「人物2」却不知道哪句是谁说的。
    """

    def attribute(
        self,
        transcript: str,
        shots: Sequence[ShotDialogueContext],
    ) -> list[DialogueTurn]: ...


@runtime_checkable
class PromptCritic(Protocol):
    """自检回路：把**自己写好的交付物**连同画面证据再看一遍。

    返回 :class:`CritiqueOutcome` —— 里面既有**要改什么**（字段级修改），
    也有**还要看什么**（``ProbeRequest``）。后者是"看不懂时去查"的入口：
    模型没把握时应该要求取证，而不是硬猜一个修改。

    它只提建议，能不能改由 ``engine/core/review.py`` 按契约白名单判定 ——
    所以它越界不会造成破坏，只会被记账。
    """

    def critique(
        self,
        *,
        prompt: str,
        evidence: Mapping[str, Any],
    ) -> CritiqueOutcome: ...


@runtime_checkable
class PromptReviser(Protocol):
    """用户意见改写（PRD F8.4「按建议重写」）：把用户的一句话意见变成**字段级修改**。

    与 :class:`PromptCritic` 的区别是**授权来源**：

    - 自检问"我写错了吗" → 只能碰描述性文字（量测结果与用户原文不许改）；
    - 改写问"**你要什么**" → 授权来自用户本人，所以它可以增删人物/镜头、
      改站位、改台词、调景别与运镜。

    但它只**提**修改：能不能落、怎么落，由 ``engine/core/revision.py``
    按契约与可改范围判定；真正的红线（编造素材绑定、改交付形态）照样拒绝并记账。
    """

    def revise(
        self,
        *,
        instruction: str,
        ir: Mapping[str, Any],
        prompt: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> RevisionOutcome: ...


@runtime_checkable
class VisionProbe(Protocol):
    """按需取证：给时间点与问题，抽帧（可裁剪放大）后让视觉模型回答。

    为什么必须有它（用户 2026-09-30：「最主要是让他先是个 agent，有能力反推」）：
    原来的视觉理解是**每个镜头固定抽 2~3 张等间隔帧、一次性描述完就结束**。
    遇到看不清的地方（屏上小字、产品细节、谁在说话、动作关键帧）只能留空。
    有了这个端口，Agent 才能"自己决定再看哪儿"。
    """

    def look(self, video_path: str, request: ProbeRequest) -> ProbeFinding: ...

    def describe_shot(
        self,
        video_path: str,
        shot: ShotBoundary,
        missing: Sequence[str],
    ) -> dict[str, str]:
        """对**单个镜头**重新描述一遍，只回填 ``missing`` 里的字段。

        与 :meth:`look` 的区别：``look`` 回答一个自由问题（用于自检时解惑），
        这里要求**结构化字段**（用于补齐残缺镜头）——
        留出集实测里出现过"4 个镜头有 3 个只剩「全景。」"，靠它补。
        返回的键只允许是 ``location/blocking/expression/emotion/framing/...``，
        看不清就留空，**绝不猜**。
        """
        ...


@runtime_checkable
class Transcriber(Protocol):
    """ASR：词级时间戳。目标准确率 ≥90%。"""

    def transcribe(
        self,
        audio_path: str,
        model: str | None = None,
    ) -> list[TranscriptSegment]: ...


@runtime_checkable
class StructuredChat(Protocol):
    """结构化对话的底层能力。上层 Provider 用它，业务不直接调。"""

    model_id: str

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        images: Sequence[str] = (),
        temperature: float = 0.2,
    ) -> Mapping[str, Any]: ...

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        images: Sequence[str] = (),
        temperature: float = 0.2,
    ) -> str: ...


@runtime_checkable
class ObjectDetector(Protocol):
    """人体/人脸检测。**可选能力**：没有就降级，不阻塞主链路。

    没有实现时，景别与站位由多模态模型判断，置信度自动降一档（VLM_ALONE）。
    """

    @property
    def available(self) -> bool: ...

    def detect_persons(
        self,
        frame_path: str,
    ) -> list[tuple[float, float, float, float]]:
        """返回 (x1, y1, x2, y2)，归一化到 0~1。"""
        ...
