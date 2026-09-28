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
    FramingEstimate,
    MotionEstimate,
    SemanticSummary,
    ShotBoundary,
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
