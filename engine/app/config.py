"""运行时配置。

**密钥只从环境变量或 .env 读取，绝不写进代码。**
.env 已在 .gitignore 里；仓库只提交 .env.example。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = ROOT / ".env"


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """读 .env。已存在的环境变量优先，不被文件覆盖。"""
    target = path or DEFAULT_ENV_FILE
    values: dict[str, str] = {}
    if not target.exists():
        return values
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


@dataclass(frozen=True, slots=True)
class Settings:
    """全部可调项集中在这里，不散落在各处。"""

    mimo_api_key: str = ""
    mimo_base_url: str = "https://api.xiaomimimo.com/v1"
    mimo_batch_base_url: str = "https://batch-api-cn.xiaomimimo.com/v1"

    # 分工（PRD §6.1.3 分层）
    model_vision: str = "mimo-v2.6-flash"   # L1 视觉理解
    model_text: str = "mimo-v2.6-flash"     # L2 语义归纳 + L3 结构化
    model_asr: str = "mimo-v2.5-asr"        # L0 语音识别

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    workspace_dir: Path = field(default_factory=lambda: ROOT / "workspace")
    request_timeout_sec: float = 180.0
    max_retries: int = 2

    # 抽帧密度：PRD 要求"支持高密度抽帧（时序）"
    frames_per_shot: int = 3
    # 切点判定阈值：64x64 缩放后逐帧色差（BT.601 加权、归一化）。
    # 实测校准：硬切 0.40~0.50；同场景内的明显色变约 0.15；轻微变化 <0.05。
    # 调低会更敏感（快速摇镜可能被误判成切镜），调高会漏切。
    scene_threshold: float = 0.10

    @property
    def has_llm(self) -> bool:
        return bool(self.mimo_api_key)

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        file_values = load_env_file(env_file)

        def pick(key: str, default: str) -> str:
            return os.environ.get(key) or file_values.get(key) or default

        return cls(
            mimo_api_key=pick("MIMO_API_KEY", ""),
            mimo_base_url=pick("MIMO_BASE_URL", cls.mimo_base_url),
            mimo_batch_base_url=pick("MIMO_BATCH_BASE_URL", cls.mimo_batch_base_url),
            model_vision=pick("MODEL_VISION", cls.model_vision),
            model_text=pick("MODEL_TEXT", cls.model_text),
            model_asr=pick("MODEL_ASR", cls.model_asr),
            ffmpeg_bin=pick("FFMPEG_BIN", cls.ffmpeg_bin),
            ffprobe_bin=pick("FFPROBE_BIN", cls.ffprobe_bin),
            workspace_dir=Path(pick("WORKSPACE_DIR", str(cls.workspace_dir))),
            request_timeout_sec=float(pick("REQUEST_TIMEOUT_SEC", "180")),
            max_retries=int(pick("MAX_RETRIES", "2")),
            frames_per_shot=int(pick("FRAMES_PER_SHOT", "3")),
            scene_threshold=float(pick("SCENE_THRESHOLD", "0.10")),
        )

    def capability_report(self) -> dict[str, object]:
        """能力矩阵来自配置，不硬编码（框架 §2.3）。"""
        return {
            "llm": {
                "configured": self.has_llm,
                "base_url": self.mimo_base_url,
                "vision_model": self.model_vision,
                "text_model": self.model_text,
                "asr_model": self.model_asr,
            },
            "media": {
                "ffmpeg": self.ffmpeg_bin,
                "ffprobe": self.ffprobe_bin,
            },
            "frame_sampling": {
                "frames_per_shot": self.frames_per_shot,
                "scene_threshold": self.scene_threshold,
            },
        }
