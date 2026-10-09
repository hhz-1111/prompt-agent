"""运行时配置。

**密钥只从环境变量或 .env 读取，绝不写进代码。**
.env 已在 .gitignore 里；仓库只提交 .env.example。
"""

from __future__ import annotations

import os
import dataclasses
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

    # 自检-重写回路：交付前让模型**回头看一遍自己写的东西**，最多几轮。
    # 1 = 自检一次（默认）；0 = 关闭，退回单向管线。每轮多一次模型调用。
    review_rounds: int = 1

    # 自检-重写的**总时间预算**（秒）。超了就停下并保留当前版本
    # （用户 2026-10-07 报"卡很久"：推理模型一次调用可能想很久，没预算会像卡死）。
    review_budget_sec: float = 480.0
    # 自检**单次调用**的超时与重试。**按实测定的**（不要凭感觉改）：
    # 实测一次真实自检调用 = **129 秒**（765 字交付物）；
    # 所以超时必须显著大于它，否则会把本来能成功的调用判成失败。
    # 重试设 0：Critic 自己还有"完整证据 → 最小证据"的一次降级重试，
    # 再叠 HTTP 重试就是乘法（180s×3 最坏 9 分半，正是用户看到的"卡很久"）。
    review_call_timeout_sec: float = 240.0
    review_call_retries: int = 0
    # 默认是否跑自检-重写。关掉=快速反推（少几次模型调用、快几分钟），
    # 代价是交付前没人回头看一遍。可以在提交反推时按次覆盖。
    review_enabled: bool = True
    # **自检专用模型**（用户 2026-10-07 批准"给自检换快模型"）。
    # 空 = 跟 model_text 一样。
    # 实测（同一个自检任务、同一份证据）：
    #   mimo-v2.6-pro-ultraspeed  41.5 秒，挑出 7 条修改
    #   mimo-v2.6-flash（原默认） 60.9 秒，挑出 1 条
    #   mimo-v2.5                328.5 秒，挑出 1 条
    # 所以默认换成"极速"那个 —— 又快又挑得多（一次样本，随时可用
    # MODEL_REVIEW= 清空退回）。
    model_review: str = "mimo-v2.6-pro-ultraspeed"
    # 取证（探针）单次调用的超时与重试。原来走全局默认（180 秒 × 3 次），
    # 结果自检的实际耗时**冲破了 480 秒预算**（实测 563.7 秒）。
    probe_call_timeout_sec: float = 120.0
    probe_call_retries: int = 0
    # 自检证据的**体积上限**（字符）。超了先砍"信息量最低"的镜头。
    # 注意实测结论：证据从 193 字到 2500 字，**主要耗时在模型思考而不是输入长度**
    # （193 字的证据也花了 60 秒）。所以这是个**防爆护栏**，不是提速主力。
    review_evidence_char_cap: int = 1200

    # ------------------------------------------------------------------
    # 用户意见改写（PRD F8.4「按建议重写」）
    # ------------------------------------------------------------------
    # **所有 key 都要可换端点**（用户 2026-10-08）：
    # REVISE_BASE_URL / REVISE_API_KEY 留空 → 跟 MiMo 用同一个端点与 Key；
    # 填了就用它（例如 DeepSeek 官方 https://api.deepseek.com/v1）。
    # 换模型只改 .env，不动一行代码。
    revise_base_url: str = ""
    revise_api_key: str = ""
    # 改写专用模型。空 = 跟 model_review（自检那个"极速"模型）。
    # 实测（2026-10-08，DeepSeek 官方端点）：deepseek-flash 做一次改写约 30~90 秒。
    model_revise: str = ""
    # 单次调用的超时与重试：改写是**同步接口**，用户在对话框前等着，
    # 不能像反推那样靠轮询掩盖延迟。所以超时要明确、重试设 0
    # （客户端自己有"JSON 修复"那次重试，再叠 HTTP 重试就是乘法）。
    revise_call_timeout_sec: float = 240.0
    revise_call_retries: int = 0
    # 改写后**要不要再跑一遍 14 项交付闸门**。默认 **False**。
    # 用户 2026-10-08 明确要求：「修改提示词不需要再重跑了，直接用 AI 改写提示词
    # 填回去就可以了」—— 所以改写就是**一次模型调用**：AI 改 → 落 IR →
    # 本地重新拼出正文（毫秒级，不是重跑）→ 填回文本框。
    #
    # 关掉闸门**不等于没有底线**：组装失败、或正文里留下没解析的【…】句柄
    # （红线 6）仍然会整体回滚 —— 那是"交付物坏了"，跟"闸门评分"是两件事。
    # 需要那套评分时把它打开（REVISE_CHECK_GATES=1），行为与自检回路一致：
    # 判定变差会如实告知，并让 AI 按闸门反馈再修一轮。
    revise_check_gates: bool = False
    # 保留最近几轮改写（含撤销用的 IR 快照）。
    revise_max_history: int = 10

    # 按需取证：自检时允许它要求"再看一眼某个时间点"的次数上限（抽帧+一次视觉调用）。
    # 3 = 默认；0 = 关闭（回路只能在已看到的东西里改）。
    probe_budget: int = 3

    # 感知补看：发现"某个镜头没看清"时回去重看该镜头的次数上限。
    # 4 = 默认；0 = 关闭。留出集实测：框架外片子里 4 个镜头有 3 个只剩「全景。」，
    # 以前**没有任何环节发现这件事**。
    perception_budget: int = 4

    # 台词归属补判：判不出"这句谁说的"时回去取证的镜头数上限。
    # 3 = 默认；0 = 关闭。
    attribution_budget: int = 3

    # 反推并发上限（用户 2026-10-07：未来给团队用，多人会同时上传）。
    # 一次反推要调十几次模型，不封顶会直接撞模型接口的限流 —— 谁都跑不完。
    max_concurrent_infer: int = 2
    # 排队上限。超过就回 503，而不是无限收任务。
    max_queued_infer: int = 20
    # 切点判定阈值：64x64 缩放后逐帧色差（BT.601 加权、归一化）。
    # 实测校准：硬切 0.40~0.50；同场景内的明显色变约 0.15；轻微变化 <0.05。
    # 调低会更敏感（快速摇镜可能被误判成切镜），调高会漏切。
    scene_threshold: float = 0.10

    @property
    def has_llm(self) -> bool:
        return bool(self.mimo_api_key)

    @property
    def revise_endpoint(self) -> tuple[str, str]:
        """改写用的 ``(base_url, api_key)``：**留空就退回 MiMo**。

        用户 2026-10-08：「所有 key，都要写成可换端点的，以便后续更换 AI，
        .env 填写后就能自动切」。所以这里是唯一的分叉点 ——
        换供应商只改 .env 里的 ``REVISE_BASE_URL`` / ``REVISE_API_KEY``。
        """
        return (self.revise_base_url or self.mimo_base_url,
                self.revise_api_key or self.mimo_api_key)

    @property
    def has_revise_llm(self) -> bool:
        return bool(self.revise_endpoint[1])

    @property
    def resolved_model_revise(self) -> str:
        """改写的实际模型名：``MODEL_REVISE`` 空则跟自检模型。"""
        return self.model_revise or self.model_review or self.model_text

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        file_values = load_env_file(env_file)

        def default_of(field_name: str) -> str:
            """取字段的**真默认值**。

            为什么不能写 ``cls.model_vision``：``Settings`` 是 ``slots=True`` 的
            数据类，类属性拿到的是**成员描述符**（``<member 'x' of 'Settings'>``），
            不是默认值。以前这些默认值恰好都被 ``.env`` 里的显式值掩盖了，
            新加字段（``model_review``）才把它暴露出来 —— 设置项会变成一个描述符对象。
            """
            for item in dataclasses.fields(cls):
                if item.name == field_name and item.default is not dataclasses.MISSING:
                    return str(item.default)
            return ""

        def pick(key: str, default: str) -> str:
            return os.environ.get(key) or file_values.get(key) or default

        return cls(
            mimo_api_key=pick("MIMO_API_KEY", ""),
            mimo_base_url=pick("MIMO_BASE_URL", default_of("mimo_base_url")),
            mimo_batch_base_url=pick("MIMO_BATCH_BASE_URL", default_of("mimo_batch_base_url")),
            model_vision=pick("MODEL_VISION", default_of("model_vision")),
            model_text=pick("MODEL_TEXT", default_of("model_text")),
            model_asr=pick("MODEL_ASR", default_of("model_asr")),
            ffmpeg_bin=pick("FFMPEG_BIN", default_of("ffmpeg_bin")),
            ffprobe_bin=pick("FFPROBE_BIN", default_of("ffprobe_bin")),
            workspace_dir=Path(pick("WORKSPACE_DIR", default_of("workspace_dir"))),
            request_timeout_sec=float(pick("REQUEST_TIMEOUT_SEC", "180")),
            max_retries=int(pick("MAX_RETRIES", "2")),
            frames_per_shot=int(pick("FRAMES_PER_SHOT", "3")),
            scene_threshold=float(pick("SCENE_THRESHOLD", "0.10")),
            review_rounds=int(pick("REVIEW_ROUNDS", "1")),
            review_budget_sec=float(pick("REVIEW_BUDGET_SEC", "480")),
            review_call_timeout_sec=float(pick("REVIEW_CALL_TIMEOUT_SEC", "240")),
            review_call_retries=int(pick("REVIEW_CALL_RETRIES", "0")),
            review_enabled=pick("REVIEW_ENABLED", "1") not in ("0", "false", "False"),
            model_review=pick("MODEL_REVIEW", default_of("model_review")),
            probe_call_timeout_sec=float(pick("PROBE_CALL_TIMEOUT_SEC", "120")),
            probe_call_retries=int(pick("PROBE_CALL_RETRIES", "0")),
            review_evidence_char_cap=int(pick("REVIEW_EVIDENCE_CHAR_CAP", "1200")),
            revise_base_url=pick("REVISE_BASE_URL", default_of("revise_base_url")),
            revise_api_key=pick("REVISE_API_KEY", default_of("revise_api_key")),
            model_revise=pick("MODEL_REVISE", default_of("model_revise")),
            revise_call_timeout_sec=float(pick("REVISE_CALL_TIMEOUT_SEC", "240")),
            revise_call_retries=int(pick("REVISE_CALL_RETRIES", "0")),
            revise_check_gates=pick("REVISE_CHECK_GATES", "0") not in ("0", "false", "False"),
            revise_max_history=int(pick("REVISE_MAX_HISTORY", "10")),
            probe_budget=int(pick("PROBE_BUDGET", "3")),
            perception_budget=int(pick("PERCEPTION_BUDGET", "4")),
            attribution_budget=int(pick("ATTRIBUTION_BUDGET", "3")),
            max_concurrent_infer=int(pick("MAX_CONCURRENT_INFER", "2")),
            max_queued_infer=int(pick("MAX_QUEUED_INFER", "20")),
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
                # 自检专用模型（用户 2026-10-07：实测换它以后自检从 563 秒降到 52 秒）
                "review_model": self.model_review,
                # 改写（用户意见）用的端点与模型 —— 换供应商只改 .env
                "revise_model": self.resolved_model_revise,
                "revise_base_url": self.revise_endpoint[0],
                "revise_configured": self.has_revise_llm,
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
