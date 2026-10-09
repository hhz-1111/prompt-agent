"""小米 MiMo 适配器（OpenAI 兼容协议）。

这是**唯一**直接跟模型说话的地方。换模型只改本文件（PRD F4.9 中台适配器）。

关键实现约束（都是实测出来的）：

1. ``json_schema`` 严格模式能返回合法 JSON，但**偶尔会串入思考文本**，
   所以必须有"校验 → 一次修复重试"的闭环，不能盲信；
2. 响应里的 ``reasoning_content`` 要显式丢弃 —— 它不是提示词内容；
3. HTTP 用标准库 ``urllib``：少一个依赖就少一份升级负担；
4. 结构化失败时**宁可抛错也不返回半成品**，由上层降级（PRD F1.12 不输出假数据）。
"""

from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from engine.core.contracts import TranscriptSegment
from engine.core.types import ConfidenceBasis, Span, compute_confidence

from ..config import Settings

# MiMo 官方推荐的中文系统提示词
BASE_SYSTEM = (
    "你是MiMo（中文名称也是MiMo），是小米公司研发的AI智能助手。"
    "你在一个视频提示词工程系统里承担分析与归纳职责。"
)

JSON_SYSTEM_SUFFIX = (
    "\n你必须只输出一个符合给定 JSON Schema 的 JSON 对象，"
    "不要输出解释、Markdown 代码块或任何多余文字。"
)


def _schema_hint(schema: Mapping[str, Any]) -> str:
    """端点不支持 ``json_schema`` 时，把 schema 写进提示词。

    实测（2026-10-08）：DeepSeek 官方端点对 ``json_schema`` 直接 400
    （``This response_format type is unavailable now``），只支持 ``json_object``。
    所以走"schema 写进系统提示词 + json_object"这条路 ——
    与 MiMo 那条路产出同样的 dict，上层 Provider 不用分叉。
    """
    return "\nJSON Schema：\n" + json.dumps(dict(schema), ensure_ascii=False)


class MimoError(RuntimeError):
    """模型调用失败。上层据此降级，不吞错。"""


# ----------------------------------------------------------------------
# 每次模型调用的计时（用户 2026-10-07：自检"卡很久"，得先看清时间花在哪）
# ----------------------------------------------------------------------
_recorder = threading.local()
"""**按线程**记账：反推跑在自己的线程里，多个人同时反推不能互相串账。"""


def record_call(
    purpose: str,
    model: str,
    seconds: float,
    *,
    ok: bool,
    detail: str = "",
    usage: Mapping[str, Any] | None = None,
) -> None:
    """记一次调用：用途、型号、耗时、成败，**以及 token 用量**。

    为什么要把 token 也记上（用户 2026-10-07 问"换模型是不是变贵了"）：
    换模型提速的代价可能正是"烧更多 token"—— 实测同一个自检任务，
    ``mimo-v2.6-pro-ultraspeed`` 用了 21152 输出 token（推理 20616），
    ``mimo-v2.6-flash`` 用了 12792（推理 12163），**多 1.65 倍**。
    没有 token 账，这件事只能靠猜（而我当时确实没量）。
    """
    records = getattr(_recorder, "records", None)
    if records is None:
        records = _recorder.records = []
    payload = usage or {}
    details = payload.get("completion_tokens_details") or {}
    records.append({
        "purpose": purpose or "未标注",
        "model": model,
        "seconds": round(seconds, 1),
        "ok": ok,
        "detail": detail[:120],
        "prompt_tokens": int(payload.get("prompt_tokens") or 0),
        "completion_tokens": int(payload.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
    })


def take_calls() -> list[dict[str, Any]]:
    """取走本线程到目前为止的调用记录（取完清空）。"""
    records = getattr(_recorder, "records", None) or []
    _recorder.records = []
    return list(records)


class MimoClient:
    """低层聊天补全客户端。上层 Provider 组合它，业务不直接用。

    **不只是 MiMo**：``base_url`` / ``api_key`` 可以按实例覆盖，
    所以同一个类既能连小米 MiMo，也能连 DeepSeek 官方 ——
    用户 2026-10-08 要求的"所有 key 都要可换端点，.env 填了就自动切"就落在这里。
    ``json_schema`` 也是实例级开关：实测 DeepSeek 官方端点
    **不接受** ``response_format={"type":"json_schema"}``（400
    `This response_format type is unavailable now`），只接受 ``json_object``，
    于是必须能退回"把 schema 写进提示词 + json_object"这条路。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        model_id: str | None = None,
        base_url: str = "",
        api_key: str = "",
        json_schema: bool = True,
    ) -> None:
        self.settings = settings
        self.model_id = model_id or settings.model_text
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.json_schema = bool(json_schema)
        self._opener = urllib.request.build_opener()

    @property
    def endpoint_base(self) -> str:
        """实际请求的端点：实例覆盖优先，否则用 MiMo 的配置。"""
        return (self.base_url or self.settings.mimo_base_url).rstrip("/")

    @property
    def token(self) -> str:
        return self.api_key or self.settings.mimo_api_key

    # ------------------------------------------------------------------
    # 公开能力
    # ------------------------------------------------------------------
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        images: Sequence[str] = (),
        temperature: float = 0.2,
        timeout_sec: float | None = None,
        retries: int | None = None,
        purpose: str = "",
    ) -> Mapping[str, Any]:
        """结构化输出。失败会做**一次**修复重试，再失败就抛错。

        ``timeout_sec`` / ``retries`` 可覆盖（见 ``_post``）：推理型长调用
        （自检-重写）要收紧，否则一次调用最坏能拖到 9 分半。
        """
        messages = [
            {"role": "system", "content": system + JSON_SYSTEM_SUFFIX},
            {"role": "user", "content": self._user_content(user, images)},
        ]
        if self.json_schema:
            payload: dict[str, Any] = {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": 8192,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "strict": True,
                        "schema": dict(schema),
                    },
                },
            }
        else:
            # 端点不支持 json_schema（实测 DeepSeek 官方 400）：把 schema
            # **写进系统提示词**，再用它支持的 json_object 兜住格式。
            payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": system + JSON_SYSTEM_SUFFIX + _schema_hint(schema),
                    },
                    {"role": "user", "content": self._user_content(user, images)},
                ],
                "temperature": temperature,
                "max_completion_tokens": 8192,
                "response_format": {"type": "json_object"},
            }
        raw = self._post(
            "/chat/completions", payload,
            timeout_sec=timeout_sec, retries=retries, purpose=purpose,
        )
        text = _message_text(raw)
        parsed = _try_parse(text)
        if parsed is not None:
            return parsed

        # 一次修复重试：把坏输出回灌，让模型只修 JSON
        repair_messages = list(payload["messages"]) + [
            {"role": "assistant", "content": text[:4000]},
            {"role": "user", "content": "上面的输出不是合法 JSON。请只重新输出合法 JSON，不要任何其他文字。"},
        ]
        repair_payload = dict(payload)
        repair_payload["messages"] = repair_messages
        repair_payload.pop("response_format", None)
        raw2 = self._post(
            "/chat/completions", repair_payload,
            timeout_sec=timeout_sec, retries=retries,
            purpose=f"{purpose}·JSON修复" if purpose else "",
        )
        parsed2 = _try_parse(_message_text(raw2))
        if parsed2 is None:
            raise MimoError(f"模型两次都未返回合法 JSON：{text[:200]!r}")
        return parsed2

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        images: Sequence[str] = (),
        temperature: float = 0.2,
        timeout_sec: float | None = None,
        retries: int | None = None,
        purpose: str = "",
    ) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._user_content(user, images)},
            ],
            "temperature": temperature,
            "max_completion_tokens": 4096,
        }
        return _message_text(self._post(
            "/chat/completions", payload,
            timeout_sec=timeout_sec, retries=retries, purpose=purpose,
        ))

    def transcribe(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str = "zh",
    ) -> list[TranscriptSegment]:
        """语音识别，返回句级时间戳片段。

        **MiMo 的 ASR 不是 OpenAI 的 ``/audio/transcriptions``**，
        而是走 ``/v1/chat/completions``，音频以 ``input_audio`` data URL 传入。
        只支持 mp3 与 wav 两种格式。
        """
        target = model or self.settings.model_asr
        path = Path(audio_path)
        if not path.exists():
            raise MimoError(f"音频文件不存在：{audio_path}")

        audio_format, mime = _audio_format(path)
        if audio_format is None:
            raise MimoError(
                f"ASR 只支持 mp3 / wav，收到 {path.suffix}；"
                "请先用 ffmpeg 转成 wav（-ac 1 -ar 16000）"
            )

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": target,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:{mime};base64,{encoded}",
                        "format": audio_format,
                    },
                }],
            }],
            "asr_options": {"language": language},
        }
        raw = self._post("/chat/completions", payload, model=target)
        return _segments_from_asr(raw, expected_span_ms=_probe_duration_ms(path))

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _user_content(self, text: str, images: Sequence[str]) -> Any:
        if not images:
            return text
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _data_url(p)}}
            for p in images
        ]
        content.append({"type": "text", "text": text})
        return content

    def _post(
        self,
        endpoint: str,
        payload: Mapping[str, Any] | None,
        *,
        raw_body: bytes | None = None,
        content_type: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        retries: int | None = None,
        purpose: str = "",
    ) -> Mapping[str, Any]:
        """发一次请求。

        ``timeout_sec`` / ``retries`` 可按调用覆盖（用户 2026-10-07 报"卡很久"）：
        默认值下**单次调用最坏能拖到 9 分半**（180 秒 × 3 次尝试），
        对"自检-重写"这种推理型长调用必须收紧，否则用户看到的就是"卡死"。

        ``purpose`` 只用于**计时记账**（「自检」「取证」「视觉理解」…），
        让用户下次能看到"这 9 分钟到底花在哪一次调用上"。
        """
        if not self.token:
            raise MimoError(
                "未配置模型 Key：请复制 .env.example 为 .env 并填入 MIMO_API_KEY"
                "（或改写专用的 REVISE_API_KEY）"
            )

        url = self.endpoint_base + endpoint
        if raw_body is None:
            raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": content_type or "application/json; charset=utf-8",
            "Accept": "application/json",
        }

        budget = float(timeout_sec or self.settings.request_timeout_sec)
        attempts = (self.settings.max_retries if retries is None else max(0, retries)) + 1
        last_error: Exception | None = None
        started = time.monotonic()
        for attempt in range(attempts):
            request = urllib.request.Request(
                url, data=raw_body, headers=headers, method="POST"
            )
            try:
                with self._opener.open(request, timeout=budget) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                record_call(
                    purpose, model or self.model_id, time.monotonic() - started,
                    ok=True, usage=parsed.get("usage") if isinstance(parsed, Mapping) else None,
                )
                return parsed
            except urllib.error.HTTPError as exc:  # 4xx/5xx：带响应体便于定位
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                last_error = MimoError(f"HTTP {exc.code} @{endpoint} {model or ''}: {detail}")
                if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                record_call(
                    purpose, model or self.model_id, time.monotonic() - started,
                    ok=False, detail=f"HTTP {exc.code}",
                )
                raise last_error from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                # **连接中途断**：实测出现过 `IncompleteRead(59 bytes read)`
                # （响应体读到一半断掉）。它不在 urllib 那两类里，原来会原样抛出去、
                # 直接算这次调用失败 —— 现在按"调用失败"处理，行为一致可预期。
                http.client.IncompleteRead,
                ConnectionError,
                OSError,
            ) as exc:
                last_error = MimoError(f"调用失败 @{endpoint}: {exc}")
                if attempt + 1 < attempts:
                    time.sleep(1.0 * (attempt + 1))
        record_call(
            purpose, model or self.model_id, time.monotonic() - started,
            ok=False, detail=str(last_error)[:80],
        )
        raise last_error or MimoError("调用失败，且没有捕获到具体错误")


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _message_text(raw: Mapping[str, Any]) -> str:
    """取助手回复正文，**显式丢弃 reasoning_content**。"""
    try:
        message = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MimoError(f"响应结构异常：{json.dumps(raw, ensure_ascii=False)[:300]}") from exc
    content = message.get("content") or ""
    if isinstance(content, list):  # 少数情况会返回分段结构
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, Mapping)
        )
    return str(content)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _try_parse(text: str) -> Mapping[str, Any] | None:
    """尽最大努力解析出 JSON 对象；失败返回 None（由调用方决定是否重试）。"""
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    fenced = _FENCE.search(stripped)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    brace = stripped.find("{")
    if brace > 0:
        candidates.append(stripped[brace:])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _data_url(path: str) -> str:
    """本地图片 → data URL。"""
    file = Path(path)
    mime = mimetypes.guess_type(file.name)[0] or "image/jpeg"
    encoded = base64.b64encode(file.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


_AUDIO_FORMATS: Mapping[str, tuple[str, str]] = {
    ".wav": ("wav", "audio/wav"),
    ".mp3": ("mp3", "audio/mpeg"),
    ".m4a": ("mp3", "audio/mpeg"),  # 需先转换
}


def _audio_format(path: Path) -> tuple[str | None, str]:
    entry = _AUDIO_FORMATS.get(path.suffix.lower())
    return entry if entry else (None, "")


def _probe_duration_ms(path: Path) -> int:
    """用 ffprobe 取音频真实时长（毫秒）。拿不到就返回 0，由上层降级。"""
    try:
        process = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=False,
        )
        return int(float(process.stdout.strip()) * 1000)
    except (ValueError, OSError):
        return 0


def _segments_from_asr(
    raw: Mapping[str, Any],
    *,
    expected_span_ms: int = 0,
) -> list[TranscriptSegment]:
    """把 ASR 响应统一成 TranscriptSegment。

    MiMo 的 ASR 返回的是识别文本，**不保证带词级时间戳**。
    这里如实处理：

    - 有 ``segments`` → 用它的时间戳；
    - 只有整段文本 + 已知音频时长 → 给出整段区间，并按"只有时间码"计置信度；
    - 只有文本、时长未知 → 时间码留 0 并降置信度，让上层决定是否标黄。

    **绝不编造词级时间戳** —— 那会让下游的对齐与时长校验全建立在假数据上。
    """
    segments = raw.get("segments")
    if isinstance(segments, list) and segments:
        out: list[TranscriptSegment] = []
        for item in segments:
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            out.append(TranscriptSegment(
                text=text,
                span=Span(int(start * 1000), int(end * 1000)),
                confidence=compute_confidence(
                    ConfidenceBasis.ASR_CLEAR,
                    consistency="single",
                    evidence="timecode_only",
                ),
                has_timestamps=True,
            ))
        if out:
            return out

    text = str(raw.get("text") or "").strip()
    if not text:
        text = _text_from_choices(raw)
    if not text:
        return []

    # 以下两种都是**没有词级时间戳**的情形。
    # 区间只是"音频总时长"推导的，必须标 has_timestamps=False，
    # 否则下游会拿它当对齐依据，把整段台词复制到每个重叠镜头。
    if expected_span_ms > 0:
        return [TranscriptSegment(
            text=text,
            span=Span(0, expected_span_ms),
            confidence=compute_confidence(
                ConfidenceBasis.ASR_CLEAR,
                consistency="single",
                evidence="timecode_only",
            ),
            has_timestamps=False,
        )]
    return [TranscriptSegment(
        text=text,
        span=Span(0, 0),
        confidence=compute_confidence(
            ConfidenceBasis.ASR_CLEAR, consistency="single", evidence="none"
        ),
        has_timestamps=False,
    )]


def _text_from_choices(raw: Mapping[str, Any]) -> str:
    """ASR 走 chat/completions，识别结果在 choices[0].message.content。"""
    try:
        return str(raw["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""
