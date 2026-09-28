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
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from engine.core.contracts import TranscriptSegment
from engine.core.types import ConfidenceBasis, Span

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


class MimoError(RuntimeError):
    """模型调用失败。上层据此降级，不吞错。"""


class MimoClient:
    """低层聊天补全客户端。上层 Provider 组合它，业务不直接用。"""

    def __init__(self, settings: Settings, *, model_id: str | None = None) -> None:
        self.settings = settings
        self.model_id = model_id or settings.model_text
        self._opener = urllib.request.build_opener()

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
    ) -> Mapping[str, Any]:
        """结构化输出。失败会做**一次**修复重试，再失败就抛错。"""
        messages = [
            {"role": "system", "content": system + JSON_SYSTEM_SUFFIX},
            {"role": "user", "content": self._user_content(user, images)},
        ]
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
        raw = self._post("/chat/completions", payload)
        text = _message_text(raw)
        parsed = _try_parse(text)
        if parsed is not None:
            return parsed

        # 一次修复重试：把坏输出回灌，让模型只修 JSON
        repair_messages = messages + [
            {"role": "assistant", "content": text[:4000]},
            {"role": "user", "content": "上面的输出不是合法 JSON。请只重新输出合法 JSON，不要任何其他文字。"},
        ]
        repair_payload = dict(payload)
        repair_payload["messages"] = repair_messages
        repair_payload.pop("response_format", None)
        raw2 = self._post("/chat/completions", repair_payload)
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
        return _message_text(self._post("/chat/completions", payload))

    def transcribe(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str = "zh",
    ) -> list[TranscriptSegment]:
        """语音识别，返回词/句级时间戳片段。"""
        target = model or self.settings.model_asr
        path = Path(audio_path)
        if not path.exists():
            raise MimoError(f"音频文件不存在：{audio_path}")

        fields = {"model": target, "language": language}
        body, content_type = _multipart(fields, "file", path)
        raw = self._post(
            "/audio/transcriptions",
            None,
            raw_body=body,
            content_type=content_type,
            model=target,
        )
        return _segments_from_asr(raw)

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
    ) -> Mapping[str, Any]:
        if not self.settings.has_llm:
            raise MimoError("未配置 MIMO_API_KEY：请复制 .env.example 为 .env 并填入 Key")

        url = self.settings.mimo_base_url.rstrip("/") + endpoint
        if raw_body is None:
            raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"

        headers = {
            "Authorization": f"Bearer {self.settings.mimo_api_key}",
            "Content-Type": content_type or "application/json; charset=utf-8",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            request = urllib.request.Request(
                url, data=raw_body, headers=headers, method="POST"
            )
            try:
                with self._opener.open(
                    request, timeout=self.settings.request_timeout_sec
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # 4xx/5xx：带响应体便于定位
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                last_error = MimoError(f"HTTP {exc.code} @{endpoint} {model or ''}: {detail}")
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = MimoError(f"调用失败 @{endpoint}: {exc}")
                time.sleep(1.0 * (attempt + 1))
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


def _multipart(
    fields: Mapping[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    """极简 multipart/form-data 编码。只为 ASR 上传服务，不做通用实现。"""
    boundary = f"----mod{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_path.name}"\r\nContent-Type: {mime}\r\n\r\n'.encode("utf-8")
    )
    chunks.append(file_path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _segments_from_asr(raw: Mapping[str, Any]) -> list[TranscriptSegment]:
    """把 ASR 响应统一成 TranscriptSegment。

    不同实现返回的字段名不一样，这里做一次兼容收敛，
    但**绝不编造时间戳**：拿不到就退化成整段（confidence 同步下调）。
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
                confidence=0.85,
            ))
        if out:
            return out

    text = str(raw.get("text") or "").strip()
    if not text:
        return []
    # 无时间戳：明确标注为低置信整段，让上层决策，而不是假装有词级对齐
    return [TranscriptSegment(
        text=text,
        span=Span(0, 0),
        confidence=0.55,
    )]


def confidence_for(basis: ConfidenceBasis) -> float:
    from engine.core.types import CONFIDENCE_BASE

    return CONFIDENCE_BASE[basis]
