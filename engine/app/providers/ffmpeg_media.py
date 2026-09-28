"""ffmpeg 适配器：技术解析、镜头切分、抽帧。

为什么切分用 ffmpeg 的 ``scdet`` 而不是自己抽帧算差分：

    ``ffmpeg -vf scdet`` 内部就是直方图差分 + 场景检测，而且
    ``showinfo`` 会把每帧的 ``lavfi.scene_score`` 打出来 —— 这既是切分结果，
    也是**可追溯的证据**（PRD F1.8「每个位置带证据」）。

真实时长只认 ffprobe（PRD §6.1.6：时长错了是致命级）。
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from engine.core.contracts import ShotBoundary, VideoInfo

from ..config import Settings

ProgressFn = Callable[[str, float], None]


class MediaError(RuntimeError):
    """ffmpeg/ffprobe 调用失败。"""


class FFmpegMedia:
    """``VideoProbe`` + ``ShotDetector`` 的 ffmpeg 实现。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_frame_failures: list[str] = []
        """最近一次抽帧里失败的帧说明。编排层据此记降级，而不是静默丢帧。"""

    # ------------------------------------------------------------------
    # 技术解析
    # ------------------------------------------------------------------
    def probe(self, path: str) -> VideoInfo:
        """取真实时长/帧率/分辨率/音轨。时长误差必须为 0。"""
        raw = self._run_json([
            self.settings.ffprobe_bin,
            "-v", "error",
            "-show_format", "-show_streams",
            "-of", "json",
            path,
        ])
        streams = raw.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if video is None:
            raise MediaError(f"文件里没有视频流：{path}")

        fmt = raw.get("format", {})
        duration = _to_float(fmt.get("duration"))
        if duration <= 0:
            duration = _to_float(video.get("duration"))

        return VideoInfo(
            duration_sec=duration,
            width=int(video.get("width") or 0),
            height=int(video.get("height") or 0),
            fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
            has_audio=audio is not None,
            codec=str(video.get("codec_name") or ""),
            audio_codec=str((audio or {}).get("codec_name") or ""),
            path=str(path),
        )

    # ------------------------------------------------------------------
    # 镜头切分
    # ------------------------------------------------------------------
    def detect(
        self,
        path: str,
        info: VideoInfo,
        *,
        on_progress: ProgressFn | None = None,
    ) -> list[ShotBoundary]:
        """镜头切分：逐帧 RGB 均值差分。

        为什么这么做：``ffmpeg`` 的 ``scdet`` 滤镜不输出判定日志，
        而 ``showinfo`` 会逐帧打印 ``mean:[R G B]`` —— 那本身就是
        降采样之后的画面特征，拿它做帧间差分既准确又**自带证据**
        （每个切点都能回指到具体帧号与均方差值）。

        代价是要逐帧解码整段；对 ≤3 分钟的视频这是可接受的。
        """
        vf = "scale=64:64,showinfo"
        stderr = self._run_text([
            self.settings.ffmpeg_bin,
            "-hide_banner", "-nostats",
            "-i", path,
            "-vf", vf,
            "-an",
            "-f", "null", "-",
        ], on_line=on_progress)

        means = parse_frame_means(stderr)
        cut_ms = detect_cuts(
            means,
            threshold=self.settings.scene_threshold,
            min_gap_ms=300,
        )
        return _build_boundaries(cut_ms, info.duration_sec)

    # ------------------------------------------------------------------
    # 抽帧 / 抽音轨
    # ------------------------------------------------------------------
    def extract_audio(self, path: str, out_path: str) -> str:
        """抽音轨为 16kHz 单声道 wav —— ASR 最省的输入形态。"""
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._run_text([
            self.settings.ffmpeg_bin,
            "-y", "-v", "error",
            "-i", path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", str(target),
        ])
        return str(target)

    def extract_frames(
        self,
        path: str,
        out_dir: str,
        *,
        timestamps_sec: Sequence[float],
        width: int = 512,
    ) -> list[str]:
        """按时间点抽帧为 JPEG。

        用 ``-ss``（输入侧定位）+ 逐帧调用：3 分钟视频的镜头数通常在 30 以内，
        这点进程开销远小于"整段解码再筛帧"的代价，且时间点精确。

        **单帧失败不拖垮整条链路**：定位点超出实际帧位置、编码器抖动等情况
        会让 ffmpeg 返回非零，但那只是一帧缺失，不该让整次反推失败。
        失败的帧会被跳过，由调用方按"帧数不足"降级处理。
        """
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        out: list[str] = []
        failures: list[str] = []
        for index, ts in enumerate(timestamps_sec, start=1):
            target = directory / f"frame_{index:03d}.jpg"
            try:
                self._run_text([
                    self.settings.ffmpeg_bin,
                    "-y", "-v", "error",
                    "-ss", f"{max(0.0, ts):.3f}",
                    "-i", path,
                    "-frames:v", "1",
                    "-vf", f"scale={width}:-2",
                    "-q:v", "3",
                    str(target),
                ])
            except MediaError as exc:
                failures.append(f"镜头{index}@{ts:.3f}s：{str(exc)[:60]}")
                target.unlink(missing_ok=True)
                continue
            if target.exists() and target.stat().st_size > 0:
                out.append(str(target))
            else:
                failures.append(f"镜头{index}@{ts:.3f}s：未产出文件")
        if failures and not out:
            # 一帧都没抽出来才算真失败 —— 否则降级继续
            raise MediaError("全部关键帧抽取失败：" + "；".join(failures[:3]))
        self.last_frame_failures = failures
        return out

    def extract_keyframes(
        self,
        path: str,
        shots: Sequence[ShotBoundary],
        out_dir: str,
    ) -> list[str]:
        """每个镜头取中间帧作为代表帧 —— 避免取到转场瞬间。"""
        timestamps = [
            (shot.start_ms + shot.end_ms) / 2000.0 for shot in shots
        ]
        return self.extract_frames(path, out_dir, timestamps_sec=timestamps)

    # ------------------------------------------------------------------
    # 子进程
    # ------------------------------------------------------------------
    def _run_json(self, args: Sequence[str]) -> Mapping[str, Any]:
        text = self._run_text(args, capture_stdout=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MediaError(f"ffprobe 输出不是 JSON：{text[:200]!r}") from exc

    def _run_text(
        self,
        args: Sequence[str],
        *,
        capture_stdout: bool = False,
        on_line: ProgressFn | None = None,
    ) -> str:
        try:
            process = subprocess.run(
                list(args),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError as exc:
            raise MediaError(
                f"找不到可执行文件 {args[0]}，请在 .env 里配置绝对路径"
            ) from exc

        if process.returncode != 0:
            raise MediaError(
                f"{Path(args[0]).name} 退出码 {process.returncode}："
                f"{(process.stderr or '')[-400:]}"
            )
        stderr = process.stderr or ""
        if on_line and stderr:
            _stream_progress(stderr, on_line)
        return process.stdout if capture_stdout else stderr


# ----------------------------------------------------------------------
# 解析工具
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FrameMean:
    """一帧的画面特征。``mean`` 是 showinfo 输出的三通道均值。"""

    index: int
    pts_time: float
    mean: tuple[float, float, float]

    @property
    def luma(self) -> float:
        return 0.299 * self.mean[0] + 0.587 * self.mean[1] + 0.114 * self.mean[2]

    def distance(self, other: "FrameMean") -> float:
        """人眼感知加权的归一化色差（BT.601 亮度权重）。

        为什么加权：等权欧氏距离会低估"暖棕→深灰"这类变化
        （实测仅 0.09，低于阈值而漏切），却高估蓝色通道的微小抖动。
        按 0.299/0.587/0.114 加权后更贴近人眼判断，
        归一化分母是加权后的最大可能距离。
        """
        weights = (0.299, 0.587, 0.114)
        total = sum(w * (a - b) ** 2 for w, a, b in zip(weights, self.mean, other.mean))
        return (total ** 0.5) / 139.14  # sqrt(0.299²+0.587²+0.114²) × 255 的近似归一


_MEAN_RE = re.compile(r"mean:\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]")
_INDEX_RE = re.compile(r"\bn:\s*(\d+)")


def parse_frame_means(stderr: str) -> list[FrameMean]:
    """从 showinfo 输出解析逐帧三通道均值。

    showinfo 的一行可能被 PowerShell/管道折成多行，所以这里按
    "含 ``mean:[…]`` 即为一帧"来抓，不依赖行结构。
    """
    frames: list[FrameMean] = []
    for line in stderr.splitlines():
        match = _MEAN_RE.search(line)
        if not match:
            continue
        pts = _extract_pts_time(line)
        if pts is None:
            continue
        index_match = _INDEX_RE.search(line)
        frames.append(FrameMean(
            index=int(index_match.group(1)) if index_match else len(frames),
            pts_time=pts,
            mean=(float(match.group(1)), float(match.group(2)), float(match.group(3))),
        ))
    frames.sort(key=lambda f: f.pts_time)
    return frames


def detect_cuts(
    frames: Sequence[FrameMean],
    *,
    threshold: float = 0.32,
    min_gap_ms: int = 300,
) -> list[int]:
    """帧间差分找切点，返回毫秒时间戳。

    ``min_gap_ms`` 抑制同一转场里的连续误报（一段渐变会被判成多次切点）。
    """
    cuts: list[int] = []
    for previous, current in zip(frames, frames[1:]):
        if current.distance(previous) < threshold:
            continue
        timestamp = int(current.pts_time * 1000)
        if cuts and timestamp - cuts[-1] < min_gap_ms:
            continue
        cuts.append(timestamp)
    return cuts


def _build_boundaries(cut_ms: Sequence[int], duration_sec: float) -> list[ShotBoundary]:
    """把切点整理成连续的镜头区间。首尾必须覆盖全片。"""
    total_ms = int(round(duration_sec * 1000))
    edges = [0]
    for cut in cut_ms:
        # 丢弃过于贴边的切点：它们通常是编码抖动，不是真切镜
        if 60 < cut < total_ms - 60:
            edges.append(cut)
    edges.append(total_ms)
    edges = sorted(set(edges))

    boundaries: list[ShotBoundary] = []
    for index in range(len(edges) - 1):
        start, end = edges[index], edges[index + 1]
        if end <= start:
            continue
        boundaries.append(ShotBoundary(index=index + 1, start_ms=start, end_ms=end))
    return boundaries


def _extract_pts_time(line: str) -> float | None:
    marker = "pts_time:"
    index = line.find(marker)
    if index < 0:
        return None
    tail = line[index + len(marker):].strip()
    token = tail.split()[0] if tail else ""
    try:
        return float(token)
    except ValueError:
        return None


def _extract_scene_score(line: str) -> float | None:
    marker = "lavfi.scene_score="
    index = line.find(marker)
    if index < 0:
        return None
    tail = line[index + len(marker):].strip()
    token = tail.split()[0] if tail else ""
    try:
        return float(token)
    except ValueError:
        return None


def _build_boundaries(cut_ms: Sequence[int], duration_sec: float) -> list[ShotBoundary]:
    """把切点整理成连续的镜头区间。首尾必须覆盖全片。"""
    total_ms = int(round(duration_sec * 1000))
    edges = [0]
    for cut in cut_ms:
        # 丢弃过于贴边的切点：它们通常是编码抖动，不是真切镜
        if 60 < cut < total_ms - 60:
            edges.append(cut)
    edges.append(total_ms)
    edges = sorted(set(edges))

    boundaries: list[ShotBoundary] = []
    for index in range(len(edges) - 1):
        start, end = edges[index], edges[index + 1]
        if end <= start:
            continue
        boundaries.append(ShotBoundary(index=index + 1, start_ms=start, end_ms=end))
    return boundaries


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_fps(raw: Any) -> float:
    text = str(raw or "")
    if "/" in text:
        num, _, den = text.partition("/")
        numerator, denominator = _to_float(num), _to_float(den)
        return numerator / denominator if denominator else 0.0
    return _to_float(text)


def _stream_progress(stderr: str, on_line: ProgressFn) -> None:
    for line in stderr.splitlines():
        if "time=" not in line:
            continue
        index = line.find("time=")
        token = line[index + 5:].strip().split()[0]
        seconds = _parse_timecode(token)
        if seconds is None:
            continue
        on_line("解码中", seconds)


def _parse_timecode(token: str) -> float | None:
    parts = token.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(p) for p in parts)
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds
