"""运镜光流量测探针：在**真实素材**上量测光流指标，用数据定阈值。

存在的理由（HANDOFF 坑 1）：合成素材上校准的参数套真实内容必然失准。
这个项目已经在 PySceneDetect 阈值和亮度运镜阈值上栽过两次。所以光流的
判定阈值不允许拍脑袋 —— 先把 24 条真实视频的原始量测摆出来。

输出：
    workspace/benchmark/motion_flow.json   原始量测（可反复分析，不用重跑）
    控制台                                 分布 + 与亮度法/视觉模型的对照

用法：
    .venv\\Scripts\\python.exe scripts\\motion_probe.py
    .venv\\Scripts\\python.exe scripts\\motion_probe.py --limit 6 --no-llm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}


def luma_estimates(settings, video: Path, shots) -> list:
    """亮度法对照：跑现有 ``cv.analyze_motion`` 的原始输入路径。

    刻意走"重新解码 showinfo"而不是复用光流的帧数组 ——
    两条路的输入必须各自独立，对照才有意义。
    """
    from engine.app.providers import cv as cv_layer
    from engine.app.providers.ffmpeg_media import parse_frame_means

    process = subprocess.run(
        [
            settings.ffmpeg_bin, "-hide_banner", "-nostats",
            "-i", str(video), "-vf", "scale=64:64,showinfo", "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    frames = parse_frame_means(process.stderr or "")
    return [cv_layer.analyze_motion(frames, shot) for shot in shots]


def main() -> int:
    parser = argparse.ArgumentParser(description="光流运镜阈值量测")
    parser.add_argument("--dir", default="测试素材")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="workspace/benchmark/motion_flow.json")
    parser.add_argument("--no-llm", action="store_true", help="不调模型，只量测+亮度法对照")
    args = parser.parse_args()

    from engine.app.config import Settings
    from engine.app.providers import optical_flow
    from engine.app.providers.ffmpeg_media import FFmpegMedia

    settings = Settings.load()
    media = FFmpegMedia(settings)
    videos = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print("[X] 目录里没有视频", file=sys.stderr)
        return 1

    inferencer = None
    if not args.no_llm:
        from engine.app.pipeline.infer import build_inferencer

        inferencer = build_inferencer(settings, with_llm=True)

    # 切分用**和正式链路同一个检测器**：量测的镜头边界必须和线上一致，
    # 否则量出来的光流指标对不上真实运行时的表现（PySceneDetect vs ffmpeg 差异很大）
    from engine.app.pipeline.infer import build_inferencer

    detector = build_inferencer(settings, with_llm=False).detector

    print(f"光流量测：{len(videos)} 条 @ {args.dir}")
    print(f"  采样：{optical_flow.FLOW_WIDTH}px 宽 / {optical_flow.FLOW_FPS}fps")
    print(f"  阈值：固定<{optical_flow.FIXED_TOTAL_MEDIAN}"
          f"　散度≥{optical_flow.ZOOM_DIVERGENCE}"
          f"　摇移≥{optical_flow.PAN_DIRECTION}　手持<{optical_flow.HANDHELD_DIRECTION}")
    print("─" * 100)

    records = []
    for index, video in enumerate(videos, start=1):
        started = time.time()
        info = media.probe(str(video))
        shots = detector.detect(str(video), info)
        frames = optical_flow.decode_gray(settings.ffmpeg_bin, str(video))
        measured = optical_flow.measure_shots(frames, shots)
        flows = [optical_flow.analyze_flow_shot(m, s) for m, s in zip(measured, shots)]
        lumas = luma_estimates(settings, video, shots)

        model_camera: list[str] = []
        if inferencer is not None:
            try:
                _, report = inferencer.infer_to_ir(
                    str(video), workspace=settings.workspace_dir / "motion_probe"
                )
                model_camera = [u.camera for u in report.understandings]
            except Exception as exc:  # noqa: BLE001
                model_camera = [f"<error {type(exc).__name__}>"]

        shots_out = []
        for i, (shot, m, flow, luma) in enumerate(zip(shots, measured, flows, lumas)):
            shots_out.append({
                "index": shot.index,
                "duration_s": round(shot.duration_sec, 2),
                "frames": m.frames,
                "translation": round(m.translation, 3),
                "direction": round(m.direction, 3),
                "divergence": round(m.divergence, 3),
                "spread": round(m.spread, 3),
                "subject_dominance": round(m.subject_dominance, 3),
                "valid_ratio": round(m.valid_ratio, 3),
                "flow_kind": flow.kind,
                "flow_conf": flow.confidence,
                "luma_kind": luma.kind,
                "model_kind": model_camera[i] if i < len(model_camera) else "",
            })

        records.append({
            "file": video.name,
            "duration_s": round(info.duration_sec, 2),
            "shot_count": len(shots),
            "shots": shots_out,
            "elapsed_s": round(time.time() - started, 1),
        })

        kinds = Counter(s["flow_kind"] for s in shots_out)
        luma_kinds = Counter(s["luma_kind"] for s in shots_out)
        print(
            f"[{index:2}/{len(videos)}] {video.name[:30]:<30} "
            f"{records[-1]['duration_s']:6.2f}s {len(shots)}镜 "
            f"光流={dict(kinds)} 亮度={dict(luma_kinds)} {records[-1]['elapsed_s']}s"
        )
        for s in shots_out:
            print(
                f"        镜{s['index']} {s['duration_s']:5.2f}s "
                f"位移={s['translation']:6.2f} 一致={s['direction']:.2f} "
                f"散度={s['divergence']:+6.2f} 主导={s['subject_dominance']:.2f} "
                f"有效={s['valid_ratio']:.2f} "
                f"→ 光流:{s['flow_kind']:<14} 亮度:{s['luma_kind']:<10} "
                f"模型:{s['model_kind'] or '—'}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("─" * 100)
    flow_counter = Counter(s["flow_kind"] for r in records for s in r["shots"])
    luma_counter = Counter(s["luma_kind"] for r in records for s in r["shots"])
    print(f"光流判定分布   {dict(flow_counter.most_common())}")
    print(f"亮度判定分布   {dict(luma_counter.most_common())}")
    print(f"原始量测 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
