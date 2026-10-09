"""把一个镜头按时间均匀抽成一条帧带，人工核实运镜结论。

**为什么需要它**：算法内部的数字（位移/散度/一致度）不能自证。
    - 亮度法说"固定"、光流说"拉远"——到底谁对？
    - 光流说"拉远"，是镜头真在拉远，还是人物朝镜头走近？
这两个问题的唯一裁判是人眼。所以每次改运镜判据，都要能一眼看到画面。

产出：workspace/motion_frames/<视频名>/镜<N>_<起-止>s.jpg
      一条横向拼贴的帧带 + 一张叠加了光流主方向的示意

用法：
    .venv\\Scripts\\python.exe scripts\\shot_strip.py --video "测试素材/兴趣岛增长中台_2.mp4" --shot 1
    .venv\\Scripts\\python.exe scripts\\shot_strip.py --video ... --shot 1 --frames 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "workspace" / "motion_frames"


def main() -> int:
    parser = argparse.ArgumentParser(description="抽帧带人工核实运镜")
    parser.add_argument("--video", required=True)
    parser.add_argument("--shot", type=int, default=0, help="镜头号；0 = 全部镜头")
    parser.add_argument("--frames", type=int, default=6, help="每个镜头抽几帧")
    parser.add_argument("--width", type=int, default=220)
    args = parser.parse_args()

    from engine.app.config import Settings
    from engine.app.providers import optical_flow
    from engine.app.providers.ffmpeg_media import FFmpegMedia
    from engine.app.providers.scene_detect import SceneDetectShots

    settings = Settings.load()
    media = FFmpegMedia(settings)
    video = Path(args.video)
    info = media.probe(str(video))
    detector = SceneDetectShots(settings)
    shot_source = detector if detector.available else media
    shots = shot_source.detect(str(video), info)

    frames = optical_flow.decode_gray(settings.ffmpeg_bin, str(video))
    measured = optical_flow.measure_shots(frames, shots)

    targets = [s for s in shots if args.shot in (0, s.index)]
    out_dir = OUT_ROOT / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for shot, measurement in zip(shots, measured):
        if shot not in targets:
            continue
        estimate = optical_flow.analyze_flow_shot(measurement, shot)
        count = max(2, args.frames)
        times = [
            shot.start_ms / 1000 + (shot.duration_sec * i / (count - 1))
            for i in range(count)
        ]
        raw_dir = out_dir / f"镜{shot.index}_raw"
        files = media.extract_frames(
            str(video), str(raw_dir), timestamps_sec=times, width=args.width
        )
        if len(files) < 2:
            print(f"  [跳过] 镜{shot.index}：抽帧不足（{len(files)}）")
            continue
        strip = out_dir / f"镜{shot.index}_{shot.start_ms / 1000:.1f}-{shot.end_ms / 1000:.1f}s.jpg"
        _hstack(settings.ffmpeg_bin, files, strip)

        print(f"\n镜{shot.index}  {shot.start_ms / 1000:.2f}s → {shot.end_ms / 1000:.2f}s"
              f"  （{shot.duration_sec:.2f}s，间隔 {shot.duration_sec / (count - 1):.2f}s）")
        print(f"  量测：位移={measurement.translation:.2f} 一致={measurement.direction:.2f} "
              f"散度={measurement.divergence:+.2f} 主导={measurement.subject_dominance:.2f} "
              f"有效={measurement.valid_ratio:.2f}")
        print(f"  光流结论：{estimate.kind}　{estimate.note}")
        print(f"  帧带 → {strip}")

    print(f"\n帧带目录 → {out_dir}")
    print("人工看点：横向看这条帧带，画面里的人/物是在变大变小（推拉）、整体平移"
          "（摇移）、还是只有局部在动而背景没动（被摄体动 → 固定机位）。")
    return 0


def _hstack(ffmpeg_bin: str, files: list[str], out: Path) -> None:
    """横向拼贴。用 ffmpeg 做，避免引入 Pillow 依赖。"""
    inputs: list[str] = []
    for path in files:
        inputs += ["-i", path]
    filters = "".join(f"[{i}:v]" for i in range(len(files)))
    chain = f"{filters}hstack=inputs={len(files)}[out]"
    subprocess.run(
        [ffmpeg_bin, "-y", "-v", "error", *inputs, "-filter_complex", chain,
         "-map", "[out]", "-q:v", "3", str(out)],
        check=False, capture_output=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
