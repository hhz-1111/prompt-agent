"""验证人体框是真人物还是背景虚警：看这块区域**动没动**。

动机：`IP演播厅带练-丹` 的检测报告 x=0.16 处有个高度 0.45 的"人"，
而画面里真人应该在中间。LED 背景屏上印着人物剪影，是最可能的干扰源。
置信度分不开它们（虚警置信度 0.51~0.74，真人也差不多），
但**运动**分得开：真人会动，屏上的图像不动。

做法：对镜头内多帧做逐帧差分，统计每个检测框区域内的运动强度，
再与全画面的平均运动对比。比值接近 1 说明这块和全画面一样在动（真人），
接近 0 说明这块是静止的（背景图像）。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_person_boxes.py --video "测试素材/IP演播厅带练-丹.mp4"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="人体框静止/运动判别")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=12, help="每镜头采样帧数")
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    import numpy as np

    from engine.app.config import Settings
    from engine.app.providers import cv as cv_layer
    from engine.app.providers.ffmpeg_media import FFmpegMedia
    from engine.app.providers.scene_detect import SceneDetectShots

    settings = Settings.load()
    media = FFmpegMedia(settings)
    video = Path(args.video)
    info = media.probe(str(video))
    detector = SceneDetectShots(settings)
    shots = detector.detect(str(video), info) if detector.available else media.detect(str(video), info)
    yolo = cv_layer.YoloPersonDetector("yolov8n", workspace=settings.workspace_dir / "models")

    out_dir = settings.workspace_dir / "box_verify" / video.stem
    for shot in shots[:2]:  # 前两个镜头足够看出问题
        times = [
            shot.start_ms / 1000 + shot.duration_sec * i / (args.frames - 1)
            for i in range(args.frames)
        ]
        files = media.extract_frames(
            str(video), str(out_dir / f"镜{shot.index}"), timestamps_sec=times, width=args.width
        )
        if len(files) < 3:
            continue
        frames = [_to_gray(path) for path in files]
        mid = frames[len(frames) // 2]
        boxes = yolo.detect(files[len(files) // 2])

        # 全画面逐帧差分强度
        global_motion = float(np.mean([
            np.mean(np.abs(b.astype(np.int16) - a.astype(np.int16)))
            for a, b in zip(frames, frames[1:])
        ]))

        print(f"\n镜{shot.index}　{shot.duration_s if hasattr(shot, 'duration_s') else shot.duration_sec:.2f}s"
              f"　全画面平均帧差={global_motion:.2f}")
        print(f"  {'框中心x':>8}{'高度':>7}{'置信':>7}{'框内帧差':>10}{'相对全画面':>11}  判定")
        for box in boxes:
            patch_motion = _box_motion(frames, box)
            ratio = patch_motion / global_motion if global_motion > 1e-6 else 0.0
            verdict = "会动（真人）" if ratio >= 0.6 else "基本静止（疑似背景图像）"
            print(f"  {box.center_x:>8.2f}{box.height_ratio:>7.2f}{box.confidence:>7.2f}"
                  f"{patch_motion:>10.2f}{ratio:>11.2f}  {verdict}")
        print(f"  帧图 → {out_dir / f'镜{shot.index}'}")

    return 0


def _to_gray(path: str) -> "object":
    import numpy as np

    data = Path(path).read_bytes()
    # 用 cv2 解码，避免为一张图引入额外依赖路径
    import cv2

    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return array


def _box_motion(frames: list, box) -> float:
    """检测框区域内的平均逐帧差。"""
    import numpy as np

    height, width = frames[0].shape[:2]
    y1 = max(0, int(box.y1 * height))
    y2 = min(height, int(box.y2 * height))
    x1 = max(0, int(box.x1 * width))
    x2 = min(width, int(box.x2 * width))
    if y2 - y1 < 2 or x2 - x1 < 2:
        return 0.0
    values = []
    for a, b in zip(frames, frames[1:]):
        patch_a = a[y1:y2, x1:x2].astype(np.int16)
        patch_b = b[y1:y2, x1:x2].astype(np.int16)
        values.append(float(np.mean(np.abs(patch_b - patch_a))))
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
