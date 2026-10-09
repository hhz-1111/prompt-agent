"""把人工标注与逐镜头原始量测并排放，用于校准阈值。

**做的是校准，不是调参凑分**：
先看标注里同一档位的样本量测分布是否可分；不可分的就如实记下"这档分不开"，
而不是把阈值挪到刚好全对。

产出：
    控制台表格（标注 / 量测 / 当前判定 / 是否命中）
    workspace/benchmark/calibration.json

用法：
    .venv\\Scripts\\python.exe scripts\\calibrate_dimensions.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from dimension_regression import (  # noqa: E402
    _ground_truth,
    _is_filled,
    _load_rows,
    ANNOTATION_DIR,
)


def main() -> int:
    from engine.app.config import Settings
    from engine.app.providers import cv as cv_layer
    from engine.app.providers import optical_flow
    from engine.app.providers.ffmpeg_media import FFmpegMedia
    from engine.app.providers.scene_detect import SceneDetectShots

    settings = Settings.load()
    media = FFmpegMedia(settings)
    detector = SceneDetectShots(settings)
    shot_source = detector if detector.available else media
    yolo = cv_layer.YoloPersonDetector("yolov8n", workspace=settings.workspace_dir / "models")

    rows = [r for r in _load_rows(ANNOTATION_DIR / "labels.csv") if _is_filled(r)]
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)

    records: list[dict[str, object]] = []
    for name, video_rows in by_video.items():
        video = Path("测试素材") / name
        info = media.probe(str(video))
        shots = shot_source.detect(str(video), info)
        frames = media.extract_keyframes(str(video), shots, str(
            settings.workspace_dir / "calibration" / video.stem
        ))
        flow_frames = optical_flow.decode_gray(settings.ffmpeg_bin, str(video))
        measured = optical_flow.measure_shots(flow_frames, shots)

        for row in video_rows:
            index = int(row["shot"])
            shot = next((s for s in shots if s.index == index), None)
            if shot is None:
                print(f"[!] {name} 镜{index}：切分结果里没有这个镜头")
                continue
            frame = frames[index - 1] if index - 1 < len(frames) else ""
            try:
                raw_boxes = yolo.detect(frame) if frame else []
            except Exception as exc:  # noqa: BLE001
                raw_boxes = []
                print(f"[!] {name} 镜{index} 检测失败：{exc}")
            boxes = raw_boxes
            framing = cv_layer.framing_from_person(max(boxes, key=lambda b: b.height_ratio)) if boxes else None
            blocking, count, _ = cv_layer.blocking_from_boxes(boxes)
            composition, midpoint, fg_count = cv_layer.composition_side(boxes)
            measurement = measured[shots.index(shot)]
            truth = _ground_truth(row)
            records.append({
                "video": name,
                "shot": index,
                "truth": truth,
                "person_ratio": round(framing.person_ratio, 3) if framing else None,
                "framing_kind": framing.kind if framing else "",
                "boxes": [
                    {"x": round(b.center_x, 3), "h": round(b.height_ratio, 3)}
                    for b in boxes
                ],
                "blocking": [{"side": b.side, "x": round(b.x_ratio, 3)} for b in blocking],
                "composition": composition,
                "midpoint": round(midpoint, 3),
                "people": count,
                "all_boxes": [
                    {"x": round(b.center_x, 3), "h": round(b.height_ratio, 3),
                     "conf": round(b.confidence, 3)}
                    for b in raw_boxes
                ],
                "translation": round(measurement.translation, 3),
                "p10": round(measurement.p10_translation, 3),
                "direction": round(measurement.direction, 3),
                "divergence": round(measurement.divergence, 3),
                "dominance": round(measurement.subject_dominance, 3),
                "flow_kind": optical_flow.analyze_flow_shot(measurement, shot).kind,
            })

    out = Path("workspace/benchmark/calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 108)
    print("景别：人工标注 vs 人体框高占比")
    print("=" * 108)
    print(f"{'视频':<22}{'镜':>3}  {'标注':<8}{'高占比':>7}{'框数':>5}  {'当前判定':<8}{'命中':<6}各框 x/高度")
    for record in records:
        hit = "✓" if record["framing_kind"] == record["truth"]["景别"] else "✗"
        boxes = " ".join(f"{b['x']:.2f}/{b['h']:.2f}" for b in record["boxes"])
        print(f"{str(record['video'])[:20]:<22}{record['shot']:>3}  "
              f"{record['truth']['景别'] or '—':<8}{record['person_ratio'] if record['person_ratio'] is not None else -1:>7.2f}"
              f"{len(record['boxes']):>5}  {record['framing_kind'] or '—':<8}{hit:<6}{boxes}")

    print()
    print("=" * 108)
    print("运镜：人工标注 vs 光流量测")
    print("=" * 108)
    print(f"{'视频':<22}{'镜':>3}  {'标注':<14}{'位移':>7}{'p10':>6}{'一致':>6}{'散度':>7}{'主导':>7}  {'当前判定':<14}{'命中'}")
    for record in records:
        hit = "✓" if record["flow_kind"] == record["truth"]["运镜"] else "✗"
        print(f"{str(record['video'])[:20]:<22}{record['shot']:>3}  "
              f"{record['truth']['运镜'] or '—':<14}{record['translation']:>7.2f}"
              f"{record['p10']:>6.2f}{record['direction']:>6.2f}"
              f"{record['divergence']:>+7.2f}{record['dominance']:>7.2f}"
              f"  {record['flow_kind']:<14}{hit}")

    print()
    print("=" * 108)
    print("站位：人工标注 vs 检测框 x（判定取「构图跨度中点」）")
    print("=" * 108)
    print(f"{'视频':<22}{'镜':>3}  {'标注':<8}{'人数':>5}  {'构图判定':<9}{'命中':<6}中点  全部检测框（x/高/置信）")
    for record in records:
        current = str(record.get("composition") or "")
        hit = "✓" if current == record["truth"]["站位"] else "✗"
        boxes = " ".join(
            f"{b['x']:.2f}/{b['h']:.2f}/{b['conf']:.2f}" for b in record.get("all_boxes", [])
        )
        print(f"{str(record['video'])[:20]:<22}{record['shot']:>3}  "
              f"{record['truth']['站位'] or '—':<8}{record['people']:>5}  "
              f"{current or '—':<9}{hit:<6}{record.get('midpoint', 0):.2f}  {boxes}")

    print()
    print("── 分档分布（判断「这档能不能分开」）──")
    _distribution(records, "景别", "person_ratio")
    _distribution(records, "运镜", "translation")
    print(f"\n原始量测 → {out}")
    return 0


def _distribution(records, dimension: str, field: str) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value is None or not record["truth"][dimension]:
            continue
        groups[str(record["truth"][dimension])].append(float(value))
    print(f"\n  {dimension}（按 {field}）")
    for label, values in sorted(groups.items(), key=lambda kv: -min(kv[1])):
        values = sorted(values)
        span = f"{values[0]:.2f} ~ {values[-1]:.2f}" if len(values) > 1 else f"{values[0]:.2f}"
        print(f"    {label:<14} n={len(values):<3} {field} ∈ [{span}]  "
              f"中位 {values[len(values) // 2]:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
