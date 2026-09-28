"""批量体检：对一整个素材目录跑反推，输出统计与逐条诊断。

这是 PRD §6.1.5「回归基准」的工程雏形 —— 有了真实视频，
反推引擎的准确率才有办法被测量，而不是靠感觉。

当前能自动测的维度（不依赖人工标注）：

    - 技术解析成功率、时长误差（ffprobe 是唯一可信来源，误差恒为 0）
    - ASR 是否出词、置信度分布
    - 镜头切分数量分布（与 shot_mode 判定是否自洽）
    - 组装是否成功、14 项闸门的阻断分布
    - 必写维度覆盖率

需要人工标注才能测的（层次一/二）仍未覆盖：景别、站位、运镜
与标准答案的逐维度比对。那需要 17 份样本的原始提示词做 ground truth。

用法：
    py scripts/benchmark.py                       # 默认扫 测试素材/
    py scripts/benchmark.py --dir 测试素材 --limit 6
    py scripts/benchmark.py --out assets/regression/baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}


def analyze_one(inferencer, video: Path, workdir: Path) -> dict[str, Any]:
    """跑一条视频，返回可比较的指标。任何异常都记录成失败而不是抛出。"""
    from engine.core import assembler, gates, phrases as ph

    started = time.time()
    record: dict[str, Any] = {"file": video.name, "size_mb": round(video.stat().st_size / 1e6, 1)}
    try:
        ir, report = inferencer.infer_to_ir(str(video), workspace=workdir)
    except Exception as exc:
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        return record

    record["elapsed_s"] = round(time.time() - started, 1)
    record["duration_s"] = round(report.video.duration_sec, 2)
    record["aspect"] = report.video.aspect_ratio
    record["shot_count"] = len(report.shots)
    record["shot_durations"] = [round(s.boundary.duration_sec, 2) for s in report.shots]
    record["motion"] = sorted({
        s.motion.kind for s in report.shots if s.motion is not None
    })
    record["framing"] = sorted({
        s.framing.kind for s in report.shots if s.framing is not None
    })
    record["genre"] = ir["meta"]["genre"]
    record["shot_mode"] = ir["meta"]["shot_mode"]
    record["scene_type"] = ir["locked"]["B1_scene"]["type"]
    record["quality_preset"] = ir["locked"]["B5_quality"]["preset"]
    record["has_logline"] = bool(ir["creative"]["C1_logline"])
    record["people_count"] = len(ir["locked"]["B2_person"])

    transcript = report.transcript
    record["asr_segments"] = len(transcript)
    record["asr_chars"] = sum(len(t.text) for t in transcript)
    record["asr_confidence"] = (
        round(statistics.fmean(t.confidence for t in transcript), 3) if transcript else None
    )
    record["asr_has_timestamps"] = all(t.has_timestamps for t in transcript) if transcript else None
    record["transcript_sample"] = transcript[0].text[:60] if transcript else ""

    # 台词落地到节拍的情况 —— 反映"能不能直接用"
    beats_with_line = sum(
        1 for b in ir["creative"]["C2_beats"] if (b.get("dialogue") or {}).get("line")
    )
    record["beats"] = len(ir["creative"]["C2_beats"])
    record["beats_with_line"] = beats_with_line

    library = ph.load_library()
    doc = assembler.assemble(ir, library)
    gate_report = gates.run_gates(ir, doc, library)
    record["char_count"] = doc.char_count
    record["strategy"] = doc.strategy.value
    record["gate_verdict"] = gate_report.verdict
    record["gate_blockers"] = [r.code for r in gate_report.blockers]
    record["gate_warnings"] = [r.code for r in gate_report.warnings]
    record["mandatory_covered"] = sum(
        1 for item in ph.check_mandatory_dimensions(ir, library, prompt_text=doc.body_text)
        if item.covered
    )
    record["degradations"] = len(report.degradations)
    record["ok"] = True
    _ = ir
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok")]
    durations = [r["duration_s"] for r in ok]

    def dist(key: str) -> dict[str, int]:
        return dict(Counter(r[key] for r in ok).most_common())

    gate_verdicts = Counter(r["gate_verdict"] for r in ok)
    blocker_freq = Counter(code for r in ok for code in r["gate_blockers"])
    warning_freq = Counter(code for r in ok for code in r["gate_warnings"])

    return {
        "total": len(records),
        "succeeded": len(ok),
        "failed": len(failed),
        "over_15s": sum(1 for d in durations if d > 15),
        "avg_duration_s": round(statistics.fmean(durations), 2) if durations else 0,
        "avg_elapsed_s": round(statistics.fmean(r["elapsed_s"] for r in ok), 1) if ok else 0,
        "avg_shots": round(statistics.fmean(r["shot_count"] for r in ok), 2) if ok else 0,
        "asr_hit_rate": round(
            sum(1 for r in ok if r["asr_segments"] > 0) / len(ok), 3
        ) if ok else 0,
        "logline_rate": round(sum(1 for r in ok if r["has_logline"]) / len(ok), 3) if ok else 0,
        "avg_char_count": round(statistics.fmean(r["char_count"] for r in ok), 1) if ok else 0,
        "gate_verdicts": dict(gate_verdicts),
        "blocker_frequency": dict(blocker_freq.most_common()),
        "warning_frequency": dict(warning_freq.most_common()),
        "genre_distribution": dist("genre"),
        "shot_mode_distribution": dist("shot_mode"),
        "quality_distribution": dist("quality_preset"),
        "asr_confidence_avg": round(
            statistics.fmean(r["asr_confidence"] for r in ok if r["asr_confidence"]), 3
        ) if any(r["asr_confidence"] for r in ok) else None,
        "failures": [{"file": r["file"], "error": r.get("error", "")} for r in failed],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="反推引擎批量体检")
    parser.add_argument("--dir", default="测试素材")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    parser.add_argument("--out", default="assets/regression/baseline.json")
    parser.add_argument("--no-llm", action="store_true", help="不调模型，只测 L0 技术层")
    args = parser.parse_args()

    from engine.app.config import Settings
    from engine.app.pipeline.infer import build_inferencer

    source = Path(args.dir)
    if not source.is_dir():
        print(f"[X] 目录不存在：{source}")
        return 1

    videos = sorted(
        p for p in source.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES
    )
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"[X] {source} 里没有视频文件")
        return 1

    settings = Settings.load()
    inferencer = build_inferencer(settings, with_llm=not args.no_llm)
    workdir = settings.workspace_dir / "benchmark"

    print(f"批量体检：{len(videos)} 条视频 @ {source}")
    print(f"  模型 {'关闭（仅 L0 技术层）' if args.no_llm else settings.model_vision}")
    print("─" * 78)

    records: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        record = analyze_one(inferencer, video, workdir / f"run{index}")
        records.append(record)
        if record.get("ok"):
            print(
                f"[{index:2}/{len(videos)}] {video.name[:34]:<34} "
                f"{record['duration_s']:6.2f}s {record['shot_count']}镜头 "
                f"{record['shot_mode']:<20} {record['gate_verdict']:<20} "
                f"{record['elapsed_s']:5.1f}s"
            )
        else:
            print(f"[{index:2}/{len(videos)}] {video.name[:34]:<34} ✗ {record['error'][:40]}")

    summary = summarize(records)
    report = {
        "note": "反推引擎批量体检结果。缺人工标注，因此只覆盖可自动测量的维度。",
        "llm_enabled": not args.no_llm,
        "model": None if args.no_llm else settings.model_vision,
        "summary": summary,
        "records": records,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("─" * 78)
    print(f"成功 {summary['succeeded']}/{summary['total']}"
          f"　平均时长 {summary['avg_duration_s']}s"
          f"　平均镜头 {summary['avg_shots']}"
          f"　平均耗时 {summary['avg_elapsed_s']}s")
    print(f"超 15 秒（需拆条）{summary['over_15s']} 条"
          f"　ASR 命中 {summary['asr_hit_rate']:.0%}"
          f"　logline 产出 {summary['logline_rate']:.0%}"
          f"　平均 {summary['avg_char_count']} 字")
    print(f"闸门判定 {summary['gate_verdicts']}")
    print(f"阻断项频次 {summary['blocker_frequency']}")
    print(f"告警项频次 {summary['warning_frequency']}")
    print(f"镜头模式分布 {summary['shot_mode_distribution']}")
    if summary["failures"]:
        print("失败：")
        for item in summary["failures"]:
            print(f"  - {item['file']}: {item['error'][:70]}")
    print(f"\n结果 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
