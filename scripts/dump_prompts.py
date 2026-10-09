"""把 24 条测试视频按当前代码全部反推一遍，存下**最终提示词文本**。

存在的理由：`benchmark.py` 只存了聚合指标（字数、闸门、镜头数），
不存正文。而"反推出来的提示词到底长什么样"才是这个项目要验收的东西 ——
尤其是要对着历史成功提示词（`_extract/`）比"固定模块有没有被复现"。

产出：
    workspace/benchmark/prompts/<视频名>.txt   最终提示词正文
    workspace/benchmark/prompts.json           正文 + 逐镜头原始判定

用法：
    .venv\\Scripts\\python.exe scripts/dump_prompts.py
    .venv\\Scripts\\python.exe scripts\\dump_prompts.py --limit 3
    .venv\\Scripts\\python.exe scripts\\dump_prompts.py --no-llm   # 只跑 L0，很快
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 24 条视频的反推提示词正文")
    parser.add_argument("--dir", default="测试素材")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=[], help="只跑这些文件名")
    parser.add_argument("--out", default="workspace/benchmark")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    from engine.app.config import Settings
    from engine.app.pipeline.infer import build_inferencer
    from engine.core import assembler, gates, phrases as ph

    settings = Settings.load()
    inferencer = build_inferencer(settings, with_llm=not args.no_llm)
    library = ph.load_library()

    source = Path(args.dir)
    videos = sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if args.only:
        wanted = set(args.only)
        videos = [v for v in videos if v.name in wanted]
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print("[X] 目录里没有视频", file=sys.stderr)
        return 1

    out_root = Path(args.out)
    prompt_dir = out_root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    print(f"导出反推提示词：{len(videos)} 条 @ {source}")
    print(f"  模型 {'关闭（仅 L0）' if args.no_llm else settings.model_vision}")
    print("─" * 92)

    for index, video in enumerate(videos, start=1):
        started = time.time()
        record: dict = {"file": video.name}
        try:
            ir, report = inferencer.infer_to_ir(
                str(video), workspace=settings.workspace_dir / "dump"
            )
        except Exception as exc:  # noqa: BLE001
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            records.append(record)
            print(f"[{index:2}/{len(videos)}] {video.name[:32]:<32} ✗ {record['error'][:40]}")
            continue

        document = assembler.assemble(ir, library)
        gate_report = gates.run_gates(ir, document, library)

        (prompt_dir / f"{video.stem}.txt").write_text(
            document.full_text(), encoding="utf-8"
        )

        flow = _motion_snapshot(inferencer, report)
        record.update(
            ok=True,
            elapsed_s=round(time.time() - started, 1),
            duration_s=round(report.video.duration_sec, 2),
            shot_count=len(report.shots),
            char_count=document.char_count,
            strategy=document.strategy.value,
            gate_verdict=gate_report.verdict,
            gate_blockers=[r.code for r in gate_report.blockers],
            gate_warnings=[r.code for r in gate_report.warnings],
            genre=ir["meta"]["genre"],
            shot_mode=ir["meta"]["shot_mode"],
            scene_type=ir["locked"]["B1_scene"]["type"],
            quality_preset=ir["locked"]["B5_quality"]["preset"],
            people=[p.get("name", "") for p in ir["locked"]["B2_person"]],
            positions=ir["locked"]["B3_blocking"]["positions"],
            framing=[s.framing.kind if s.framing else "" for s in report.shots],
            motion=flow,
            composition=[s.composition for s in report.shots],
        )
        records.append(record)
        print(
            f"[{index:2}/{len(videos)}] {video.name[:32]:<32} "
            f"{record['shot_count']}镜 {record['char_count']:>5}字 "
            f"{record['gate_verdict']:<20} {record['elapsed_s']:>5.1f}s"
        )

    (out_root / "prompts.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ok = sum(1 for r in records if r.get("ok"))
    print("─" * 92)
    print(f"成功 {ok}/{len(records)}　正文 → {prompt_dir}　索引 → {out_root / 'prompts.json'}")
    return 0


def _motion_snapshot(inferencer, report) -> list[dict]:
    """顺带存下光流量测，后面比"提示词里写的运镜"和"量测到的运镜"是否一致。"""
    analyzer = getattr(inferencer, "motion_analyzer", None)
    measurements = getattr(analyzer, "measurements", {}) or {}
    snapshot = []
    for signal in report.shots:
        measurement = measurements.get(signal.boundary.index)
        snapshot.append({
            "shot": signal.boundary.index,
            "kind": signal.motion.kind if signal.motion else "",
            "confidence": signal.motion.confidence if signal.motion else 0.0,
            "translation": round(measurement.translation, 3) if measurement else None,
            "p10": round(measurement.p10_translation, 3) if measurement else None,
            "divergence": round(measurement.divergence, 3) if measurement else None,
        })
    return snapshot


if __name__ == "__main__":
    raise SystemExit(main())
