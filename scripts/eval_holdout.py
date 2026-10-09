"""留出集评测：拿"框架外"的片子量反推质量。

## 为什么需要它（这是所有改动的尺子）

本轮吃过一次亏：景别那个改动，逻辑上很顺，改完却**更糟**
（`a4ed` 把真人排除掉、去量大屏上的照片）。原因是**没有真值** ——
只能靠个案感觉判断好坏，等于每次改动都在赌。

所以评测集要满足两点：

1. **片子必须在框架之外**：不能是参与过框架校准的那批（训练集成绩没有意义）；
2. **每条片子都要有人工写的"必须包含什么"**：关键事实（人数/人物/位置/台词）、
   关键约束（永不可裁项）、以及"绝不能出现什么"。

## 怎么用

    # 1) 生成草稿（跑一遍反推，把结果整理成待人工修正的要点表）
    python scripts/eval_holdout.py --draft "测试素材/留出集/新片子.mp4" --name 产品特写

    # 2) 人工把草稿改成"必须包含什么"（草稿只是省你打字，**必须人工核一遍**）
    #    改完保存为 assets/eval/holdout.json

    # 3) 打分（默认用已有项目的存储 IR，不重跑反推；加 --infer 才会跑模型）
    python scripts/eval_holdout.py
    python scripts/eval_holdout.py --infer        # 端到端重跑（慢、花模型调用）

分数只回答一个问题：**这条片子的提示词，该有的有没有、不该有的有没有。**
它不评价文笔，也不评价生成效果（那是下一层，需要平台回流数据）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.app.config import Settings  # noqa: E402
from engine.app.store import ProjectStore  # noqa: E402
from engine.core import assembler, gates, phrases as ph  # noqa: E402

MANIFEST = Path("assets/eval/holdout.json")
DRAFT_DIR = Path("assets/eval/drafts")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "cases": []}
    return json.loads(path.read_text(encoding="utf-8"))


def score_case(text: str, case: dict[str, Any]) -> dict[str, Any]:
    """按人工要点打分。命中率是唯一指标 —— 不做"看起来不错"的主观评价。"""
    must_include = [str(x) for x in case.get("must_include", [])]
    must_avoid = [str(x) for x in case.get("must_avoid", [])]
    hit = [item for item in must_include if item in text]
    miss = [item for item in must_include if item not in text]
    leaked = [item for item in must_avoid if item in text]
    total = len(must_include) + len(must_avoid)
    ok = len(hit) + (len(must_avoid) - len(leaked))
    return {
        "name": case.get("name") or case.get("id") or "未命名",
        "framework_fit": bool(case.get("framework_fit", True)),
        "total": total,
        "ok": ok,
        "score": (ok / total) if total else 0.0,
        "hit": hit,
        "miss": miss,
        "leaked": leaked,
    }


def _text_of(ir: dict[str, Any], library: ph.PhraseLibrary) -> str:
    return assembler.assemble(ir, library).full_text()


def _ir_for_case(
    case: dict[str, Any],
    store: ProjectStore,
    settings: Settings,
    *,
    run_inference: bool,
) -> tuple[dict[str, Any] | None, str]:
    """取这条片子的 IR。优先用已有项目（免费）；`--infer` 才真跑模型。"""
    project_id = str(case.get("project_id") or "")
    if project_id:
        project = store.get(project_id)
        if project is not None and project.ir:
            return project.ir, f"项目 {project_id}"
    video = str(case.get("video") or "")
    if not video or not Path(video).exists():
        return None, "缺视频/缺 project_id"
    if not run_inference:
        return None, "没有项目 IR（加 --infer 现场跑一遍）"
    from engine.app.pipeline.infer import build_inferencer

    inferencer = build_inferencer(settings)
    ir, _report = inferencer.infer_to_ir(video, workspace=settings.workspace_dir)
    return ir, f"现场反推 {Path(video).name}"


def draft_case(video: str, name: str, settings: Settings) -> dict[str, Any]:
    """生成**草稿**要点表 —— 只是省你打字，必须人工核对后才算数。

    草稿里放的是"当前管线的产出"，恰好就是我们要检验的东西：
    如果人工只是照抄草稿，这个评测集就变成了自我印证，没有意义。
    所以草稿会带一句 `_TODO` 提示。
    """
    from engine.app.pipeline.infer import build_inferencer

    inferencer = build_inferencer(settings)
    ir, report = inferencer.infer_to_ir(video, workspace=settings.workspace_dir)
    library = ph.load_library()
    doc = assembler.assemble(ir, library)
    text = doc.full_text()
    gate_report = gates.run_gates(ir, doc, library)
    beats = ir.get("creative", {}).get("C2_beats", [])
    people = ir.get("locked", {}).get("B2_person", [])
    return {
        "id": Path(video).stem,
        "name": name or Path(video).stem,
        "video": video,
        "framework_fit": None,
        "_TODO": "人工把 must_include 改成『这条片子的提示词必须包含什么』（事实/约束/台词），"
                 "must_avoid 写『绝不能出现什么』；framework_fit 填 true/false："
                 "这条片子是否套得进现有框架。**不要照抄草稿**。",
        "must_include": [
            f"共 {len(beats)} 个镜头",
            *[f"{p.get('name')}：{p.get('position')}" for p in people if p.get("name")],
            *[f"台词：{str((b.get('dialogue') or {}).get('line'))[:20]}"
              for b in beats if (b.get("dialogue") or {}).get("line")],
        ],
        "must_avoid": ["unknown", "【"],
        "_draft_text": text[:2000],
        "_gate_verdict": gate_report.verdict,
        # **为什么被拦也要记下来**：分数只说明"命中了多少"，
        # 闸门说明"哪里还没过关"，两者合起来才是一次可用的体检。
        "_gate_blockers": [f"{r.code} {r.name}：{r.detail}" for r in gate_report.blockers],
        "_gate_warnings": [f"{r.code} {r.name}：{r.detail}" for r in gate_report.warnings],
        "_report_degradations": list(getattr(report, "degradations", ())),
        "_report_reviews": [
            f"[{'已改' if n.applied else n.action}] {n.path}｜{n.reason}｜{n.detail}"
            for n in getattr(report, "reviews", ())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="留出集评测")
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--draft", nargs="+", help="为这些视频生成草稿要点表")
    parser.add_argument("--name", default="", help="草稿的名字")
    parser.add_argument("--infer", action="store_true", help="名单里没有项目 IR 时现场反推")
    parser.add_argument("--json", action="store_true", help="输出机读结果")
    args = parser.parse_args()

    settings = Settings.load()
    library = ph.load_library()
    store = ProjectStore(settings.workspace_dir / "projects")

    if args.draft:
        DRAFT_DIR.mkdir(parents=True, exist_ok=True)
        for video in args.draft:
            case = draft_case(video, args.name, settings)
            out = DRAFT_DIR / f"{case['id']}.json"
            out.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[草稿] {out}（**必须人工核对后再并入 {args.manifest}**）")
        return 0

    manifest = load_manifest(Path(args.manifest))
    cases = manifest.get("cases", [])
    if not cases:
        print(f"[空] {args.manifest} 里还没有用例。")
        print("步骤：① 放 6~8 条**框架外**的片子；② --draft 生成草稿；"
              "③ 人工改成『必须包含什么』；④ 再跑本脚本打分。")
        return 1

    results: list[dict[str, Any]] = []
    for case in cases:
        ir, source = _ir_for_case(case, store, settings, run_inference=args.infer)
        if ir is None:
            print(f"[跳过] {case.get('name')}: {source}")
            continue
        text = _text_of(ir, library)
        doc = assembler.assemble(ir, library)
        report = gates.run_gates(ir, doc, library)
        row = score_case(text, case)
        row.update({
            "source": source,
            "verdict": report.verdict,
            "chars": doc.char_count,
            "content_chars": doc.content_char_count,
        })
        results.append(row)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if not results:
        print("没有可评测的用例。")
        return 1
    print(f"{'用例':22s} {'框架内':6s} {'要点命中':10s} {'得分':6s} {'闸门':18s} 字数")
    print("-" * 78)
    for row in results:
        fit = "是" if row["framework_fit"] else "**否**"
        print(f"{row['name'][:20]:22s} {fit:6s} "
              f"{str(row['ok']) + '/' + str(row['total']):10s} "
              f"{row['score']:6.0%} {row['verdict']:18s} {row['content_chars']}")
        for item in row["miss"]:
            print(f"    缺：{item}")
        for item in row["leaked"]:
            print(f"    不该出现：{item}")
    overall = sum(r["ok"] for r in results) / max(1, sum(r["total"] for r in results))
    outside = [r for r in results if not r["framework_fit"]]
    print("-" * 78)
    print(f"总分：{overall:.0%}（{sum(r['ok'] for r in results)}/{sum(r['total'] for r in results)}）")
    if outside:
        outside_score = sum(r["ok"] for r in outside) / max(1, sum(r["total"] for r in outside))
        print(f"其中**框架外**用例：{outside_score:.0%}（{len(outside)} 条）"
              f" ← 这个数字才是 agent 能力的真实刻度")
    else:
        print("注意：还没有标 framework_fit=false 的用例 —— 全是框架内的片子，"
              "分数再高也说明不了 agent 能力。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
