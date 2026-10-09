"""一次性诊断：答案卷（_extract 的 10 份提示词）能不能对上考卷（测试素材 的 24 条视频）。

做维度级准确率回归（PRD §6.1.5）需要 (视频, 标准答案) 配对。
本脚本不猜，只把**可机器比对的结构特征**摆出来，让人来判断。

三条线索：
  1. 答案卷正文里直接写着的 .mp4 文件名（最强证据）
  2. 镜头数：答案卷的分镜条数 vs 视频的实测切分镜头数
  3. 一镜到底 / 同角度分段信号 vs 视频的 shot_mode

用法：
    .venv\\Scripts\\python.exe scripts\\pair_probe.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTRACT = ROOT / "_extract"
VIDEO_DIR = ROOT / "测试素材"
BASELINE = ROOT / "assets" / "regression" / "baseline.json"
ANALYSIS = EXTRACT / "_corpus-analysis.json"

MP4_RE = re.compile(r"[\w\u4e00-\u9fff ()（）\-]{3,80}?\.(?:mp4|MP4|mov)")

# 答案卷里点名了"分镜/镜头 N"的，这一条就是逐镜标准答案
SHOT_HEAD_RE = re.compile(r"(?:^|\n)\s*(?:镜头|分镜)\s*[0-9１-９一二三四五六七八九十]")


def answer_sheet_shot_count(text: str) -> int:
    """答案卷里显式列出的镜头条数（分镜段）。"""
    return len(SHOT_HEAD_RE.findall(text))


def video_refs(text: str) -> list[str]:
    """答案卷正文里直接引用的源视频文件名。"""
    seen: list[str] = []
    for raw in MP4_RE.findall(text):
        name = raw.strip()
        if name not in seen:
            seen.append(name)
    return seen


def main() -> int:
    if not BASELINE.exists() or not ANALYSIS.exists():
        print("缺少 baseline.json 或 _corpus-analysis.json，先跑 benchmark.py", file=sys.stderr)
        return 2

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    analysis = {a["name"]: a for a in json.loads(ANALYSIS.read_text(encoding="utf-8"))}
    videos = {r["file"]: r for r in baseline["records"]}
    video_names = set(videos)

    print("=" * 78)
    print("答案卷（10）  ×  考卷（24）  配对诊断")
    print("=" * 78)
    print("问的是一件事：这 10 份提示词，哪一份是这 24 条视频里某一条的「标准答案」？\n")

    print("── 线索 1：答案卷正文里直接写着的视频文件名 ──")
    linked: dict[str, list[str]] = {}
    for path in sorted(EXTRACT.glob("*.txt")):
        refs = video_refs(path.read_text(encoding="utf-8"))
        hit = [r for r in refs if r in video_names]
        linked[path.stem] = hit
        if refs:
            print(f"  {path.stem}")
            for r in refs:
                mark = "  ✅ 在测试素材/" if r in video_names else "  ❌ 测试素材里没有"
                print(f"      {r}{mark}")
    total_hits = sum(len(v) for v in linked.values())
    print(f"\n  结论：命中 {total_hits} 条。0 条意味着**没有任何一条视频是这些答案的直接素材**。\n")

    print("── 线索 2：镜头数比对（答案卷显式镜头条数 vs 视频实测切分）──")
    print(f"  {'答案卷':<14}{'镜头数':>6}   {'实测镜头数分布':<20}pipeline 判定的 shot_mode")
    for path in sorted(EXTRACT.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        n = answer_sheet_shot_count(text)
        a = analysis.get(path.stem, {})
        modes = [r["shot_mode"] for r in videos.values() if r["shot_count"] == n]
        mode_hint = "/".join(sorted(set(modes))) or "—"
        one_take = "一镜到底" if a.get("oneTakeSignals") else ""
        print(f"  {path.stem:<14}{n or '—':>6}   {str(n) + ' 镜的视频 ' + str(len(modes)) + ' 条':<20}{mode_hint} {one_take}")

    print("\n  实测镜头数分布（24 条）：")
    dist: dict[int, list[str]] = {}
    for r in videos.values():
        dist.setdefault(r["shot_count"], []).append(r["file"])
    for k in sorted(dist):
        print(f"      {k} 镜 × {len(dist[k]):>2}  {dist[k][0] if len(dist[k]) == 1 else ''}")

    print("\n── 线索 3：答案卷的题材 vs 视频实测题材 ──")
    for path in sorted(EXTRACT.glob("*.txt")):
        a = analysis.get(path.stem, {})
        presets = ",".join(a.get("presetCandidates", []))
        print(f"  {path.stem:<14}preset候选={presets or '—'}")

    print("\n" + "=" * 78)
    print("判定")
    print("=" * 78)
    if total_hits == 0:
        print("  答案卷引用的源视频，一条都不在 测试素材/ 里。")
        print("  两份素材是**不同批次**：答案卷是历史生成项目的提示词，")
        print("  考卷是另行收集的爆款视频。")
    else:
        print(f"  有 {total_hits} 条可以硬对应（答案卷点名了源视频）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
