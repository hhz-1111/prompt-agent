"""在标注数据上搜索最优分档阈值，并检查阈值是否落在**空档**里。

为什么单独一个脚本：调阈值最容易变成"凑分"。这里做两件事防止自欺：

1. **穷举**所有候选切点，报告最优解与次优解的差距 —— 差距小就说明数据分不开；
2. **检查切点两侧最近的样本距离** —— 阈值两边都空着（margin 大）才算真的分开了；
   贴着样本切的阈值是过拟合，换个素材就崩。

只在 16 个标注样本上做，结论**必须**标注样本量小。

用法：
    .venv\\Scripts\\python.exe scripts\\fit_framing_bands.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALIBRATION = ROOT / "workspace" / "benchmark" / "calibration.json"
ORDER = ["全景", "中全景", "中景", "中近景", "近景", "脸部特写"]


def main() -> int:
    if not CALIBRATION.exists():
        print(f"[X] 缺 {CALIBRATION}，先跑 scripts/calibrate_dimensions.py", file=sys.stderr)
        return 1
    records = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    samples = [
        (float(r["person_ratio"]), str(r["truth"]["景别"]))
        for r in records
        if r.get("person_ratio") is not None and r["truth"].get("景别")
    ]
    if not samples:
        print("[X] 没有可用的景别标注样本", file=sys.stderr)
        return 1

    print(f"样本 {len(samples)} 个，按高占比排序：")
    for ratio, label in sorted(samples):
        print(f"  {ratio:.2f}  {label}")

    print("\n" + "=" * 72)
    print("一、三档（全景 / 中景 / 近景）—— 数据支撑的粒度")
    print("=" * 72)
    _fit(samples, ["全景", "中景", "近景"])

    print("\n" + "=" * 72)
    print("二、六档（框架当前口径）—— 看数据支不支持这么细")
    print("=" * 72)
    _fit(samples, ORDER)

    return 0


def _predict(ratio: float, cuts: tuple[float, ...], order: list[str]) -> str:
    """按切点分档。

    ``order`` 是**由远到近**的档位表（全景在最前），``cuts`` 升序。
    ratio 越大 = 人物占画面越大 = 档位越近，所以落到第 i 段就给 ``order[i]``。
    """
    step = 0
    while step < len(cuts) and ratio >= cuts[step]:
        step += 1
    return order[min(step, len(order) - 1)]


def _fit(samples: list[tuple[float, str]], order: list[str]) -> None:
    values = sorted({ratio for ratio, _ in samples})
    if len(values) < 2:
        print("  样本不足以搜索")
        return
    # 候选切点取样本值之间的中点，外加两端之外各一个
    candidates = [values[0] - 0.05]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates.append(values[-1] + 0.05)

    n_cuts = len(order) - 1
    best: list[tuple[int, tuple[float, ...]]] = []
    total = len(samples)
    for cuts in combinations(candidates, n_cuts):
        hit = sum(1 for ratio, label in samples if _predict(ratio, cuts, order) == label)
        best.append((hit, cuts))
    best.sort(key=lambda item: (-item[0], item[1]))

    top_hit, top_cuts = best[0]
    print(f"  最优切点 {tuple(round(c, 3) for c in top_cuts)}"
          f"　命中 {top_hit}/{total} = {top_hit / total:.0%}")

    # 次优解（切点不同但命中数相同的，只报第一个不同解）
    alternatives = [item for item in best if item[0] < top_hit]
    if alternatives:
        second_hit, second_cuts = alternatives[0]
        print(f"  次优切点 {tuple(round(c, 3) for c in second_cuts)}"
              f"　命中 {second_hit}/{total} = {second_hit / total:.0%}"
              f"　（最优仅领先 {top_hit - second_hit} 个样本）")
    else:
        print("  没有更差的切点组合 —— 说明这组切点是唯一解（或样本太少）")

    # 每个切点两侧最近的样本距离：margin 大才说明真的分开
    print("  各切点的样本间隔（margin 越大越可信）：")
    for cut in top_cuts:
        below = max((v for v in values if v < cut), default=None)
        above = min((v for v in values if v > cut), default=None)
        margin = (above - below) if (below is not None and above is not None) else None
        print(f"    {cut:.3f}　下方最近样本 {below if below is not None else '—'}"
              f"　上方最近样本 {above if above is not None else '—'}"
              f"　间隔 {f'{margin:.2f}' if margin is not None else '—'}")

    print("  逐样本比对：")
    for ratio, label in sorted(samples):
        guess = _predict(ratio, top_cuts, order)
        print(f"    {ratio:.2f}  标注={label:<8} 反推={guess:<8}{'✓' if guess == label else '✗'}")


if __name__ == "__main__":
    raise SystemExit(main())
