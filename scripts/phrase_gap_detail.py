"""针对 phrase_overlap 报出的"零复现"模块，逐个查真因。

`phrase_overlap` 用的是"整句子串/相似度"，一条模块只要有一处措辞不同就会
报"没写"。这个脚本反过来做：对每条待查模块，取它的**关键词**，
在反推提示词里逐条搜，把命中的原句打出来，让人自己判断
到底是"真没写"还是"写法不同"。

用法：
    .venv\\Scripts\\python.exe scripts\\phrase_gap_detail.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "workspace" / "benchmark" / "prompts"
OVERLAP = ROOT / "workspace" / "benchmark" / "phrase_overlap.json"

# 每条待查模块 → 用来定位的关键词（任一个出现就把那句打出来）
WATCH: dict[str, tuple[str, ...]] = {
    "不添加话筒、提词器、道具等未说明元素": ("话筒", "提词器"),
    "不出现台下观众、其他人物、额外设备": ("观众", "额外设备"),
    "鼻翼和脸颊有真实的毛孔细节和细小肌理，整体皮肤质感清晰、真实、自然光照": ("毛孔", "鼻翼"),
    "参考中的动作，1:1复刻的肢体轨迹、身体角度、抬手弯腰幅度，不出现任何其他动作，动作自然连贯": ("1:1", "肢体轨迹", "动作复刻"),
    "真实相机拍摄质感": ("拍摄质感", "相机质感"),
    "眨眼：人物每3-4秒眨眼一次，每次持续0.1秒": ("眨眼",),
    "不出现字幕": ("字幕", "透明字幕"),
    "禁止画面出现黑色图形和黑色方块": ("黑色", "黑块"),
    "人物形象参考多视图，锁定人物五官脸型，全程同一人物，五官稳定不漂移": ("五官稳定", "多视图"),
    "皮肤质感：光影自然，人物面光自然充足": ("面光", "通透"),
    "标红处为视频呈现的场景、人物主体和音频对话内容可根据需求更改": ("标红", "红色字体"),
}


def main() -> int:
    if not OVERLAP.exists():
        print("[X] 先跑 scripts/phrase_overlap.py", file=sys.stderr)
        return 1
    prompts = {p.stem: p.read_text(encoding="utf-8") for p in PROMPTS.glob("*.txt")}
    if not prompts:
        print("[X] 没有反推提示词", file=sys.stderr)
        return 1
    sample = "菜市场采访-苏"
    body = prompts.get(sample) or next(iter(prompts.values()))

    print(f"逐条核查（样本视频：{sample}）")
    print("=" * 96)
    for block, keywords in WATCH.items():
        print(f"\n▸ 模块：{block[:64]}")
        found = False
        for keyword in keywords:
            for line in body.splitlines():
                if keyword in line:
                    print(f"    [{keyword}] {line.strip()[:110]}")
                    found = True
        if not found:
            print(f"    ❌ 关键词 {keywords} 一处都没出现 → **真的没写**")

    print("\n" + "=" * 96)
    print("再看全量：每条模块在 24 条反推里的总出现次数（用关键词粗查）")
    print("=" * 96)
    for block, keywords in WATCH.items():
        hits = sum(
            1 for text in prompts.values()
            if any(keyword in text for keyword in keywords)
        )
        bar = "█" * round(20 * hits / len(prompts))
        print(f"  {hits:>3}/{len(prompts):<3} {bar:<21}{block[:52]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
