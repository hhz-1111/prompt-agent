"""固定提示词模块对照：历史成功提示词（答案卷） vs 反推出来的提示词。

**为什么需要这个**：用户提供的测试素材，其反推提示词**逐字复用**了一批固定
模块（如"全程不生成任何字幕文字，100%透明字幕"、"每 3-4 秒眨眼一次"）。
所以两边可以用文本重合来对照 —— 这也解释了 `scripts/pair_probe.py`
为什么找不到源视频文件：复用的是**模块**，不是同一条片子。

## 一个踩过的测量坑（别重犯）

第一版用"整句子串 / 整句相似度"判断模块在不在，结果把大批模块报成
"零复现"，而实际上 24/24 条都写了 —— 只是措辞不同：

    "禁止人为添加话筒、提词器、课程器械等上游未提供的道具。"（反推）
    vs "不添加话筒、提词器、道具等未说明元素"（答案卷）

整句比对要求**整句**都对上，一句话里有半句不同就归零。
而人工写提示词本来就不会逐字照抄，所以这个口径量的是"抄得像不像"，
不是"有没有写"。

现在的口径是**行为表述级**：人工登记每条模块的"核心行为片段"，
只要片段逐字出现就算写了。这个口径可复核（`--verbose` 会打出命中原句），
而且不会因为半句措辞差异误报。

用法：
    .venv\\Scripts\\python.exe scripts\\phrase_overlap.py
    .venv\\Scripts\\python.exe scripts\\phrase_overlap.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTRACT = ROOT / "_extract"
PROMPTS = ROOT / "workspace" / "benchmark" / "prompts"
OUT = ROOT / "workspace" / "benchmark" / "phrase_overlap.json"

MIN_BLOCK_CHARS = 8
SPLIT_RE = re.compile(r"[\n。；;！!？?]+")
PLACEHOLDER_RE = re.compile(r"【[^】]{0,30}】|\[[^\]]{0,30}\]")
LEAD_RE = re.compile(r"^[\s\-·•*#>\d\.、（）()：:]+")
DIGIT_RE = re.compile(r"[0-9０-９]+(?:[.\-~～][0-9０-９]+)*")
EN_SPACE_RE = re.compile(r"(?<=[A-Za-z,])\s+(?=[A-Za-z,])")

# 每条固定模块的**判定片段**：任一逐字出现即算"这条被复现了"。
# 片段取的是行为表述的核心部分，而不是整句 —— 理由见模块开头。
PROBE_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "全程不生成任何字幕文字，100%透明字幕": ("100%透明字幕", "不生成任何字幕"),
    "禁止出现bgm、logo、水印、时间戳、分镜格线、批注文字或设计说明元素": ("bgm", "水印"),
    "禁止插画/线稿感、虚化、运动模糊、景深效果": ("插画", "虚化"),
    "禁止错误肢体如扭曲手指、多关节": ("扭曲手指",),
    "禁止改变人物身份或动作方向": ("人物身份",),
    "禁止无关人物或道具漂移": ("道具漂移",),
    "禁止添加未提供的人物性别、年龄、服装、道具或职业细节": ("未提供",),
    "禁止自行增补对白或口型停顿要求之外的台词": ("口型", "增补对白"),
    "不添加话筒、提词器、道具等未说明元素": ("话筒", "提词器"),
    "不出现台下观众、其他人物、额外设备": ("观众",),
    "禁止画面出现黑色图形和黑色方块": ("黑色图形", "黑块", "黑色方块"),
    "人物每隔3-4秒自然快速眨眼一次（每次约0.1秒），融入说话节奏，眨眼频率自然": ("眨眼",),
    "人物皮肤保留毛孔、轻微纹理与柔和光泽，无过度磨皮": ("毛孔", "磨皮"),
    "鼻翼和脸颊有真实的毛孔细节和细小肌理，整体皮肤质感清晰、真实、自然光照": ("鼻翼", "肌理"),
    "可以看到自然毛孔、轻微皮肤纹理和柔和光泽，肤质自然不过度磨皮": ("毛孔", "磨皮"),
    "皮肤质感：光影自然，人物面光自然充足": ("皮肤",),
    "人物形象参考多视图，锁定人物五官脸型，全程同一人物，五官稳定不漂移，服装身形匹配参考图，真实皮肤质感": (
        "五官稳定不漂移", "锁定人物五官",
    ),
    "人物外观魏巍老师保持形象，用于魏巍老师的五官细节、面部轮廓、外貌、发型、体态和气质参考，呈现视觉辨识度，保持自然真实质感": (
        "五官细节", "呈现视觉辨识度",
    ),
    "原生相机噪点，细节完整，动态流畅自然": ("噪点",),
    "真实相机拍摄质感": ("拍摄质感", "相机质感"),
    "画面中不出现字幕、logo、水印、时间戳、分镜格线、批注文字": ("水印",),
    "禁止字幕、logo、水印、时间戳、批注文字、设计说明": ("水印",),
    "画质：autowhitebalanceslightlycool,slightlyunderexposed,naturalphonecameranoisetexture": (
        "auto white balance", "natural phone camera",
    ),
    "参考中的动作，1:1复刻的肢体轨迹、身体角度、抬手弯腰幅度，不出现任何其他动作，动作自然连贯": (
        "肢体轨迹", "动作复刻", "不出现任何其他动作",
    ),
    "标红处为视频呈现的场景、人物主体和音频对话内容可根据需求更改": ("标红",),
    "注意：红色字体为可修改文案": ("可修改文案", "标红"),
    "符号：只是用于参考对象位置，不保留在文案里": ("不保留在文案", "参考对象位置"),
}


DOC_META = "文档约定（不是提示词内容，不该计入分母）"
TEMPLATE_ALT = "模板库写法差异（引擎只会输出模板库那一版）"
CONTENT_TYPE = "内容类型不匹配（该场景当前链路不做）"
SYNONYM = "同义重复登记（与另一条模块同义）"
REAL_GAP = "真缺失（反推该写没写）"

GAP_CLASSIFICATION: dict[str, str] = {
    "注意：红色字体为可修改文案": DOC_META,
    "标红处为视频呈现的场景、人物主体和音频对话内容可根据需求更改": DOC_META,
    "符号：只是用于参考对象位置，不保留在文案里": DOC_META,
    "人物每3-4秒眨眼一次，每次持续0.1秒，融入对话节奏，眨眼频率自然": TEMPLATE_ALT,
    "眨眼：人物每3-4秒眨眼一次，每次持续0.1秒": TEMPLATE_ALT,
    "眨眼设定：人物每隔3秒自然快速眨眼一次（每次约0.1秒），融入动作节奏，眨眼频率自然": (
        TEMPLATE_ALT
    ),
    "参考中的动作，1:1复刻的肢体轨迹、身体角度、抬手弯腰幅度，不出现任何其他动作，动作自然连贯": (
        CONTENT_TYPE
    ),
    "原生相机噪点，细节完整，动态流畅自然": SYNONYM,
    "皮肤质感：光影自然，人物面光自然充足，画面有通透感，可以看到自然毛孔、轻微皮肤纹理和柔和光泽，肤质自然不过度磨皮": (
        SYNONYM
    ),
    "人物外观魏巍老师保持形象，用于魏巍老师的五官细节、面部轮廓、外貌、发型、体态和气质参考，呈现视觉辨识度，保持自然真实质感": (
        SYNONYM
    ),
    "禁止出现字幕、logo、水印、时间戳、分镜格线、批注文字或设计说明元素": SYNONYM,
    "禁止画面出现黑色图形和黑色方块": REAL_GAP,
}


def normalize(text: str) -> str:
    text = PLACEHOLDER_RE.sub("", text)
    text = text.replace("　", " ")
    text = re.sub(r"\s+", "", text)
    text = LEAD_RE.sub("", text)
    # 句末句号在"是不是同一条模块"上没有信息量：答案卷里同一句
    # 有的带句号有的不带，不归一就会被当成两条模块，污染分母。
    return text.strip(" -·•*#>\t：。")


def canonical(text: str) -> str:
    """归一 + 抹掉数字差异，用于**归并同一条模块的不同写法**。"""
    return DIGIT_RE.sub("#", EN_SPACE_RE.sub("", normalize(text)))


def extract_blocks(text: str) -> list[str]:
    blocks = []
    for raw in SPLIT_RE.split(text):
        cleaned = normalize(raw)
        if len(cleaned) >= MIN_BLOCK_CHARS:
            blocks.append(cleaned)
    return list(dict.fromkeys(blocks))


def main() -> int:
    parser = argparse.ArgumentParser(description="固定提示词模块对照")
    parser.add_argument("--min-docs", type=int, default=2,
                        help="一个模块至少出现在几份答案卷里才算固定模块")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--verbose", action="store_true", help="打印每条命中/缺失的原句")
    args = parser.parse_args()

    if not PROMPTS.is_dir():
        print("[X] 还没有反推提示词，先跑 scripts/dump_prompts.py", file=sys.stderr)
        return 1

    # ---- 1. 抽固定模块 ----
    sheets: dict[str, list[str]] = {}
    for path in sorted(EXTRACT.glob("*.txt")):
        sheets[path.stem] = extract_blocks(path.read_text(encoding="utf-8"))
    if not sheets:
        print("[X] _extract 里没有答案卷", file=sys.stderr)
        return 1

    canonical_docs: dict[str, set[str]] = defaultdict(set)
    canonical_sample: dict[str, str] = {}
    for stem, blocks in sheets.items():
        for block in blocks:
            key = canonical(block)
            canonical_docs[key].add(stem)
            if key not in canonical_sample or len(block) < len(canonical_sample[key]):
                canonical_sample[key] = block

    fixed_keys = {k for k, docs in canonical_docs.items() if len(docs) >= args.min_docs}
    fixed = sorted(canonical_sample[k] for k in fixed_keys)

    print(f"答案卷 {len(sheets)} 份，抽出候选句子 {len(canonical_docs)} 条（已按数字/写法归并）")
    print(f"  跨 ≥{args.min_docs} 份重复的**固定模块** {len(fixed)} 条"
          f"　只出现 1 份的各片专有内容 {len(canonical_docs) - len(fixed)} 条")

    # ---- 2. 逐视频算复现率 ----
    prompts = {p.stem: p.read_text(encoding="utf-8") for p in PROMPTS.glob("*.txt")}
    if not prompts:
        print("[X] prompts 目录是空的", file=sys.stderr)
        return 1

    def matched(block: str, text: str) -> tuple[bool, str]:
        """判定片段是否出现。

        两侧都要过同一套规范化，否则会误报成"没写"：
        答案卷写 `画质：auto white balance slightly cool`（有空格），
        反推写同一串但空格被规范化掉了 —— 只归一一边就永远对不上。

        **规范化后变成空串的片段必须跳过**：空串是任何字符串的子串，
        会让这条模块永远"命中"。踩过这个坑：片段 `1:1` 里的数字被
        ``DIGIT_RE`` 抹成 `#`，两边冒号再被剥掉，最后剩下空串 ——
        于是"动作1:1复刻"这条模块在所有视频上都报命中，而实际一条都没写。
        """
        haystack = canonical(text)
        fragments = PROBE_FRAGMENTS.get(block)
        if not fragments:
            return (canonical(block) in haystack, "整句")
        for fragment in fragments:
            key = canonical(fragment)
            if key and len(key) >= 2 and key in haystack:
                return True, fragment
        return False, ""

    records = []
    for name, text in prompts.items():
        hit, miss = [], []
        for block in fixed:
            ok, why = matched(block, text)
            (hit if ok else miss).append({"block": block, "via": why})
        records.append({
            "video": name,
            "chars": len(text),
            "total": len(fixed),
            "hit": len(hit),
            "coverage": round(len(hit) / len(fixed), 3) if fixed else 0.0,
            "hit_blocks": [item["block"] for item in hit],
            "missing": [item["block"] for item in miss],
        })
    records.sort(key=lambda r: -r["coverage"])

    print("\n" + "=" * 100)
    print("固定模块复现率（判定口径：行为片段逐字出现）")
    print("=" * 100)
    for record in records:
        bar = "█" * round(30 * record["coverage"])
        print(f"  {record['hit']:>3}/{record['total']:<3} {record['coverage']:>5.0%} "
              f"{bar:<31}{record['video'][:30]}")
    if records:
        average = sum(r["coverage"] for r in records) / len(records)
        print(f"\n  平均复现率 {average:.0%}")

    # ---- 3. 逐条模块 ----
    per_block: dict[str, int] = {b: 0 for b in fixed}
    for record in records:
        for block in record["hit_blocks"]:
            per_block[block] += 1
    print("\n" + "=" * 100)
    print("每条固定模块在多少条反推里被复现（最缺的在最前）")
    print("=" * 100)
    for block, count in sorted(per_block.items(), key=lambda kv: kv[1]):
        bar = "█" * round(24 * count / len(records)) if records else ""
        note = GAP_CLASSIFICATION.get(block)
        suffix = f"　← {note}" if note and count == 0 else ""
        print(f"  {count:>3}/{len(records):<3} {bar:<25}{block[:50]}{suffix}")

    # ---- 3b. 零复现的模块分类：不是所有"缺"都该算在算法头上 ----
    gap_classes: dict[str, list[str]] = defaultdict(list)
    for block, count in per_block.items():
        if count == 0:
            gap_classes[GAP_CLASSIFICATION.get(block, "未分类（需人工判断）")].append(block)
    print("\n" + "=" * 100)
    print("零复现模块的分类（**这个分类比数字重要**）")
    print("=" * 100)
    for kind, blocks in sorted(gap_classes.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  【{kind}】{len(blocks)} 条")
        for block in blocks:
            print(f"      {block[:66]}")
    scored = len(fixed) - len(gap_classes.get(DOC_META, []))
    print(f"\n  可用于评判反推的模块 = {len(fixed)} - {len(gap_classes.get(DOC_META, []))}"
          f"（文档约定）= {scored} 条")

    if args.verbose:
        sample = records[0]["video"] if records else ""
        print(f"\n{'=' * 100}\n逐句核查（样本：{sample}）\n{'=' * 100}")
        text = prompts.get(sample, "")
        for block in fixed:
            ok, why = matched(block, text)
            mark = f"✓ 命中片段「{why}」" if ok else "✗ 未命中"
            print(f"  {mark:<22}{block[:62]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sheets": {k: len(v) for k, v in sheets.items()},
        "fixed_blocks": fixed,
        "block_presence": per_block,
        "videos": records,
        "probe_fragments": {k: list(v) for k, v in PROBE_FRAGMENTS.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n明细 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
