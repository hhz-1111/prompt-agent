"""语料入库：把 _extract 里的样本原文归档进 assets/corpus，并重算跨样本统计。

设计原则：**样本是 ground truth，只读不改。**
脚本只做三件事——归档原文、抽取结构化信号、产出统计证据，
绝不"顺手美化"原文（原文任何改动都会让回归基准失去意义）。

用法：
    py scripts/ingest_corpus.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "_extract"
CORPUS = ROOT / "assets" / "corpus"
RAW = CORPUS / "raw"

# 文档名 → 展示名与已知的结构族元信息（来自 TEMPLATES §7 的两大结构族）
DOC_MAP: dict[str, dict[str, Any]] = {
    "AI情景对话": {"title": "AI情景对话", "family": "A", "genre": "dialogue"},
    "IP教学": {"title": "IP教学", "family": "A", "genre": "teaching"},
    "IP演播厅": {"title": "IP演播厅", "family": "A", "genre": "studio"},
    "主持人访谈-苏": {"title": "主持人访谈-苏", "family": "B", "genre": "studio"},
    "偷拍视角-苏": {"title": "偷拍视角-苏", "family": "B", "genre": "dialogue"},
    "原生-夫妻对话": {"title": "原生-夫妻对话", "family": "A", "genre": "dialogue"},
    "多视角提示词": {"title": "多视角提示词", "family": "A", "genre": "teaching"},
    "提示词": {"title": "提示词", "family": "A", "genre": "replacement"},
    "替换": {"title": "替换", "family": "A", "genre": "replacement"},
    "素人街采-苏": {"title": "素人街采-苏", "family": "B", "genre": "street_interview"},
}

# ---- 抽取用的信号词表（与 PHRASES/TEMPLATES 的维度定义对齐）----
SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "关键一致性": ("关键一致性", "一致性规则", "一致性"),
    "故事节拍": ("故事节拍", "分镜", "镜头", "故事内容"),
    "对白与口型": ("对白与口型", "口型和对白", "对白"),
    "禁止项": ("禁止项",),
    "简短剧情": ("简短剧情", "故事主线", "剧情"),
}

LOCK_SIGNALS: dict[str, tuple[str, ...]] = {
    "站位": ("始终在画面左侧", "始终在画面右侧", "左边", "右边", "不交换位置"),
    "视线": ("视线", "望向"),
    "遮挡": ("遮挡",),
    "场景锚定": ("场景始终为", "保持同一位置和光线氛围"),
    "眨眼": ("眨眼",),
    "皮肤质感": ("毛孔", "磨皮", "皮肤质感"),
    "音色": ("音色",),
    "画面比例": ("9:16", "竖屏"),
    "一镜到底锁定": ("一镜到底", "机位全程不变", "镜头无跳切", "不推拉"),
}

BEAT_SIGNALS: dict[str, tuple[str, ...]] = {
    "hook": ("钩子", "竟然", "没想到", "震惊", "大变样", "遗憾通知"),
    "setup": ("铺垫", "话说", "是这样的"),
    "demo": ("示范", "动作", "跟练", "展示"),
    "proof": ("证明", "真的", "效果", "变化"),
    "cta": ("点击", "下方链接", "报名", "课程", "名额"),
    "transition": ("转场",),
}

FORBIDDEN_SIGNALS: tuple[str, ...] = (
    "字幕", "bgm", "BGM", "logo", "LOGO", "水印", "时间戳", "分镜格线",
    "批注文字", "设计说明", "插画", "线稿", "虚化", "运动模糊", "景深",
    "话筒", "提词器", "道具", "细化",
)

PRESET_SIGNALS: dict[str, tuple[str, ...]] = {
    "P1_phone": ("auto white balance", "phone camera noise", "no color grading"),
    "P2_selfie": ("前置摄像头", "自拍质感", "纪实生活化"),
    "P3_ARRI": ("ARRI",),
    "P4_skin": ("毛孔", "磨皮", "皮肤质感"),
}

# 景别：必须真的点名了景别词，或在分镜里给出了技术头（族 B 的"分镜"段天然带景别）
FRAMING_WORDS: tuple[str, ...] = (
    "全景", "中全景", "中景", "中近景", "近景", "特写", "景别",
)
FRAMING_RE = re.compile("|".join(FRAMING_WORDS))

# 情绪：必须有显式的情绪/表演落点。语气/表演动词算，光有 beat 结构不算。
EMOTION_WORDS: tuple[str, ...] = (
    "情绪", "语气", "表情", "神情", "神色", "眼神",
    "惊讶", "好奇", "自信", "无奈", "温柔", "温和", "激动", "委屈", "从容",
)
EMOTION_RE = re.compile("|".join(EMOTION_WORDS))

# 机位运镜：必须点名机位或运镜，或出现"机位全程不变"这类锁定
CAMERA_WORDS: tuple[str, ...] = (
    "运镜", "机位", "推近", "推拉", "横摇", "跟拍", "过肩", "俯拍", "仰拍",
    "平视", "固定镜头", "手持", "斯坦尼康",
)
CAMERA_RE = re.compile("|".join(CAMERA_WORDS))

PLACEHOLDER_RE = re.compile(r"【([^】]{1,24})】")
QUOTED_RE = re.compile(r"[“\"]([^”\"]{1,200})[”\"]")
SHOT_RE = re.compile(r"(?:镜头|第)\s*(\d{1,2})")


def count_hits(text: str, words: tuple[str, ...]) -> int:
    return sum(text.count(word) for word in words)


def detect_sections(text: str) -> list[str]:
    """按出现顺序识别段落标题。"""
    found: list[str] = []
    for line in text.splitlines():
        clean = line.strip().lstrip("#").strip()
        clean = clean.strip("【】[] ")
        for canonical, aliases in SECTION_ALIASES.items():
            if any(clean == alias or clean.startswith(alias) for alias in aliases):
                found.append(canonical)
                break
    return found


def analyze(name: str, text: str) -> dict[str, Any]:
    lock_hits = [key for key, words in LOCK_SIGNALS.items() if count_hits(text, words) > 0]
    beat_hits = [key for key, words in BEAT_SIGNALS.items() if count_hits(text, words) > 0]
    presets = [key for key, words in PRESET_SIGNALS.items() if count_hits(text, words) > 0]

    forbidden_raw: dict[str, int] = {}
    for word in FORBIDDEN_SIGNALS:
        hits = text.count(word)
        if hits:
            forbidden_raw[word] = hits

    placeholders: dict[str, int] = {}
    for hit in PLACEHOLDER_RE.findall(text):
        key = hit.strip()
        placeholders[key] = placeholders.get(key, 0) + 1

    quoted = [q.strip() for q in QUOTED_RE.findall(text) if q.strip()]
    shots = sorted({int(m) for m in SHOT_RE.findall(text)})

    return {
        "chars": len(text),
        "section_order": detect_sections(text),
        "shot_numbers": shots,
        "shot_count": len(shots),
        "one_take_signals": [
            w for w in ("一镜到底", "机位全程不变", "镜头无跳切", "镜头不推拉", "无剪辑切换")
            if w in text
        ],
        "same_angle_signals": [
            w for w in ("同一固定镜头", "机位同镜头", "同机位") if w in text
        ],
        "lock_hits": lock_hits,
        "beat_hits": beat_hits,
        "framing_hits": FRAMING_RE.findall(text),
        "emotion_hits": EMOTION_RE.findall(text),
        "camera_hits": CAMERA_RE.findall(text),
        "preset_candidates": presets,
        "forbidden_hits": forbidden_raw,
        "forbidden_total": sum(forbidden_raw.values()),
        "placeholders": placeholders,
        "placeholder_total": sum(placeholders.values()),
        "placeholder_kinds": sorted({_kind_of(k) for k in placeholders}),
        "quoted_lines": quoted,
        "quoted_line_count": len(quoted),
    }


# 占位符 → 素材类型。与框架 §2.1 的对应表一致。
KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("multi_view", ("多视图", "多视角", "形象图", "人物图")),
    ("scene", ("场景图", "场景参考图", "场景", "背景")),
    ("blocking", ("位置图", "位置关系图", "站位图", "方位图", "人物和场景位置关系图")),
    ("pose_sheet", ("体式图", "体式拆解图", "动作拆解图", "动作图", "体式动作图", "体式动作参考")),
    ("video", ("参考视频", "动作视频", "体式视频", "视频")),
    ("audio", ("音频", "音色")),
)


def _kind_of(placeholder: str) -> str:
    for kind, words in KIND_RULES:
        if any(word in placeholder for word in words):
            return kind
    return "other"


def main() -> int:
    if not EXTRACT.exists():
        print(f"[X] 找不到 {EXTRACT}", file=sys.stderr)
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []

    for path in sorted(EXTRACT.glob("*.txt")):
        if path.name.startswith("_"):
            continue
        name = path.stem
        meta = DOC_MAP.get(name, {"title": name, "family": "A", "genre": "dialogue"})
        text = path.read_text(encoding="utf-8")
        target = RAW / f"{name}.txt"
        shutil.copyfile(path, target)

        analysis = analyze(name, text)
        samples.append({
            "id": _slug(name),
            "name": name,
            "title": meta["title"],
            "family": meta["family"],
            "genre": meta["genre"],
            "raw_path": f"assets/corpus/raw/{name}.txt",
            "source_doc": f"{name}.docx",
            **analysis,
        })

    manifest = {
        "note": "爆款样本语料库（只读）。样本既是回归基准，也是短语资产的证据来源。",
        "count": len(samples),
        "statistics": _cross_stats(samples),
        "samples": samples,
    }
    out = CORPUS / "manifest.json"
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] 归档 {len(samples)} 份样本 → {RAW}")
    print(f"[OK] 统计清单 → {out}")
    print("\n跨样本关键覆盖率：")
    for key, value in manifest["statistics"]["dimension_coverage"].items():
        print(f"  {key:<24} {value['hit']:>2}/{value['total']}  {value['rate']:.0%}")
    return 0


def _slug(name: str) -> str:
    return (
        name.replace(" ", "-")
        .replace("_", "-")
        .lower()
    )


def _cross_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """跨样本一致性统计 —— 短语库"必写维度"的直接依据。"""
    total = len(samples) or 1

    def rate(pred) -> dict[str, Any]:
        hit = sum(1 for s in samples if pred(s))
        return {"hit": hit, "total": total, "rate": round(hit / total, 4)}

    coverage = {
        "眨眼节拍": rate(lambda s: "眨眼" in s["lock_hits"]),
        "皮肤质感": rate(lambda s: "皮肤质感" in s["lock_hits"]),
        "景别": rate(lambda s: bool(s["framing_hits"])),
        "站位": rate(lambda s: "站位" in s["lock_hits"]),
        "视线": rate(lambda s: "视线" in s["lock_hits"]),
        "遮挡": rate(lambda s: "遮挡" in s["lock_hits"]),
        "场景锚定": rate(lambda s: "场景锚定" in s["lock_hits"]),
        "音色": rate(lambda s: "音色" in s["lock_hits"]),
        "人物一致性": rate(lambda s: "multi_view" in s["placeholder_kinds"]),
        "禁止字幕": rate(lambda s: "字幕" in s["forbidden_hits"]),
        "禁止BGM": rate(lambda s: any(k.lower() == "bgm" for k in s["forbidden_hits"])),
        "禁止杂元素": rate(lambda s: any(
            k in s["forbidden_hits"] for k in ("logo", "LOGO", "水印", "时间戳")
        )),
        "禁止臆造道具": rate(lambda s: any(
            k in s["forbidden_hits"] for k in ("话筒", "提词器", "道具")
        )),
        "机位": rate(lambda s: bool(s["camera_hits"]) or bool(
            s["one_take_signals"] or s["same_angle_signals"]
        )),
        "运镜": rate(lambda s: bool(s["camera_hits"]) or bool(s["one_take_signals"])),
        "情绪": rate(lambda s: bool(s["emotion_hits"])),
    }

    forbidden_totals: dict[str, int] = {}
    for sample in samples:
        for word, hits in sample["forbidden_hits"].items():
            forbidden_totals[word] = forbidden_totals.get(word, 0) + hits

    placeholder_kinds: dict[str, int] = {}
    for sample in samples:
        for kind in sample["placeholder_kinds"]:
            placeholder_kinds[kind] = placeholder_kinds.get(kind, 0) + 1

    return {
        "note": "覆盖率 ≥60% 的维度即「必须显式钉住」的必写维度（框架 §11 G13）。",
        "dimension_coverage": coverage,
        "forbidden_frequency": dict(
            sorted(forbidden_totals.items(), key=lambda kv: -kv[1])
        ),
        "reference_kind_frequency": dict(
            sorted(placeholder_kinds.items(), key=lambda kv: -kv[1])
        ),
        "one_take_samples": [
            s["id"] for s in samples if s["one_take_signals"]
        ],
        "same_angle_samples": [
            s["id"] for s in samples if s["same_angle_signals"]
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
