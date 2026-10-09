"""维度级回归的标注工作台：导出标注清单 + 生成审片页 + 算准确率。

PRD §6.1.5 要求「运镜≥90%、站位≥95%、景别≥85%」——
这三个指标**必须有 ground truth 才能算**。PRD 原本设想用 17 份历史成功
提示词当标准答案，但 `_extract/` 那 10 份答案卷引用的源视频
一条都不在 `测试素材/` 里（`scripts/pair_probe.py` 已证实），
所以标准答案只能人工标注。

本脚本因此做三件事：

    export  为指定视频生成标注清单（CSV 骨架 + 每镜头缩略图 + 审片页 HTML）
    score   把填好的标注与反推结果逐镜头比对，算维度准确率
    status  看标注进度

标注清单刻意**只填时间轴、不预填模型结论** —— 预填会诱导标注者
顺着模型的答案打勾，量出来的准确率就是假的。

用法：
    .venv\\Scripts\\python.exe scripts\\dimension_regression.py export --sample
    .venv\\Scripts\\python.exe scripts\\dimension_regression.py score
    .venv\\Scripts\\python.exe scripts\\dimension_regression.py status
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
ANNOTATION_DIR = ROOT / "assets" / "regression" / "annotations"

# 标注用的闭集词表。必须与 engine 侧取值一致，否则比对会全是"不匹配"
FRAMING_VOCAB = ["脸部特写", "近景", "中近景", "中景", "中全景", "全景", "无人物"]
MOTION_VOCAB = ["fixed", "push_in", "pull_out", "pan_or_track", "handheld", "unknown"]
BLOCKING_VOCAB = ["left", "center", "right"]

# 人工标注很少刚好写成词表里的那个词。这里只做**等价改写**，
# 不做"宽松匹配" —— 把"固定，向右移镜头"强行当 fixed 会丢掉真实信息。
MOTION_ALIASES = {
    "固定镜头": "fixed", "固定机位": "fixed", "固定": "fixed",
    "定机位": "fixed", "机位固定": "fixed", "不动": "fixed",
    "推近": "push_in", "推近镜头": "push_in", "推镜": "push_in",
    "前推": "push_in", "推进": "push_in", "zoom in": "push_in",
    "拉远": "pull_out", "拉远镜头": "pull_out", "后拉": "pull_out",
    "拉镜": "pull_out", "zoom out": "pull_out",
    "摇镜": "pan_or_track", "移镜": "pan_or_track", "横移": "pan_or_track",
    "跟拍": "pan_or_track", "摇移": "pan_or_track", "pan": "pan_or_track",
    "手持": "handheld", "手持晃动": "handheld",
    # "正反打"是**剪辑手法**（两个人来回切），不是运镜术语 —— 单看机位它就是不动的。
    # 本回归量的维度是"运镜"，所以按 fixed 计，但归一说明里必须留下痕迹，
    # 免得日后看报告的人以为这是算法判对了。
    "正反打": "fixed",
}

FRAMING_ALIASES = {
    "特写": "脸部特写", "大特写": "脸部特写", "脸部特写": "脸部特写",
    "近景": "近景", "中近景": "中近景", "中景": "中景",
    "中全景": "中全景", "全景": "全景", "远景": "全景",
    "无人物": "无人物", "空镜": "无人物",
}

BLOCKING_ALIASES = {
    "居中": "center", "中间": "center", "画面中央": "center", "center": "center", "中": "center",
    "左侧": "left", "左": "left", "左半边": "left", "left": "left",
    "右侧": "right", "右": "right", "右半边": "right", "right": "right",
}

# 一条标注里同时出现多个运镜词时（如"固定，向右移镜头"），取**更具体**的那个：
# 用户写了两个词，说明这是一个复合描述，具体动作才是他要表达的重点。
MOTION_PRIORITY = ["push_in", "pull_out", "pan_or_track", "handheld", "fixed"]

# PRD §6.1.5 的目标线
TARGETS = {"景别": 0.85, "运镜": 0.90, "站位": 0.95}


def main() -> int:
    parser = argparse.ArgumentParser(description="维度级准确率回归")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="导出标注清单与审片页")
    p_export.add_argument("--dir", default="测试素材")
    p_export.add_argument("--videos", nargs="*", default=[], help="只导出这些文件名")
    p_export.add_argument("--sample", action="store_true", help="只导出推荐抽样集")
    p_export.add_argument("--limit", type=int, default=0)
    p_export.add_argument("--force", action="store_true", help="覆盖已有标注")

    p_score = sub.add_parser("score", help="算维度准确率")
    p_score.add_argument("--annotations", default=str(ANNOTATION_DIR))
    p_score.add_argument("--out", default="assets/regression/dimension_report.json")
    p_score.add_argument(
        "--framing-tolerance", choices=["strict", "adjacent"], default="adjacent",
        help="景别比对口径。adjacent = 相邻档位算命中（PRD：语义等价即命中）",
    )

    sub.add_parser("status", help="看标注进度")

    p_show = sub.add_parser("show", help="看归一后的标准答案（打分前先核对）")
    p_show.add_argument("--annotations", default=str(ANNOTATION_DIR))

    args = parser.parse_args()
    if args.command == "export":
        return cmd_export(args)
    if args.command == "score":
        return cmd_score(args)
    if args.command == "show":
        return cmd_show(args)
    return cmd_status(args)


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------
def cmd_export(args) -> int:
    from engine.app.config import Settings
    from engine.app.pipeline.infer import build_inferencer
    from engine.app.providers.ffmpeg_media import FFmpegMedia
    from engine.app.providers.scene_detect import SceneDetectShots

    settings = Settings.load()
    media = FFmpegMedia(settings)
    detector = SceneDetectShots(settings)
    shot_source = detector if detector.available else media

    source = Path(args.dir)
    videos = sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if args.videos:
        wanted = set(args.videos)
        videos = [v for v in videos if v.name in wanted]
    if args.sample:
        videos = _pick_sample(videos)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print("[X] 没有匹配的视频", file=sys.stderr)
        return 1

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    sheet = ANNOTATION_DIR / "labels.csv"
    thumbs_root = ANNOTATION_DIR / "thumbs"

    rows: list[dict[str, str]] = []
    pages: list[str] = []
    for video in videos:
        info = media.probe(str(video))
        shots = shot_source.detect(str(video), info)
        thumb_dir = thumbs_root / video.stem
        frames = media.extract_keyframes(str(video), shots, str(thumb_dir))
        pages.append(_render_page(video, info, shots, frames, len(rows)))
        for index, shot in enumerate(shots):
            rows.append({
                "video": video.name,
                "shot": str(shot.index),
                "start_s": f"{shot.start_ms / 1000:.2f}",
                "duration_s": f"{shot.duration_sec:.2f}",
                "thumb": frames[index] if index < len(frames) else "",
                "景别": "",
                "运镜": "",
                "站位": "",
                "备注": "",
            })
        print(f"  {video.name:<34} {len(shots)} 镜  {len(frames)} 张缩略图")

    if sheet.exists() and not args.force:
        existing = _load_sheet(sheet)
        filled = {k: v for k, v in existing.items() if _is_filled(v)}
        if filled:
            print(f"\n[!] {sheet} 已有 {len(filled)} 条填写内容，未覆盖。")
            print("    要重做请加 --force（会丢失已填内容），或先备份。")
            _write_review_page(pages)
            return 0

    _write_sheet(sheet, rows)
    review = _write_review_page(pages)

    print(f"\n标注清单 → {sheet}")
    print(f"审片页   → {review}")
    print(f"填完后跑 → .venv\\Scripts\\python.exe scripts\\dimension_regression.py score")
    print("\n词表（必须用这些值，否则比对会全判不匹配）：")
    print(f"  景别  {'/'.join(FRAMING_VOCAB)}")
    print(f"  运镜  {'/'.join(MOTION_VOCAB)}")
    print(f"  站位  {'/'.join(BLOCKING_VOCAB)}（画面里主要人物在左/中/右；只有单人时填其所在半边）")
    return 0


def _pick_sample(videos: list[Path]) -> list[Path]:
    """抽样原则：覆盖不同镜头模式与题材，且总量控制在一次能标完的规模。

    **不是随便挑**：单镜头（一镜到底）与多镜头各自的运镜判据不同，
    只标一种会让准确率虚高；题材也要含教学/对话/演播厅三类。
    """
    preferred = [
        "菜市场采访-苏.MP4",       # 一镜到底 · 街采 · 人物大
        "访谈4.mp4",               # 一镜到底 · 对话 · 双人
        "IP演播厅带练-丹.mp4",      # 多镜头 · 演播厅
        "爆款视频AI复刻刘畊宏.mp4",   # 多镜头 · 教学 · 有运镜
        "兴趣岛增长中台_5 (2).mp4",  # 含明显摇镜（位移 3.94）
        "中老年游艇口播-丹.mp4",     # 灰区样本：位移中等、有散度
    ]
    by_name = {v.name: v for v in videos}
    picked = [by_name[n] for n in preferred if n in by_name]
    return picked or videos[:6]


def _render_page(video: Path, info, shots, frames, row_offset: int) -> str:
    cards = []
    for index, shot in enumerate(shots):
        thumb = frames[index] if index < len(frames) else ""
        rel = ""
        if thumb:
            try:
                rel = Path(thumb).relative_to(ANNOTATION_DIR).as_posix()
            except ValueError:
                rel = Path(thumb).as_posix()
        image = (
            f'<img src="{html.escape(rel)}" alt="镜头{shot.index}">'
            if rel else '<div class="missing">未抽到帧</div>'
        )
        cards.append(f"""
        <div class="shot">
          {image}
          <div class="meta">
            <b>镜头 {shot.index}</b>
            <span>{shot.start_ms / 1000:.2f}s → {shot.end_ms / 1000:.2f}s</span>
            <span>时长 {shot.duration_sec:.2f}s</span>
            <span class="row">CSV 第 {row_offset + index + 2} 行</span>
          </div>
        </div>""")
    return f"""
    <section>
      <h2>{html.escape(video.name)}</h2>
      <p class="tech">{info.width}×{info.height} · {info.fps:.2f}fps ·
         {info.duration_sec:.2f}s · {info.aspect_ratio} · {len(shots)} 个镜头</p>
      <div class="shots">{''.join(cards)}</div>
    </section>"""


def _write_review_page(pages: list[str]) -> Path:
    target = ANNOTATION_DIR / "review.html"
    body = "\n".join(pages)
    target.write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>维度回归 · 审片页</title>
<style>
 body {{ font: 14px/1.6 system-ui, "Microsoft YaHei", sans-serif; margin: 24px;
        background:#f6f7f9; color:#1a1a1a; }}
 h1 {{ font-size: 20px; }}
 .hint {{ background:#fff8e1; border-left:4px solid #f0ad4e; padding:10px 14px;
          margin:16px 0; }}
 section {{ background:#fff; border:1px solid #e3e6ea; border-radius:8px;
            padding:14px 18px; margin:16px 0; }}
 .tech {{ color:#666; margin:4px 0 12px; }}
 .shots {{ display:flex; flex-wrap:wrap; gap:12px; }}
 .shot {{ width:190px; border:1px solid #e3e6ea; border-radius:6px; overflow:hidden;
          background:#fafbfc; }}
 .shot img {{ width:100%; display:block; background:#000; }}
 .missing {{ height:120px; display:flex; align-items:center; justify-content:center;
             color:#999; background:#eee; }}
 .meta {{ padding:6px 8px; font-size:12px; display:flex; flex-direction:column;
          gap:2px; color:#555; }}
 .meta b {{ color:#111; font-size:13px; }}
 .row {{ color:#0a7; }}
 code {{ background:#eef1f4; padding:1px 4px; border-radius:3px; }}
</style></head><body>
<h1>维度回归 · 审片页</h1>
<div class="hint">
  <b>怎么用</b>：打开 <code>assets/regression/annotations/labels.csv</code>，
  对照本页每个镜头的缩略图（需要看清运动时直接打开原视频），
  填 <code>景别 / 运镜 / 站位</code> 三列。<br>
  <b>为什么不让模型先填</b>：预填会诱导你顺着模型的答案打勾，
  那样量出来的"准确率"是假的。这里刻意只给时间轴与画面。<br>
  <b>拿不准就留空</b>：空值不计入准确率分母，比填一个猜的值有用。
</div>
{body}
</body></html>
""", encoding="utf-8")
    return target


# ----------------------------------------------------------------------
# show：打分前先看归一结果，避免"算法错了"和"口径不同"混在一起
# ----------------------------------------------------------------------
def cmd_show(args) -> int:
    sheet = Path(args.annotations) / "labels.csv"
    if not sheet.exists():
        print(f"[X] 没有标注清单：{sheet}", file=sys.stderr)
        return 1
    rows = _load_rows(sheet)
    _, encoding = _read_text(sheet)
    print(f"标注清单：{sheet}")
    print(f"编码：{encoding}　行数：{len(rows)}"
          f"　已填：{sum(1 for r in rows if _is_filled(r))}\n")
    print(f"{'视频':<24}{'镜':>3}  {'景别':<8}{'运镜':<14}{'站位':<8}归一说明")
    for row in rows:
        if not _is_filled(row):
            continue
        truth = _ground_truth(row)
        print(f"{row['video'][:22]:<24}{row['shot']:>3}  "
              f"{truth['景别'] or '—':<8}{truth['运镜'] or '—':<14}"
              f"{truth['站位'] or '—':<8}{truth['_归一说明']}")
    print("\n注意：归一说明非空的那些格子，是**脚本按等价改写映射的**。")
    print("      如果映射错了，请直接改 labels.csv 写成词表里的原值。")
    return 0


# ----------------------------------------------------------------------
# score
# ----------------------------------------------------------------------
def cmd_score(args) -> int:
    from engine.app.pipeline.infer import build_inferencer

    sheet = Path(args.annotations) / "labels.csv"
    if not sheet.exists():
        print(f"[X] 没有标注清单：{sheet}", file=sys.stderr)
        print("    先跑 export", file=sys.stderr)
        return 1

    rows = _load_rows(sheet)
    videos = sorted({r["video"] for r in rows if _is_filled(r)})
    if not videos:
        print("[X] 标注清单还是空的 —— 需要人工填 景别/运镜/站位 三列")
        return 1

    _, encoding = _read_text(sheet)
    if encoding != "utf-8-sig":
        print(f"[!] 标注清单编码是 {encoding}（表格软件常见行为）。"
              f"本仓库约定 UTF-8；读取已兼容，但请勿用脚本事后改这个文件。")

    settings_inferencer = build_inferencer()
    source = Path("测试素材")
    report: dict[str, object] = {"videos": {}, "summary": {}}
    totals = {"景别": [0, 0], "运镜": [0, 0], "站位": [0, 0]}
    mismatches: list[dict[str, str]] = []
    normalizations: list[dict[str, str]] = []
    adjacent_hits: list[dict[str, str]] = []

    for name in videos:
        video = source / name
        if not video.exists():
            print(f"  [跳过] 找不到视频 {name}")
            continue
        print(f"  反推 {name} …")
        try:
            _, result = settings_inferencer.infer_to_ir(
                str(video), workspace=settings_inferencer.settings.workspace_dir / "dimension"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] {name}: {type(exc).__name__}: {exc}")
            continue

        predicted = {s.boundary.index: s for s in result.shots}
        per_video = {"景别": [0, 0], "运镜": [0, 0], "站位": [0, 0]}
        for row in rows:
            if row["video"] != name or not _is_filled(row):
                continue
            truth = _ground_truth(row)
            if truth["_归一说明"]:
                normalizations.append({
                    "video": name, "shot": row["shot"], "说明": truth["_归一说明"],
                })
            index = int(row["shot"])
            signals = predicted.get(index)
            if signals is None:
                mismatches.append({"video": name, "shot": row["shot"], "维度": "镜头切分",
                                   "标准答案": "-", "反推": "该镜头不存在",
                                   "说明": "切分结果与标注不一致，需人工核对切分点"})
                continue
            for dimension, actual, guess, measured in (
                ("景别", truth["景别"],
                 signals.framing.kind if signals.framing else "",
                 _framing_evidence(signals)),
                ("运镜", truth["运镜"],
                 signals.motion.kind if signals.motion else "",
                 _motion_evidence(signals)),
                ("站位", truth["站位"], _blocking_side(signals), _blocking_evidence(signals)),
            ):
                if not actual:
                    continue
                totals[dimension][1] += 1
                per_video[dimension][1] += 1
                hit = (
                    _framing_hit(actual, guess, args.framing_tolerance)
                    if dimension == "景别" else actual == guess
                )
                adjacent = (
                    dimension == "景别" and guess != actual
                    and _framing_hit(actual, guess, "adjacent")
                )
                if hit:
                    totals[dimension][0] += 1
                    per_video[dimension][0] += 1
                else:
                    mismatches.append({
                        "video": name, "shot": row["shot"], "维度": dimension,
                        "标准答案": actual, "反推": guess or "留空",
                        "量测": measured, "说明": row.get("备注", ""),
                    })
                if adjacent:
                    adjacent_hits.append({
                        "video": name, "shot": row["shot"],
                        "标准答案": actual, "反推": guess,
                    })
        report["videos"][name] = {
            k: {"hit": v[0], "total": v[1],
                "accuracy": round(v[0] / v[1], 3) if v[1] else None}
            for k, v in per_video.items()
        }

    report["summary"] = {
        k: {"hit": v[0], "total": v[1],
            "accuracy": round(v[0] / v[1], 3) if v[1] else None,
            "target": TARGETS[k],
            "meets_target": (v[0] / v[1] >= TARGETS[k]) if v[1] else None}
        for k, v in totals.items()
    }
    report["mismatches"] = mismatches
    report["normalizations"] = normalizations
    report["adjacent_hits"] = adjacent_hits
    report["framing_tolerance"] = args.framing_tolerance

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 62)
    print("维度级准确率 vs PRD §6.1.5 目标")
    print("=" * 62)
    print(f"  样本：{len(videos)} 条视频 / {sum(1 for r in rows if _is_filled(r))} 个镜头"
          f"　景别口径：{args.framing_tolerance}")
    for dimension, item in report["summary"].items():
        if item["total"] == 0:
            print(f"  {dimension}  无标注样本")
            continue
        flag = "达标" if item["meets_target"] else "未达标"
        print(f"  {dimension}  {item['hit']}/{item['total']} = {item['accuracy']:.0%}"
              f"　目标 {item['target']:.0%}　→ {flag}")
    if adjacent_hits:
        print(f"\n  其中 {len(adjacent_hits)} 个景别样本是**相邻档位**命中"
              f"（如标注中景/反推中近景）：")
        for item in adjacent_hits:
            print(f"    {item['video'][:20]:<20} 镜{item['shot']:<3} "
                  f"{item['标准答案']} ↔ {item['反推']}")
    if mismatches:
        print(f"\n偏差明细 {len(mismatches)} 条（带量测证据，便于判断是算法错还是标注口径不同）：")
        for item in mismatches[:30]:
            print(f"  {item['video'][:20]:<20} 镜{item['shot']:<3} {item['维度']} "
                  f"标注={item['标准答案']:<10} 反推={item['反推']:<14} {item.get('量测', '')}")
    print(f"\n报告 → {out}")
    return 0


# ----------------------------------------------------------------------
# 量测证据：偏差明细里带上原始量，才能分清"算法错"与"口径不同"
# ----------------------------------------------------------------------
def _framing_evidence(signals) -> str:
    if not signals.framing:
        return "无景别结论"
    return f"人体框高占比={signals.framing.person_ratio:.2f}"


def _motion_evidence(signals) -> str:
    if not signals.motion:
        return "无运镜结论"
    motion = signals.motion
    return (f"位移={motion.magnitude:.2f} 散度={motion.divergence:+.2f}"
            f" conf={motion.confidence:.2f}")


def _blocking_evidence(signals) -> str:
    if not signals.blocking:
        return "无站位结论"
    return " ".join(f"{b.side}@{b.x_ratio:.2f}" for b in signals.blocking)


def _blocking_side(signals) -> str:
    """镜头的「站位」判定 = 画面构图偏侧。

    **不是**逐个人物的左右（``blocking``）。标注口径实测为"整个画面居中不居中"：
    双人一左一右的镜头，标注一致填 center。逐人的左右仍然保留在
    ``ShotSignals.blocking`` 里作为证据，但那是另一个问题。
    """
    return getattr(signals, "composition", "") or ""


# 景别相邻档位表。PRD §6.1.5 的口径是「语义等价即命中」，
# "中近景 vs 近景"是程度之争不是事实之争，不该算算法错。
FRAMING_ORDER = ["全景", "中全景", "中景", "中近景", "近景", "脸部特写"]


def _framing_hit(actual: str, guess: str, tolerance: str) -> bool:
    if actual == guess:
        return True
    if tolerance != "adjacent":
        return False
    if actual not in FRAMING_ORDER or guess not in FRAMING_ORDER:
        return False
    return abs(FRAMING_ORDER.index(actual) - FRAMING_ORDER.index(guess)) == 1


# ----------------------------------------------------------------------
# status / CSV 读写
# ----------------------------------------------------------------------
def cmd_status(args) -> int:
    sheet = ANNOTATION_DIR / "labels.csv"
    if not sheet.exists():
        print("还没有标注清单 —— 先跑 export")
        return 1
    rows = _load_rows(sheet)
    total = len(rows) * 3
    filled = sum(1 for r in rows for k in ("景别", "运镜", "站位") if r.get(k))
    per_video: dict[str, list[int]] = {}
    for row in rows:
        slot = per_video.setdefault(row["video"], [0, 0])
        slot[1] += 3
        slot[0] += sum(1 for k in ("景别", "运镜", "站位") if row.get(k))
    print(f"标注进度 {filled}/{total} 格")
    for name, (done, cap) in per_video.items():
        bar = "█" * int(12 * done / cap) + "·" * (12 - int(12 * done / cap))
        print(f"  {bar} {done:>3}/{cap:<3} {name}")
    return 0


def _read_text(path: Path) -> tuple[str, str]:
    """读文本，容忍被 Excel/WPS 存成 GBK 的情况。

    **这个仓库的铁律是文本文件只用 UTF-8**（HANDOFF 第一节），但人工标注
    大概率是双击 CSV 用表格软件改的，那东西在中文 Windows 上默认存 GBK。
    与其报错让人重做，不如读进来并**明确告知编码不是 UTF-8**。
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("gbk", errors="replace"), "gbk"


def _load_rows(sheet: Path) -> list[dict[str, str]]:
    text, _ = _read_text(sheet)
    return [dict(row) for row in csv.DictReader(text.splitlines())]


def _load_sheet(sheet: Path) -> dict[str, dict[str, str]]:
    return {
        f"{row['video']}#{row['shot']}": row for row in _load_rows(sheet)
    }


def _normalize(value: str, aliases: dict[str, str], vocab: list[str],
               priority: list[str] | None = None) -> tuple[str, str]:
    """把人工写的自然语言标注映射到词表值。返回 ``(规范值, 说明)``。

    **不做宽松匹配**。匹配不上就原样返回并给出说明 —— 让报告里出现一个
    看得懂的"无法归一"比悄悄判成不匹配要好，人一眼能看出是标注还是算法的问题。
    """
    text = (value or "").strip()
    if not text:
        return "", ""
    if text in vocab:
        return text, ""
    if text in aliases:
        return aliases[text], f"归一：{text} → {aliases[text]}"

    hits: list[str] = []
    for key in sorted(aliases, key=len, reverse=True):
        if key in text and aliases[key] not in hits:
            hits.append(aliases[key])
    if not hits:
        return text, f"无法归一：{text!r} 不在词表里，按原文比对"

    if len(hits) == 1:
        return hits[0], f"归一：{text} → {hits[0]}"
    if priority:
        for candidate in priority:
            if candidate in hits:
                return candidate, f"归一：{text} → {candidate}（复合描述，取更具体的那个）"
    return hits[0], f"归一：{text} → {hits[0]}（多义，取首个）"


def _ground_truth(row: dict[str, str]) -> dict[str, str]:
    """标注行 → 规范化的三维标准答案，附归一说明。"""
    framing, note_f = _normalize(row.get("景别", ""), FRAMING_ALIASES, FRAMING_VOCAB)
    motion, note_m = _normalize(
        row.get("运镜", ""), MOTION_ALIASES, MOTION_VOCAB, MOTION_PRIORITY
    )
    blocking, note_b = _normalize(row.get("站位", ""), BLOCKING_ALIASES, BLOCKING_VOCAB)
    return {
        "景别": framing, "运镜": motion, "站位": blocking,
        "_归一说明": "；".join(n for n in (note_f, note_m, note_b) if n),
    }


def _is_filled(row: dict[str, str]) -> bool:
    return any((row.get(k) or "").strip() for k in ("景别", "运镜", "站位"))


def _write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["video", "shot", "start_s", "duration_s", "thumb", "景别", "运镜", "站位", "备注"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
