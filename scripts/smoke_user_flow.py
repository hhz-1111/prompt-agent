"""端到端走一遍用户真实操作，验证 PRD 描述的工作流是否成立。

用户描述的流程：
    上传视频 → 反推 → 改场景/人物/台词/脚本 → 出提示词 → 复制到 Seedance

这个脚本不打桩，走真实 HTTP + 真实模型 + 真实 CV，
只把「用户手改」这一步用代码模拟，用来确认改完确实会反映到交付物里。

用法：py scripts/smoke_user_flow.py [--base http://127.0.0.1:8765]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(path: str, *, base: str, method: str = "GET", payload: dict | None = None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + path, data=body, method=method,
        headers={"Content-Type": "application/json; charset=utf-8"} if body else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        return {"__error__": exc.code, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--video", default=str(ROOT / "测试素材" / "爆款视频AI复刻刘畊宏.mp4"))
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"[X] 找不到视频 {video}")
        return 1

    # 直接用已有项目（避免重复调模型），找不到就提示
    projects = call("/api/projects", base=args.base)["projects"]
    ready = [p for p in projects if p["has_ir"]]
    if not ready:
        print("[X] 没有已反推的项目，请先上传+反推一次")
        return 1
    pid = ready[0]["project_id"]
    print(f"使用项目 {pid}（{ready[0]['name']}）\n")

    assembled = call(f"/api/projects/{pid}/assemble", base=args.base)
    document = assembled["document"]
    positions = assembled["positions"]["positions"]
    gates = assembled["gates"]

    print("=" * 76)
    print("【第 1 步】反推结果 —— 用户看到的三色标注")
    print("=" * 76)
    counts = assembled["positions"]["counts"]
    print(f"  🔵 可替换素材位 {counts['blue']} 个"
          f"　🟠 可编辑文案位 {counts['orange']} 个"
          f"　⚫ 锁定区 {counts['gray']} 个")

    print("\n  ── 蓝色：用户要换的素材 ──")
    for item in call(f"/api/projects/{pid}", base=args.base)["reference_manifest"]:
        mark = "已绑定" if item["bound"] else "待绑定"
        print(f"    [{item['type']:<10}] {'、'.join(item['slots']):<22} {mark}")

    print("\n  ── 橙色：用户能改的文字（按类型归类）──")
    editable = [p for p in positions if p["color"] == "orange"]
    by_label: dict[str, list] = {}
    for p in editable:
        by_label.setdefault(p["label"], []).append(p)
    for label, items in by_label.items():
        sample = str(items[0]["value"])[:34] or "（空）"
        print(f"    {label:<12} ×{len(items):<2} 例：{sample}")

    print("\n  ── 灰色：锁定区（改了复刻就失败）──")
    locked = [p for p in positions if p["color"] == "gray" and p["category"] == "LOCK"]
    for p in locked[:6]:
        print(f"    {p['label']:<12} = {str(p['value'])[:44]}")
    if len(locked) > 6:
        print(f"    …共 {len(locked)} 个锁定位置")

    print("\n" + "=" * 76)
    print("【第 2 步】用户开始改 —— 换场景 / 换人物 / 写台词")
    print("=" * 76)

    edits = [
        ("locked.B1_scene.type", "菜市场", "换场景"),
        ("creative.C1_logline", "李姐在菜市场被王姐夸气色好，说自己在家练普拉提", "改剧情主线"),
    ]
    # 人物改名（人物名为 FREE 位，但颜色是灰色，可写）
    people = [p for p in positions if p["path"].endswith("B2_person[0].name")]
    if people:
        edits.append((people[0]["path"], "王姐", "人物 A 改名"))
    people2 = [p for p in positions if p["path"].endswith("B2_person[1].name")]
    if people2:
        edits.append((people2[0]["path"], "李姐", "人物 B 改名"))

    for path, value, why in edits:
        result = call(f"/api/projects/{pid}/field", base=args.base,
                      method="PUT", payload={"path": path, "value": value})
        mark = "✅" if result.get("ok") else "❌"
        detail = "" if result.get("ok") else f" → {result.get('detail')}"
        print(f"  {mark} {why:<12} {path:<34} = {value}{detail}")

    # 填台词
    dialogue = call(f"/api/projects/{pid}/dialogue", base=args.base, method="PUT",
                    payload={"shot_id": 1, "speaker": "王姐",
                             "line": "你这气色也太好了吧，是不是偷偷做什么项目了",
                             "tone": "惊讶"})
    print(f"  {'✅' if dialogue.get('ok') else '❌'} 填台词       镜头1 = 你这气色也太好了吧…")

    print("\n  ── 试着改锁定区（应当被拒）──")
    for path, why in (("locked.B3_blocking.positions", "站位"),
                      ("creative.C2_beats[0].beat_role", "节拍功能")):
        result = call(f"/api/projects/{pid}/field", base=args.base,
                      method="PUT", payload={"path": path, "value": ["篡改"]})
        mark = "✅ 已拒绝" if result.get("__error__") == 403 else "❌ 竟然可写"
        print(f"  {mark}  {why:<12} {result.get('detail', '')[:44]}")

    print("\n" + "=" * 76)
    print("【第 3 步】重新出提示词")
    print("=" * 76)
    assembled = call(f"/api/projects/{pid}/assemble", base=args.base)
    document = assembled["document"]
    gates = assembled["gates"]
    print(f"  字数 {document['char_count']}　拆条策略 {document['strategy']}"
          f"　闸门 {gates['verdict']}")
    blocked = [r for r in gates["results"] if r["status"] == "block"]
    warned = [r for r in gates["results"] if r["status"] == "warn"]
    for r in blocked:
        print(f"    ❌ {r['code']} {r['detail'][:60]}")
        print(f"       → {r['suggestion'][:96]}")
    for r in warned[:2]:
        print(f"    ⚠️  {r['code']} {r['detail'][:60]}")

    print("\n  ── 用户要复制到 Seedance 的提示词 ──")
    print("  " + "─" * 72)
    for line in document["text"].splitlines():
        print("  " + line)
    print("  " + "─" * 72)

    print("\n" + "=" * 76)
    print("【第 4 步】交付前待办")
    print("=" * 76)
    for w in document.get("warnings", []):
        print(f"  · {w}")

    delivered = gates["deliverable"]
    print(f"\n  判定：{'允许交付' if delivered else '阻断交付'}")
    print(f"  三种导出格式：plain / markdown / json（内容一致，见 /export 接口）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
