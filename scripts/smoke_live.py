"""活体冒烟：对着真实运行的服务跑一遍完整业务流。

与单测的区别：这里打的是真实 HTTP 端口、真实视频、真实模型调用，
用来确认"启动起来真的能用"。单测用替身保证逻辑，这个脚本保证联通。

    py scripts/smoke_live.py [--base http://127.0.0.1:8000] [--video path.mp4]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(base: str, path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json; charset=utf-8"} if body else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit(f"[X] {method} {path} -> HTTP {exc.code}: {detail}") from exc


def upload(base: str, project_id: str, video: Path) -> dict:
    boundary = f"----smoke{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(video.name)[0] or "video/mp4"
    chunks = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{video.name}"\r\nContent-Type: {mime}\r\n\r\n'.encode("utf-8"),
        video.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    request = urllib.request.Request(
        f"{base}/api/projects/{project_id}/video",
        data=b"".join(chunks),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--video", default=str(ROOT / ".probe" / "vid" / "talk.mp4"))
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"[X] 找不到测试视频 {video}")
        return 1

    print(f"活体冒烟 @ {args.base}")
    health = call(args.base, "/api/health")
    print(f"  框架 {health['framework_version']} · 短语库 {health['phrase_lib_version']}")
    if not health["capabilities"]["llm"]["configured"]:
        print("[X] 服务端未配置模型 Key")
        return 1
    print(f"  模型 {health['capabilities']['llm']['vision_model']}")
    print(f"  CV 缺口 {len(health['cv']['missing'])} 项（对应维度置信度已降档）")

    project = call(args.base, "/api/projects", method="POST", payload={"name": "活体冒烟"})
    pid = project["project_id"]
    print(f"\n1) 建项目 {pid}")

    uploaded = upload(args.base, pid, video)
    print(f"2) 上传完成 {uploaded['bytes'] / 1024:.0f} KB")

    call(args.base, f"/api/projects/{pid}/infer", method="POST")
    print("3) 已提交反推，轮询中…")
    started = time.time()
    while True:
        status = call(args.base, f"/api/projects/{pid}/status")
        print(f"   [{status['progress']:5.0%}] {status['stage']}")
        if status["status"] in ("ready", "failed"):
            break
        if time.time() - started > 600:
            print("[X] 反推超时")
            return 1
        time.sleep(2)
    if status["status"] != "ready":
        print(f"[X] 反推失败：{status['error']}")
        return 1
    print(f"   反推完成，用时 {time.time() - started:.1f}s")

    detail = call(args.base, f"/api/projects/{pid}")
    counts = detail["positions"]["counts"]
    print(f"\n4) 三色位置：🔵{counts['blue']} 🟠{counts['orange']} ⚫{counts['gray']}")

    pending = [
        p for p in detail["positions"]["positions"]
        if p["provenance"] and p["provenance"]["confidence"] < 0.7
    ]
    print(f"   待用户确认 {len(pending)} 处")

    print("\n5) 素材清单：")
    for slot in detail["reference_manifest"]:
        mark = "已绑定" if slot["bound"] else "待绑定"
        print(f"   [{slot['type']:12}] {slot['slots']} · {mark}")

    # 门禁：锁定位必须被 API 拦住
    print("\n6) 门禁校验（锁定位应被拒绝）：")
    blocked, allowed = 0, 0
    for payload in (
        {"path": "locked.B3_blocking.positions", "value": ["篡改"]},
        {"path": "creative.C2_beats[1].beat_role", "value": "demo"},
        {"path": "creative.C1_logline", "value": "用户改写的剧情主线"},
    ):
        request = urllib.request.Request(
            f"{args.base}/api/projects/{pid}/field",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="PUT",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                json.loads(response.read().decode("utf-8"))
            allowed += 1
            print(f"   ✗ {payload['path']} 竟然可写")
        except urllib.error.HTTPError as exc:
            blocked += 1
            reason = json.loads(exc.read().decode("utf-8")).get("detail", "")
            print(f"   ✓ 拒绝 {payload['path']}：{reason[:44]}")
    if blocked < 2:
        print("[X] 锁定位门禁失效")
        return 1

    assembled = call(args.base, f"/api/projects/{pid}/assemble")
    gate = assembled["gates"]
    print(f"\n7) 闸门：{gate['verdict']}（阻断 {gate['blocker_count']} / 告警 {gate['warning_count']}）")
    for item in gate["results"]:
        if item["status"] != "ok":
            print(f"   {item['code']} {item['status']:5} {item['detail'][:60]}")

    document = assembled["document"]
    print(f"\n8) 交付物 {document['char_count']} 字 · 策略 {document['strategy']}")
    print("   " + "─" * 56)
    for line in document["text"].splitlines()[:6]:
        print("   " + line[:70])
    print("   …")

    print(f"\n界面：{args.base}/?project={pid}")
    print(f"详情接口：{args.base}/api/projects/{pid}")
    print("\n[OK] 活体冒烟通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
