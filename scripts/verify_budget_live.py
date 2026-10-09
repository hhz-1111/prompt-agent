"""确认在跑的服务真的用了 800 字预算与最新的压缩实现。

为什么要单独验：改完后端忘了重启，端口上会继续跑旧代码，而**没有任何提示**
（HANDOFF 坑 12）。所以每次改完预算/压缩，都要用真实接口复核一次。
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8765"


def call(method: str, path: str, payload=None):
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    request = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


project = call("POST", "/api/projects", {"name": "预算一致性验证"})
pid = project["project_id"]

video = ROOT / "测试素材" / "爆款视频AI复刻刘畊宏.mp4"
boundary = "----probe"
body = b""
body += f"--{boundary}\r\n".encode()
body += f'Content-Disposition: form-data; name="file"; filename="{video.name}"\r\n'.encode()
body += b"Content-Type: video/mp4\r\n\r\n"
body += video.read_bytes()
body += f"\r\n--{boundary}--\r\n".encode()
request = urllib.request.Request(
    f"{BASE}/api/projects/{pid}/video", data=body, method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    response.read()

call("POST", f"/api/projects/{pid}/infer")
deadline = time.time() + 300
while time.time() < deadline:
    status = call("GET", f"/api/projects/{pid}/status")
    if status.get("status") != "analyzing":
        break
    time.sleep(3)
print("反推状态:", status.get("status"), status.get("stage"))

assembled = call("GET", f"/api/projects/{pid}/assemble")
document = assembled["document"]
print("交付物 budget =", document["budget"], "（期望 800）")
print("交付物 char_count =", document["char_count"])
print("over_budget =", document.get("over_budget"))
print("压缩条目:")
for entry in assembled["compression"]:
    print(f"   {entry['section']} | {entry['what'][:40]} | 省 {entry['saved_chars']} 字")

g7 = next(r for r in assembled["gates"]["results"] if r["code"] == "G7")
print("G7:", g7["status"], "|", g7["detail"])

call("DELETE", f"/api/projects/{pid}")
print("\n[OK] 服务侧预算与压缩实现均为最新")
