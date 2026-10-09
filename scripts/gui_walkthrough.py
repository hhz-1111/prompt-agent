"""前端走查：用 CDP 驱动 headless Edge/Chrome，把界面**真点一遍**。

为什么需要它：前端没有单测，改完只能靠"我看了一眼"。这份脚本把
「左栏在不在 / 点新建能不能回上传页 / 刷新会不会被拽回旧项目 /
点项目能不能切回去 / 删项目会不会弹 404」变成可重复的断言，并且留截图。

    .venv\\Scripts\\python.exe scripts\\gui_walkthrough.py
    .venv\\Scripts\\python.exe scripts\\gui_walkthrough.py --delete   # 多走一步"删除项目"

前置：8765 上服务已起、`web/dist` 已构建。**rebuild 后不用重启服务** ——
`/assets` 是 StaticFiles 挂载、`/` 每次读盘，浏览器刷新即可。

安全约定：`--delete` **自己造一个临时项目来删**（走查前建、走查后兜底再删一次），
绝不碰用户已有的项目。

依赖 `websockets` —— 它随 `uvicorn[standard]` 一起装进来，不是独立依赖。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)

_LIST_JS = (
    "fetch('/api/projects').then(r=>r.json())"
    ".then(d=>d.projects.map(p=>({id:p.project_id,name:p.name,has_ir:p.has_ir})))"
)


class CDP:
    """够用就行的 CDP 客户端：发命令、等同名 id 的回复、顺手截图。"""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.n = 0

    async def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expression: str):
        res = await self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        return res.get("result", {}).get("value")

    async def shot(self, path: Path) -> Path:
        res = await self.call("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(res["data"]))
        return path


async def _ws_url(port: int, deadline: float = 30.0) -> str:
    end = time.time() + deadline
    while time.time() < end:
        try:
            raw = urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2).read()
            pages = [t for t in json.loads(raw) if t.get("type") == "page"]
            if pages:
                return pages[0]["webSocketDebuggerUrl"]
        except Exception:
            pass
        await asyncio.sleep(0.4)
    raise RuntimeError("DevTools 端口没起来")


async def run(args) -> int:
    import websockets  # noqa: PLC0415 —— 软依赖，缺了才报

    browser = args.browser or next((p for p in BROWSERS if Path(p).exists()), "")
    if not browser:
        print("[X] 找不到 Edge / Chrome，用 --browser 指一个")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.gettempdir()) / "mod-gui-walkthrough"
    proc = subprocess.Popen(
        [
            browser, "--headless=new", "--disable-gpu", "--no-first-run",
            f"--user-data-dir={profile}", f"--remote-debugging-port={args.port}",
            "--window-size=1440,900", "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    checks: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, detail: object = "") -> None:
        # detail 留空时就把真假值本身报出来，免得最后打一堆空括号
        checks.append((label, bool(ok), str(detail) if detail != "" else str(bool(ok))))

    try:
        async with websockets.connect(await _ws_url(args.port), max_size=40 * 1024 * 1024) as ws:
            cdp = CDP(ws)
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Page.navigate", url=args.base)
            await asyncio.sleep(4)

            # ① 首屏：左栏 + 结果页
            projects = await cdp.js(_LIST_JS) or []
            check("左侧栏存在", await cdp.js("!!document.querySelector('.side')"))
            check("侧栏列出项目", len(projects) >= 1, f"{len(projects)} 个")
            check("有「新建项目」按钮",
                  "新建项目" in (await cdp.js("document.querySelector('.side__new').textContent") or ""))
            print(f"[1] 首屏：{await cdp.js('document.querySelector(\'.top__title strong\').textContent')}"
                  f"　项目数 {len(projects)}　→ {await cdp.shot(out / 'gui-1-result.png')}")

            # ② 点「新建项目」→ 回上传页（用户要的核心：不用删项目就能重新传视频）
            await cdp.js("document.querySelector('.side__new').click()")
            await asyncio.sleep(1.2)
            up = await cdp.js("document.querySelector('.up__title')?.textContent || ''")
            check("点新建后出现上传页", "拖到这里" in (up or ""), up)
            check("新建按钮高亮",
                  await cdp.js("document.querySelector('.side__new').classList.contains('is-active')"))
            check("列表不再高亮任何项目",
                  await cdp.js("document.querySelectorAll('.side__item.is-active').length") == 0)
            check("标题变成新建项目",
                  await cdp.js("document.querySelector('.top__title strong').textContent") == "新建项目")
            check("地址栏变成 ?new=1",
                  await cdp.js("window.location.search") == "?new=1",
                  await cdp.js("window.location.search"))
            print(f"[2] 点新建 → 上传页「{up}」　→ {await cdp.shot(out / 'gui-2-new.png')}")

            # 刷新必须还停在上传页（否则用户看着像"新建没生效"）
            await cdp.call("Page.reload")
            await asyncio.sleep(3.5)
            check("刷新后仍在上传页",
                  "拖到这里" in (await cdp.js("document.querySelector('.up__title')?.textContent || ''") or ""))

            # ③ 从上一步的新建状态点一个**已完成**的项目 → 必须切得过去并显示结果
            #    （回落逻辑最容易在这里打架：draft 刚置 false、project 还没回来）
            done = next((i for i, p in enumerate(projects) if p.get("has_ir")), None)
            if done is None:
                print("[3] 跳过：没有任何已完成的项目，无法验证切回结果页")
            else:
                target = projects[done]
                await cdp.js(f"document.querySelectorAll('.side__item')[{done}].click()")
                await asyncio.sleep(2.5)
                body = await cdp.js("document.querySelector('.pv__text')?.value.slice(0,20) || ''")
                check("点项目切到该项目",
                      await cdp.js("window.location.search") == f"?project={target['id']}",
                      await cdp.js("window.location.search"))
                check("该项目重新高亮",
                      await cdp.js("document.querySelectorAll('.side__item.is-active').length") == 1)
                check("显示结果页提示词", "【" in (body or ""), (body or "").splitlines()[:1])
                print(f"[3] 点「{target['name']}」→ 结果页　→ {await cdp.shot(out / 'gui-3-back.png')}")

            # ③b 修改提示词对话框（PRD F8.4）：验"打得开、写得进、关得上"。
            #     **不在这里真提交** —— 那会烧一次模型调用（实测 10~40 秒），
            #     接口路径由 test_revision.py 的替身测试与一次 live 实测守着。
            if done is not None:
                revise_btn = (
                    "[...document.querySelectorAll('.pv__actions .btn')]"
                    ".find(b => b.textContent.includes('修改提示词'))"
                )
                check("结果页有「修改提示词」按钮", await cdp.js(f"!!({revise_btn})"))
                await cdp.js(f"({revise_btn}).click()")
                await asyncio.sleep(0.8)
                check("对话框打开了", await cdp.js("!!document.querySelector('.rev__box')"))
                title = (await cdp.js("document.querySelector('.rev__title').textContent") or "").strip()
                check("标题是「AI改写」", title == "AI改写", title)
                title_size = await cdp.js(
                    "parseFloat(getComputedStyle(document.querySelector('.rev__title')).fontSize)"
                )
                check("标题比正文大（≥18px）", (title_size or 0) >= 18, f"{title_size}px")
                from_text = (
                    await cdp.js("document.querySelector('.rev__from').textContent") or ""
                ).strip()
                check("标题下方有灰色小字：改写结果来自…",
                      from_text.startswith("改写结果来自") and len(from_text) > len("改写结果来自"),
                      from_text)
                check("灰色小字真的是灰的",
                      await cdp.js("document.querySelector('.rev__from').classList.contains('muted')"))
                check("对话框里没有多余说明文字（只留标题）",
                      not await cdp.js("!!document.querySelector('.rev__hint')"))
                check("提交按钮叫「提交」",
                      (await cdp.js("document.querySelector('.rev__bar .btn--primary').textContent")
                       or "").strip() == "提交")
                check("输入框为空（无 placeholder 提示）",
                      not await cdp.js("document.querySelector('.rev__input').placeholder"))
                check("有意见输入框", await cdp.js("!!document.querySelector('.rev__input')"))
                check("没写意见时提交按钮是禁用的",
                      await cdp.js("document.querySelector('.rev__bar .btn--primary').disabled"))
                # React 受控组件要**走原生 setter 再派发 input**，直接赋 value 它不认
                await cdp.js(
                    "(() => {"
                    "  const el = document.querySelector('.rev__input');"
                    "  const setter = Object.getOwnPropertyDescriptor("
                    "    window.HTMLTextAreaElement.prototype, 'value').set;"
                    "  setter.call(el, '场景换成咖啡馆');"
                    "  el.dispatchEvent(new Event('input', { bubbles: true }));"
                    "  return true;"
                    "})()"
                )
                await asyncio.sleep(0.4)
                check("写了意见后提交按钮可用",
                      not await cdp.js("document.querySelector('.rev__bar .btn--primary').disabled"))
                # 截图要在**对话框开着的时候**拍，否则截到的是关闭后的结果页
                print(f"[3b] 提修改意见对话框　→ {await cdp.shot(out / 'gui-3b-revise.png')}")
                await cdp.js("document.querySelector('.rev__head .btn').click()")
                await asyncio.sleep(0.6)
                check("点 ✕ 能关掉对话框",
                      not await cdp.js("!!document.querySelector('.rev__box')"))

            # ④ 可选：删项目。**自己造一个临时项目来删**，不动用户已有的项目。
            if args.delete:
                tid = await cdp.js(
                    "fetch('/api/projects',{method:'POST',headers:{'Content-Type':'application/json'},"
                    "body:JSON.stringify({name:'走查用临时项目'})}).then(r=>r.json())"
                    ".then(d=>d.project_id)"
                )
                print(f"[4] 临时项目 {tid}（走查后会自动清掉）")
                await cdp.call("Page.navigate", url=f"{args.base}?project={tid}")
                await asyncio.sleep(3.5)
                clicked = await cdp.js(
                    "(() => {"
                    "  const item = document.querySelector('.side__item.is-active');"
                    "  if (!item) return 'no-active';"
                    "  item.parentElement.querySelector('.side__del').click();"
                    "  return 'clicked';"
                    "})()"
                )
                check("找得到要删的当前项目", clicked == "clicked", clicked)
                await asyncio.sleep(2.5)
                check("删掉当前项目后回到上传页",
                      "拖到这里" in (await cdp.js(
                          "document.querySelector('.up__title')?.textContent || ''") or ""))
                check("标题变成新建项目",
                      await cdp.js("document.querySelector('.top__title strong').textContent")
                      == "新建项目")
                check("没有弹 404 错误条",
                      not await cdp.js("document.querySelector('.banner')?.textContent || ''"))
                # 兜底：界面上没删干净就直接调接口删，别给用户留垃圾
                gone = await cdp.js(
                    f"fetch('/api/projects/{tid}',{{method:'DELETE'}})"
                    ".then(r=>({status:r.status})).catch(e=>({status:String(e)}))"
                )
                check("临时项目已清理", (gone or {}).get("status") in (200, 404), gone)
                print(f"[4] 删除路径走完　→ {await cdp.shot(out / 'gui-4-deleted.png')}")
    finally:
        proc.terminate()

    print()
    failed = 0
    for label, ok, detail in checks:
        print(f"  {'✓' if ok else '✗'} {label}　（{detail}）")
        failed += 0 if ok else 1
    print(f"\n{'[OK] 前端走查全部通过' if failed == 0 else f'[FAIL] {failed} 项没过'}"
          f"　截图在 {out}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="前端走查（headless 浏览器真点一遍）")
    parser.add_argument("--base", default="http://127.0.0.1:8765/")
    parser.add_argument("--browser", default="", help="Edge/Chrome 可执行文件路径")
    parser.add_argument("--port", type=int, default=9223, help="DevTools 端口")
    parser.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "mod-gui-shots"))
    parser.add_argument("--delete", action="store_true",
                        help="多走一步：造一个临时项目并删掉它，验证删除路径")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except ImportError:
        print("[X] 缺 websockets：pip install -e \".[app]\" 会带上（uvicorn[standard]）")
        return 2


if __name__ == "__main__":
    sys.exit(main())
