"""启动脚本：一个命令跑起整个应用。

    py scripts/run.py            # 默认 127.0.0.1:8000
    py scripts/run.py --port 9000

会自动检查环境（.env、ffmpeg、前端构建产物），有问题直接说清楚，
而不是启动后跑一半才报错。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(settings) -> list[str]:
    """返回阻断级问题列表。**启动前把问题说清楚**，不留到运行时报错。"""
    problems: list[str] = []

    if not settings.has_llm:
        problems.append(
            "未配置 MIMO_API_KEY：复制 .env.example 为 .env 并填入 Key"
            "（反推需要视觉模型与 ASR）"
        )
    if shutil.which(settings.ffmpeg_bin) is None:
        problems.append(f"找不到 ffmpeg（{settings.ffmpeg_bin}），请在 .env 里配置绝对路径")
    if shutil.which(settings.ffprobe_bin) is None:
        problems.append(f"找不到 ffprobe（{settings.ffprobe_bin}），请在 .env 里配置绝对路径")

    dist = ROOT / "web" / "dist"
    if not dist.is_dir():
        problems.append(
            "前端还没构建：cd web && npm run build"
            "（或开发期单跑 npm run dev，用 5173 端口访问）"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="启动爆款视频复刻提示词 Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    from engine.app.api import create_app
    from engine.app.config import Settings

    settings = Settings.load()

    if not args.skip_checks:
        problems = check(settings)
        if problems:
            print("[X] 启动前检查未通过：\n")
            for item in problems:
                print(f"  - {item}")
            print("\n提示：--skip-checks 可跳过检查（仅用于排查问题）")
            return 1

    print("爆款视频复刻 · 提示词 Agent")
    print(f"  框架版本   {__import__('engine.core.types', fromlist=['x']).FRAMEWORK_VERSION}")
    print(f"  视觉模型   {settings.model_vision}")
    print(f"  文本模型   {settings.model_text}")
    # **自检模型要单独打出来**（用户 2026-10-07 问"你是不是没换模型"）：
    # 它默认和文本模型**不一样**（实测换它以后自检 563 秒 → 52 秒），
    # 但横幅原来只印三项，看起来就像没换过。
    review_model = settings.model_review or settings.model_text
    same = "" if settings.model_review else "（跟随文本模型）"
    print(f"  自检模型   {review_model}{same}")
    # 改写模型同理（用户 2026-10-08：所有 key 都要可换端点）：
    # 它可能跑在**另一个端点**上，所以端点也一并印出来 ——
    # 否则"我明明填了 DeepSeek，怎么还是原来的模型"会再问一次。
    revise_base = settings.revise_endpoint[0]
    revise_same = "" if settings.model_revise else "（跟随自检模型）"
    revise_host = revise_base.split("//")[-1].split("/")[0]
    print(f"  改写模型   {settings.resolved_model_revise}{revise_same}"
          f"  @ {revise_host}"
          f"{'' if settings.revise_api_key else '（用 MiMo 的 Key）'}")
    print(f"  ASR 模型   {settings.model_asr}")
    print(f"  工作区     {settings.workspace_dir}")
    print(f"\n  界面地址   http://{args.host}:{args.port}")
    print(f"  接口文档   http://{args.host}:{args.port}/docs\n")

    import uvicorn

    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
