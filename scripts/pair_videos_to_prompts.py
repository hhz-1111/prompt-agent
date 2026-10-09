"""把「测试素材视频」和「答案卷里的具体句子」逐句对上，人工核对用。

用户的要求：不要拿测试视频一句一句去找答案，要**能说出哪条视频该用哪句提示词**。
这个脚本就是那个配对过程的可复核版本：对每条测试视频，打印
  - 它命中了哪份答案卷、命中了哪些**具体句子**（逐字）
  - 以及为什么（台词重合 / 场景词 / 固定模块）
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTRACT = ROOT / "_extract"
PROMPTS = ROOT / "workspace" / "benchmark" / "prompts"

SPLIT_RE = re.compile(r"[\n。；;]+")
PLACEHOLDER_RE = re.compile(r"【[^】]{0,40}】")


def sentences(text: str) -> list[str]:
    out = []
    for raw in SPLIT_RE.split(text):
        cleaned = PLACEHOLDER_RE.sub("", raw).replace(" ", "").strip(" -·•*#>\t：")
        if len(cleaned) >= 6:
            out.append(cleaned)
    return out


# 每条视频 → 它最像的答案卷（按台词与场景词人工确认后写死，供复核）
PAIRING = {
    "菜市场采访-苏": "素人街采-苏",
    "访谈4": "主持人访谈-苏",
    "魏巍演播厅体式组-丹": "IP演播厅",
    "IP演播厅带练-丹": "IP演播厅",
    "IP户外教学-丹": "IP教学",
    "瑜伽反站式带练-丹": "IP教学",
    "爆款视频AI复刻刘畊宏": "AI情景对话",
    "39f50b88e63ddaa52888e1e6a50631d8": "AI情景对话",
    "9d7790464aa4fed8d4b5a5534aa5ea8c": "AI情景对话",
    "a67995ba0177b8ad2febfd83412c930b": "AI情景对话",
}


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for video_name, sheet_name in PAIRING.items():
        if only and only not in video_name:
            continue
        sheet = EXTRACT / f"{sheet_name}.txt"
        prompt = PROMPTS / f"{video_name}.txt"
        if not sheet.exists() or not prompt.exists():
            print(f"[跳过] {video_name} / {sheet_name}")
            continue

        generated = prompt.read_text(encoding="utf-8")
        generated_flat = PLACEHOLDER_RE.sub("", generated).replace(" ", "")
        sheet_sentences = sentences(sheet.read_text(encoding="utf-8"))

        hits = [s for s in sheet_sentences if s and s in generated_flat]
        print("=" * 96)
        print(f"视频：{video_name}")
        print(f"对应答案卷：{sheet_name}")
        print(f"答案卷句子数 {len(sheet_sentences)}，其中**逐字命中** {len(hits)} 条")
        if hits:
            print("逐字命中的句子：")
            for s in hits:
                print(f"   ✓ {s[:78]}")
        remainder = [s for s in sheet_sentences if s not in hits]
        print(f"未命中的句子 {len(remainder)} 条（前 8 条）：")
        for s in remainder[:8]:
            print(f"   · {s[:78]}")
        print()

    # 反向：生成产物里的每一句，追溯它来自答案卷哪一句
    if only:
        prompt = PROMPTS / f"{only}.txt"
        if prompt.exists():
            print("=" * 96)
            print(f"反向追溯：{only} 的每一句 → 答案卷里的出处")
            pool = {}
            for sheet in EXTRACT.glob("*.txt"):
                for s in sentences(sheet.read_text(encoding="utf-8")):
                    pool.setdefault(s, sheet.stem)
            for line in prompt.read_text(encoding="utf-8").splitlines():
                flat = PLACEHOLDER_RE.sub("", line).replace(" ", "").strip()
                if len(flat) < 8:
                    continue
                source = "（引擎生成，答案卷里没有）"
                best = ""
                for s, stem in pool.items():
                    if s and s in flat:
                        source = stem
                        best = s
                        break
                print(f"   [{source:<8}] {line[:74]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
