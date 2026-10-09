"""从规范文档构建短语资产库（assets/phrases/phrases.json）。

为什么要脚本而不是手写 JSON：

    1. 禁止项库与句式模板的**逐字一致性**是硬要求（PRD F3.5 / G14）。
       手抄一遍就可能抄错一个字，而错一个字就毁掉约束强度。
    2. 规范文档（TEMPLATES）是唯一权威来源，资产必须可重建。
    3. 每条资产记录"来源文档"，评审时能一键回溯。

用法：
    py scripts/build_phrases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "phrases" / "phrases.json"

PHRASE_LIB_VERSION = "mod-phrases-v1"

# ==========================================================================
# T-* 句式模板（框架 §3、TEMPLATES §2）
# 逐字来自规范文档。tier 见 PHRASES §1.1 约束强度光谱。
# ==========================================================================
TEMPLATES: dict[str, dict[str, Any]] = {
    "T-PERSON": {
        "label": "人物一致性",
        "dimension": "人物一致性",
        "source": "TEMPLATES §2.1",
        "sentence": "{角色名}形象参考{多视图句柄}，用于{角色名}的五官细节、面部轮廓、外貌、发型、体态和气质参考，呈现视觉辨识度，保持自然真实质感。",
        "tier": "template",
        "levels": {
            "standard": "锁定人物五官脸型，全程同一人物，五官稳定不漂移",
            "extreme": "面部特征100%相同，保持相同的骨骼结构、眼距和下颚几何形状，发型100%还原，禁止美化，禁止改变年龄",
        },
        "mandatory_fragments": {
            "standard": ["锁定人物五官脸型", "五官稳定不漂移"],
            "extreme": ["面部特征100%相同", "禁止美化"],
        },
        "wardrobe": "着装100%还原{多视图句柄}",
    },
    "T-SKIN": {
        "label": "皮肤质感",
        "dimension": "皮肤质感",
        "source": "TEMPLATES §2.2",
        "tier": "fixed",
        "variants": {
            "full": "光影自然，人物面光自然充足，画面有通透感，可以看到自然毛孔、轻微皮肤纹理和柔和光泽，肤质自然不过度磨皮。鼻翼和脸颊有真实的毛孔细节和细小肌理，整体皮肤质感清晰、真实、自然光照。",
            "brief": "人物皮肤保留毛孔、轻微纹理与柔和光泽，无过度磨皮；鼻翼、脸颊肌理清晰可见。",
        },
        "mandatory_fragments": ["毛孔", "磨皮"],
    },
    "T-QUALITY": {
        "label": "画质预设",
        "dimension": "画质预设",
        "source": "TEMPLATES §2.3",
        "tier": "vocab",
        "variants": {
            "P1": "auto white balance slightly cool, slightly underexposed, natural phone camera noise texture, no color grading, no filters, unedited, 2024 casual photo aesthetic, mundane everyday scene",
            "P2": "苹果手机前置摄像头原生自拍质感，无滤镜，纪实生活化，真实彩色写实画质，允许轻微自然噪点、运动模糊、镜头随机微抖动和呼吸感，非完美打光，非完美画质，皮肤纹理清晰可见",
            "P3": "ARRI原生色彩，光影逻辑连贯，色彩真实自然；原生相机噪点，细节完整，动态流畅自然",
        },
    },
    "T-BLINK": {
        "label": "眨眼节拍",
        "dimension": "眨眼节拍",
        "source": "TEMPLATES §2.4",
        "tier": "vocab",
        # **逐字照抄 TEMPLATES §2.4 的三档原句**。
        # 曾经这里的两档是手打的，抄漏了「隔」并多出空格：
        #   ✗ 人物每 3-4 秒自然眨眼一次，每次约 0.1 秒，融入说话节奏，眨眼频率自然
        #   ✓ 人物每隔3-4秒自然快速眨眼一次（每次约0.1秒），融入说话节奏，眨眼频率自然
        # 后果很严重：引擎把"每 3-4 秒"这句写进交付物，与历史成功提示词
        # **逐字对不上** —— 而固定模块的价值恰恰在逐字。
        # 修正后 scripts/phrase_overlap.py 的逐字复现率立即上升。
        "variants": {
            "3-4s": "人物每隔3-4秒自然快速眨眼一次（每次约0.1秒），融入说话节奏，眨眼频率自然。",
            "5s": "人物每隔5秒自然快速眨眼一次（每次约0.1秒），融入对话节奏，眨眼频率自然。",
            "3s": "人物每隔3秒自然快速眨眼一次（每次约0.1秒），融入动作节奏，眨眼频率自然。",
        },
        "mandatory_fragments": ["眨眼"],
        "adjustable": "数字（3/3-4/5 秒）可改，节奏动词与「每次约0.1秒」「眨眼频率自然」必须逐字",
    },
    "T-SCENE": {
        "label": "场景一致性",
        "dimension": "场景一致性",
        "source": "TEMPLATES §2.5",
        "tier": "template",
        "sentence": "场景始终为{场景图句柄}的{场景类型}环境，保持同一位置和光线氛围，镜头无跳切，背景不虚化，环境清晰可见，自然色调。",
        "mandatory_fragments": ["保持同一位置和光线氛围", "镜头无跳切", "背景不虚化"],
    },
    "T-BLOCK": {
        "label": "站位与视线",
        "dimension": "空间与站位",
        "source": "TEMPLATES §2.6",
        "tier": "template",
        "sentence": "{人物A}始终在画面左侧、{人物B}始终在画面右侧，两人位置全程禁止变动。",
        "mandatory_fragments": ["位置全程禁止变动"],
        "variants": {
            "occlusion": "{前方人物}不要完全遮挡{后方人物}，两人位置全程禁止变动。",
            "eye_line": "{角色}视线方向始终望向{方向}，不转头看镜头。",
            "height": "左边人物和右边人物身高是一样高的，比例要协调。",
        },
    },
    "T-ACTION": {
        "label": "动作参考",
        "dimension": "动作",
        "source": "TEMPLATES §2.7 + 用户 2026-10-07 原话",
        "tier": "template",
        # 用户原话（大意）：「视频里老师和学员的动作参考这个视频，1:1复刻视频里的
        # 肢体轨迹、身体角度、抬手弯腰幅度，不出现任何其他动作，动作自然连贯。」
        # 槽位：{who}=人物（"人物1、人物2"或"画面中人物"）、
        #       {参考视频句柄}=[参考动作视频1]。
        # 开头用**资产的视频用途关键词**「动作轨迹参考」：G12 靠它认定这条素材
        # 已在正文里声明了用途（实测第一版写「的动作参考」被判"没有用途声明"）。
        "sentence": (
            "{who}的动作轨迹参考{参考视频句柄}，1:1复刻其中的肢体轨迹、身体角度、"
            "抬手弯腰幅度，不出现任何其他动作，动作自然连贯。"
        ),
        "mandatory_fragments": ["1:1复刻", "不出现任何其他动作"],
    },
    "T-AUDIO": {
        "label": "声音一致性",
        "dimension": "音色",
        "source": "TEMPLATES §2.8",
        "tier": "template",
        "sentence": "{角色}声音音色参考{音频句柄}，百分百还原{音频句柄}的声音音色，全片保持声音一致性。",
        "mandatory_fragments": ["百分百还原", "全片保持声音一致性"],
        "whisper": {
            "prefix": "（极低压声）",
            "suffix": "气口自然顺滑，全程压低音量的悄悄话，不额外添加台词",
        },
    },
    "T-LIPSYNC": {
        "label": "口型要求",
        "dimension": "口型",
        "source": "TEMPLATES §2.9",
        "tier": "template",
        "sentence": "所有对白保持自然停顿和情绪起伏，口型与台词同步。",
        "mandatory_fragments": ["口型与台词同步"],
    },
    "T-SHOT": {
        "label": "镜头模板",
        "dimension": "镜头与运镜",
        "source": "TEMPLATES §2.10",
        "tier": "vocab",
        "variants": {
            "shot_line": "镜头{序号}：{景别}{机位}，{场景}。{动作}，{表情}，{情绪}。",
            "beat_line": "{序号}. {场景}，{人物}{动作}，{情绪}。",
        },
    },
}

# ==========================================================================
# D 区禁止项库（TEMPLATES §3，逐字）
# ==========================================================================
# **默认开关由用户 2026-09-29 明确指定：只开 D0 / D1 / D7，其余全部关闭。**
#
# 关闭的组不代表条目不存在 —— 它们仍在库里，用户可以在「固定提示词」表格页
# 自己勾选（`whitelist_addable` 只允许增、不允许减）。默认不勾是因为：
#   D2 群体防细化   —— 只在画面里真有群体角色时才有意义
#   D3 结构防改     —— 只在多镜头切分时才有意义
#   D4 泛化约束防补 —— 中频，且与 B2「着装100%还原」容易读起来打架
#   D5 画面纯净化   —— 实测 10 份历史文档里只出现 3/10、1/10、4/10，确认低频
#   D6 内容安全     —— 属平台侧兜底，不必每条提示词都写
#   D8 素材边界     —— 只在素材替换模式下才需要
FORBIDDEN_GROUPS: dict[str, dict[str, Any]] = {
    "D0": {
        "label": "通用底线",
        "default_on": True,
        "note": "用户指定默认开启；几乎每篇必现，属于「永不可裁」条目",
        "items": [
            "全程不生成任何字幕文字，100%透明字幕。",
            "禁止出现bgm、logo、水印、时间戳、分镜格线、批注文字或设计说明元素。",
            "禁止添加未提供的人物性别、年龄、服装、道具或职业细节。",
            "禁止改变人物身份或动作方向。",
            "禁止插画/线稿感、虚化、运动模糊、景深效果。",
            "禁止错误肢体如扭曲手指、多关节。",
            "禁止无关人物或道具漂移。",
            "禁止自行增补对白或口型停顿要求之外的台词。",
        ],
    },
    "D1": {
        "label": "道具防臆造",
        "default_on": True,
        "note": "用户指定默认开启；高频",
        "items": [
            "禁止人为添加话筒、提词器、课程器械等上游未提供的道具。",
            "禁止猜测或标注人物性别、年龄、职业等未提供信息。",
        ],
    },
    "D2": {
        "label": "群体防细化",
        "default_on": False,
        "note": "高频；出现群体角色时开启",
        "items": [
            "禁止将「台下观众」「学员」「小区阿姨」等群体细化成特定个体、性别、年龄或位置。",
            "不改变其他观众的坐姿位置。",
        ],
    },
    "D3": {
        "label": "结构防改",
        "default_on": False,
        "note": "中频；多镜头切分时开启",
        "items": [
            "禁止改变镜头顺序或数量，不并镜、不重排，不自行增加镜头数量。",
            "禁止生成跳镜、并镜、重排或增加额外镜头。",
        ],
    },
    "D4": {
        "label": "泛化约束防补",
        "default_on": False,
        "note": "用户指定默认关闭。中频，很特殊：与 B2「着装100%还原」不矛盾——后者有具体参考对象（确定约束），本条禁的是无对象的泛化表述",
        "items": [
            "禁止补充「服装保持一致」「参考图延续」等泛化一致性描述（上游未提供时）。",
        ],
    },
    "D5": {
        "label": "画面纯净化",
        "default_on": False,
        "note": "中频；属于压缩时可优先丢弃的低频条目",
        "items": [
            "禁止画面出现黑色图形和黑色方块。",
            "禁止出现箭头、标注框、小字批注、编号、标题、导演说明、情绪指示、光线指示。",
            "不要插画、线稿、漫画、3D渲染、CG感。",
        ],
    },
    "D6": {
        "label": "内容安全",
        "default_on": False,
        "note": "用户指定默认关闭；属平台侧兜底，不必每条提示词都写",
        "items": [
            "禁止添加任何视频生成禁项（如性、暴力、政治敏感等）。",
        ],
    },
    "D7": {
        "label": "对白锁定",
        "default_on": True,
        "note": "中频；属于「永不可裁」条目",
        "items": [
            "不要增删对白内容，不要为无声镜头额外补配台词。",
            "禁止修改对白台词、语序、语气词；口型必须与对白同步。",
            "禁止更改对白内容、说话人身份或口型节奏。",
        ],
    },
    "D8": {
        "label": "素材边界",
        "default_on": False,
        "note": "低频；强约束不额外增补",
        "items": [
            "禁止引用未提供的参考素材，不得声称参考了未绑定的素材。",
        ],
    },
}

# ==========================================================================
# 维度改写规则（PRD §6.2.4 / PHRASES §4）
# lock_mode: free=自由填 / vocab=只能从词库换 / fixed=必须逐字
# ==========================================================================
DIMENSIONS: list[dict[str, Any]] = [
    {
        "key": "framing", "label": "景别", "lock_mode": "vocab",
        "vocab": ["全景", "中全景", "中景", "中近景", "近景", "脸部特写"],
        "mandatory": [], "ir_paths": ["creative.C2_beats[].framing"],
        "note": "必写维度，样本覆盖率 10/10 —— 缺失则画面无节奏",
    },
    {
        "key": "camera", "label": "机位", "lock_mode": "vocab",
        "vocab": ["平视", "正面", "过肩", "俯拍", "仰拍", "固定", "手持"],
        # **两个字段都算**：这个维度的词表本来就同时含**角度**词（平视/过肩/俯拍）
        # 与**运镜**词（固定/手持）。原来只挂 `camera`（运镜），于是交付物里
        # 每镜都写着「平视机位拍摄」时，G13 仍报「缺：机位」—— 实测踩到的误报。
        "mandatory": [],
        "ir_paths": ["creative.C2_beats[].camera", "creative.C2_beats[].camera_angle"],
        "note": "必写维度，样本覆盖率 7/10；角度或运镜任一写清即算覆盖",
    },
    {
        "key": "movement", "label": "运镜", "lock_mode": "vocab",
        "vocab": ["推近", "跟拍", "固定", "横摇", "手持"],
        # 三选一：原文里这三个表述等价，命中任一即满足
        "mandatory": [["一镜到底", "机位全程不变", "镜头无跳切"]],
        "ir_paths": ["locked.B4_camera.rig"],
        "note": "必写维度；运镜词可从词库换，但三个锁定表述必须逐字",
    },
    {
        "key": "blink", "label": "眨眼节拍", "lock_mode": "vocab",
        "vocab": ["3-4s", "5s", "3s"],
        "mandatory": ["自然快速", "每次约0.1秒", "眨眼频率自然"],
        "ir_paths": ["locked.B2_person[].blink"],
        "note": "必写维度，覆盖率 10/10 —— 人脸长时间不眨眼，AI 感明显。数字可改，节奏动词逐字",
    },
    {
        "key": "skin", "label": "皮肤质感", "lock_mode": "fixed",
        "vocab": [], "mandatory": ["毛孔", "无过度磨皮"],
        "ir_paths": ["locked.B2_person[].skin", "locked.B5_quality.skin_text"],
        "note": "必写维度，覆盖率 10/10 —— 整句逐字，缺失则过度磨皮成假脸",
    },
    {
        "key": "emotion", "label": "情绪", "lock_mode": "free",
        "vocab": ["惊讶", "好奇", "温和", "自信", "无奈", "轻快", "关切", "平稳", "从容", "肯定"],
        "mandatory": [],
        "ir_paths": ["creative.C2_beats[].emotion", "creative.C2_beats[].dialogue.tone"],
        "note": "必写维度，覆盖率 7/10 —— 可自由填，也可从词库选；目标准确率 ≥70% 即合理",
    },
    {
        "key": "person_consistency", "label": "人物一致性", "lock_mode": "vocab",
        "vocab": ["standard", "extreme"],
        "mandatory": ["锁定人物五官脸型", "五官稳定不漂移"],
        "ir_paths": ["locked.B2_person[].lock_strength"],
        "note": "必写维度，覆盖率 8/10 —— 角色名可改，锁定档位只能切换，硬表述必须逐字",
    },
    {
        "key": "action", "label": "动作", "lock_mode": "vocab",
        "vocab": [],
        "mandatory": ["1:1复刻", "不出现任何其他动作"],
        "ir_paths": ["creative.C2_beats[].blocking"],
        "note": "素材名可改，两个硬表述必须逐字；把 1:1复刻 改成「参考一下」是最常见的改写事故",
    },
    {
        "key": "audio", "label": "音色", "lock_mode": "vocab",
        "vocab": [], "mandatory": ["百分百还原", "全片保持声音一致性"],
        "ir_paths": ["locked.B6_audio[].audio_ref"],
        "note": "必写维度，覆盖率 4/10；素材名可改，两个硬表述必须逐字",
    },
    {
        "key": "dialogue", "label": "台词", "lock_mode": "free",
        "vocab": [], "mandatory": [],
        "ir_paths": ["creative.C2_beats[].dialogue.line"],
        "note": "用户填，Agent 绝不代写（红线 3）",
    },
    {
        "key": "forbidden", "label": "禁止项", "lock_mode": "vocab",
        "vocab": list(FORBIDDEN_GROUPS.keys()),
        "mandatory": ["100%透明字幕"],
        "ir_paths": ["forbidden"],
        "note": "可增不可减；通用禁止项全库逐字",
    },
    {
        "key": "scene", "label": "场景一致性", "lock_mode": "vocab",
        "vocab": ["菜市场", "茶楼雅间", "小区花园步行道", "商场走廊", "室内客厅",
                  "户外公园", "瑜伽室", "演播厅舞台", "演播厅访谈区", "客厅沙发区"],
        "mandatory": ["保持同一位置和光线氛围", "镜头无跳切", "背景不虚化"],
        "ir_paths": ["locked.B1_scene.type", "locked.B1_scene.anchor"],
        "note": "场景类型可从词库换，锚定句必须逐字",
    },
    {
        "key": "blocking", "label": "站位", "lock_mode": "fixed",
        "vocab": [], "mandatory": ["位置全程禁止变动"],
        "ir_paths": ["locked.B3_blocking.positions"],
        "note": "LOCK 位，一字不可动；错了人物关系错乱（致命级）",
    },
    {
        "key": "quality", "label": "画质预设", "lock_mode": "vocab",
        "vocab": ["P1", "P2", "P3"],
        "mandatory": [], "ir_paths": ["locked.B5_quality.preset"],
        "note": "覆盖率 4/10；三套预设按原片质感选",
    },
]

# ==========================================================================
# 约束有效性分级（PHRASES §5 / PRD §6.3.4）
# ==========================================================================
GRADE_PATTERNS: list[dict[str, str]] = [
    {"grade": "F", "pattern": r"尽量|尽可能|最好|最好是|希望|大概|差不多|争取", "label": "愿望式"},
    {"grade": "E", "pattern": r"保持一致(?!性参照)|参考图延续|延续参考|自然一些|好看一点|风格统一", "label": "泛化表述"},
    {"grade": "A", "pattern": r"\d+(\.\d+)?\s*(%|秒|s|倍|张|个|条)", "label": "量化"},
    {
        "grade": "D",
        # 「禁止 + 枚举」：出现顿号/逗号串联的**名词**清单即为逐条禁止。
        # 首项必须是非数字名词，否则「覆盖率100%的维度」这种量词会被误当枚举项。
        # 单纯一个「禁止…」是 C 级目标态；列举才是 D 级。
        "pattern": r"禁止[^。；\n]{0,60}?(?!\d)[\u4e00-\u9fffA-Za-z]{2,}[、，,][^。；\n]{0,60}",
        "label": "逐条禁止",
    },
    {"grade": "C", "pattern": r"100%透明|透明字幕|锁定|全程|始终|必须|禁止", "label": "目标状态"},
    {"grade": "B", "pattern": r"鼻翼|脸颊|眼距|下颚|毛孔|肌理|骨骼结构|面部轮廓|五官|脸型", "label": "几何"},
]

# 强约束降级检测（PRD F3.5）：左边是必须逐字的原文，右边是会把强度改弱的替换写法。
#
# 设计纪律：**替换词只收"明确的弱化短语"**，不收裸的短词。
# 裸的「参考」会命中「形象参考[图1]」「音色参考[音频1]」——那是规范强制写法；
# 而「动作参考一下原片」恰恰是最典型的降级，必须能抓到。
DOWNGRADE_RULES: list[dict[str, Any]] = [
    {
        "id": "DG-ACTION",
        "protected": "1:1复刻",
        "downgrades": ["参考一下", "模仿一下", "参考着做", "照着做就行", "大致还原", "差不多就行"],
        "why": "把 1:1复刻 改成参考一下，会直接导致人物漂移 —— 最常见的改写事故",
    },
    {
        "id": "DG-LOCK-FACE",
        "protected": "锁定人物五官脸型",
        "downgrades": ["保持人物相似", "人物基本一致", "尽量像", "长相接近", "看起来像"],
        "why": "五官锁定一旦弱化，跨镜头必然漂移",
    },
    {
        "id": "DG-SUBTITLE",
        "protected": "100%透明字幕",
        "downgrades": ["不要有字幕", "尽量不出现字幕", "字幕可有可无", "减少字幕"],
        "why": "正向目标态才算约束；愿望式等于没写",
    },
    {
        "id": "DG-AUDIO",
        "protected": "百分百还原",
        "downgrades": ["音色接近", "声音相似", "参考音色", "尽量像原声"],
        "why": "音色约束弱化会让全片声音漂移",
    },
    {
        "id": "DG-NO-OTHER-ACTION",
        "protected": "不出现任何其他动作",
        "downgrades": ["动作不要太多", "尽量少加动作", "动作简洁"],
        "why": "泛化表述无法约束模型",
    },
    {
        "id": "DG-BLOCKING",
        "protected": "位置全程禁止变动",
        "downgrades": ["位置基本不变", "站位尽量保持", "不要换位置"],
        "why": "站位是致命级维度，弱化即失去复刻意义",
    },
]


def main() -> int:
    corpus_manifest = ROOT / "assets" / "corpus" / "manifest.json"
    coverage: dict[str, Any] = {}
    if corpus_manifest.exists():
        data = json.loads(corpus_manifest.read_text(encoding="utf-8"))
        coverage = data.get("statistics", {}).get("dimension_coverage", {})

    # 把实证覆盖率贴回维度定义 —— 必写维度不能凭感觉，必须有数据
    for dim in DIMENSIONS:
        stat = coverage.get(dim["label"])
        dim["evidence_coverage"] = stat["rate"] if stat else None
        dim["evidence_hit"] = f"{stat['hit']}/{stat['total']}" if stat else None

    mandatory = [d["key"] for d in DIMENSIONS if (d["evidence_coverage"] or 0) >= 0.6]

    library = {
        "phrase_lib_version": PHRASE_LIB_VERSION,
        "framework_version": "mod-framework-v1",
        "note": "短语资产库。tier 见约束强度光谱：fixed=必须逐字 / template=可换槽位 / vocab=只能从词库换 / free=可自由填。",
        "mandatory_dimensions": mandatory,
        "mandatory_note": "覆盖率 ≥60% 的维度即必写维度（框架 §11 G13）。眨眼与皮肤在全部样本中出现，是模型默认会做错的地方。",
        "templates": TEMPLATES,
        "forbidden_groups": FORBIDDEN_GROUPS,
        "dimensions": DIMENSIONS,
        "grade_patterns": GRADE_PATTERNS,
        "downgrade_rules": DOWNGRADE_RULES,
        "capacity": {
            "note": "Seedance 能力矩阵不硬编码，这里是随版本更新的默认配置（框架 §2.3）。",
            # prompt_char_budget：交付提示词的**中文字数**预算（英文/标点不计，见
            # assembler.PromptDocument.char_count）。框架 §11 写的是"≤500 字建议值"，
            # 这里是**运行时采用的**预算 —— 2026-09-29 按用户要求调为 800。
            #
            # 为什么可以调高而不是改框架：框架 §9.2 末尾原文是
            # 「≤500 字**或已标注压缩**」，本身就是建议值而非硬上限；
            # 而实测 24 条真实爆款视频即便按 §9.2 全量压缩后仍在 650~850 字，
            # 因为"永不可裁"区（B3 站位/B6 音色/C3 台词/D0/D7）与节拍行
            # （每行 100~110 字，不在可裁清单里）本身就占满了。
            # 500 会让 G7 对**每一条**都告警，"闸门全绿率"因此恒为 0，指标失去意义。
            #
            # 改这个数只影响**压缩力度**与 G7 判定，不影响任何内容规则。
            "models": {
                "seedance-2.0": {
                    "image": [1, 9], "video": [0, 3], "audio": [0, 3],
                    "mixed_total": 12,
                    "video_duration_sec": [2, 15], "audio_duration_sec": [2, 15],
                    "clip_duration_sec": [4, 15],
                    "return_last_frame": True,
                    "prompt_char_budget": 800,
                },
                "seedance-2.5": {
                    "image": [1, 30], "video": [0, 10], "audio": [0, 10],
                    "mixed_total": 40,
                    "video_duration_sec": [2, 30], "audio_duration_sec": [2, 30],
                    "clip_duration_sec": [4, 15],
                    "return_last_frame": True,
                    "prompt_char_budget": 800,
                },
            },
        },
        "speech_rate": {
            "note": "语速估算（AGENT §5.4）。当前为估算值，上线后用真实生成音频反算校准。",
            "chars_per_sec": {"min": 4.0, "typical": 5.0, "max": 6.0},
            "boundary": "慢速4字/秒、常规5字/秒、快速6字/秒",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(library, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] 短语资产库 → {OUT}")
    print(f"     句式模板 {len(TEMPLATES)} 个，维度 {len(DIMENSIONS)} 个")
    print(f"     禁止项 {sum(len(g['items']) for g in FORBIDDEN_GROUPS.values())} 条 / {len(FORBIDDEN_GROUPS)} 组")
    print(f"     降级规则 {len(DOWNGRADE_RULES)} 条")
    print(f"     必写维度（覆盖率≥60%）：{mandatory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
