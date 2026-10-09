"""反推结果 → Framework IR。

这个模块把"分析结论"翻译成"契约结构"。它的纪律只有一条，但最要紧：

    **低置信度宁可留空，也不猜。**（PRD 红线 7）

所以每个字段的赋值前都要过 ``_apply``：置信度 <0.50 就不写，
并把该位置登记为"待用户补填"。猜错会导致后续全链路偏移，而用户很难发现。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from engine.core import framework as fw
from engine.core.contracts import InferenceReport, SemanticSummary
from engine.core.types import (
    CONFIDENCE_REVIEW,
    UNKNOWN,
    BlinkBeat,
    ConfidenceBasis,
    ContentKind,
    Genre,
    Provenance,
    QualityPreset,
    RefType,
    ShotMode,
    Span,
)

# 场景类型 → 场景锚定句（有具体参考对象才算确定约束）。
# `{拍摄质感}` 是 T-SCENE-1 规范里的占位符（TEMPLATES §2.5），
# 位置在"保持同一位置和光线氛围"之后、"镜头无跳切"之前。
SCENE_ANCHOR_WITH_TYPE = (
    "场景始终为{场景句柄}的{场景类型}环境，保持同一位置和光线氛围，"
    "{拍摄质感}，镜头无跳切，背景不虚化，环境清晰可见，自然色调"
)
SCENE_ANCHOR_NO_TYPE = (
    "场景始终为{场景句柄}，保持同一位置和光线氛围，"
    "{拍摄质感}，镜头无跳切，背景不虚化，环境清晰可见，自然色调"
)

# `{拍摄质感}` 按画质档位取值。措辞取自 10 份历史成功提示词的实测写法：
# 手机原生 P1 写"苹果手机拍摄质感"，手机前置自拍 P2 写"苹果手机前置摄像头原生自拍质感"，
# ARRI P3 写"ARRI 原生色彩"，iPhone17 P4 写"iPhone17手机拍摄质感"。
# 这一条在原来的实现里被整块漏掉了 —— 于是"真实相机拍摄质感"这个
# 出现频率最高的固定模块，反推产物 0/24 条写出来（详见 docs/反推提示词对照.md）。
CAMERA_TEXTURE_BY_PRESET: Mapping[str, str] = {
    QualityPreset.P1_PHONE.value: "苹果手机拍摄质感",
    QualityPreset.P2_SELFIE.value: "苹果手机前置摄像头原生自拍质感",
    QualityPreset.P3_ARRI.value: "ARRI原生色彩",
}
CAMERA_TEXTURE_FALLBACK = "真实相机拍摄质感"
"""档位未知时的兜底。**这是观感描述而不是技术参数**，任何实拍素材都成立。"""

# 内容类型 → 眨眼节拍（框架 §3 B5）
BLINK_BY_GENRE: Mapping[Genre, BlinkBeat] = {
    Genre.DIALOGUE: BlinkBeat.FIVE,
    Genre.TEACHING: BlinkBeat.THREE_TO_FOUR,
    Genre.STUDIO: BlinkBeat.THREE_TO_FOUR,
    Genre.STREET_INTERVIEW: BlinkBeat.FIVE,
    Genre.REPLACEMENT: BlinkBeat.THREE,
}


def build_ir(report: InferenceReport) -> dict:
    """把一次反推组装成 Framework IR 树。"""
    ir = fw.empty_ir(
        genre=(report.semantic.genre.value if report.semantic else Genre.DIALOGUE.value),
        shot_mode=(
            report.semantic.shot_mode.value if report.semantic else ShotMode.ONE_TAKE.value
        ),
        aspect_ratio=report.video.aspect_ratio,
    )
    # 时长为 0 是致命错误，必须如实记录而不是编一个
    ir["meta"]["total_duration_sec"] = round(report.video.duration_sec, 3)
    ir["meta"]["replacement_level"] = "L2"
    _fill_content_kind(ir, report)

    _fill_scene(ir, report)
    _fill_persons(ir, report)
    _fill_camera(ir, report)
    _fill_blocking(ir, report)
    _fill_quality(ir, report)
    _fill_beats(ir, report)
    _fill_logline(ir, report)
    _fill_audio(ir, report)
    _fill_forbidden(ir, report)
    _fill_delivery(ir, report)
    # 素材清单最后填：它依赖人物与音色清单已经就位
    ir["references"] = reference_manifest(ir)
    _ensure_action_video_slot(ir)
    return ir


def _ensure_action_video_slot(ir: dict) -> None:
    """动作类片子：**登记一条"参考动作视频"素材**，句柄写成 ``[参考动作视频1]``。

    用户 2026-10-07：「我要的是提示词，你把『参考上传的原片』改成
    **[参考动作视频1]**，我直接复制提示词上去，这个参考动作视频我会自己在
    生成平台上面上传。」

    所以：
    - 素材清单里登记一条视频素材，带自定义句柄标签「参考动作视频」；
    - ``asset_id`` 留空 —— 它确实**还没绑定**（用户会在平台上自己传），
      我们不假装它绑好了；
    - 句柄在正文里出现时，G1 会因为"这是本 IR 声明的句柄"而放行
      （见 ``gates._find_unresolved_handles``）。
    """
    from engine.core.assembler import needs_action_physics

    if not needs_action_physics(ir):
        return
    for ref in ir.get("references", []):
        if str(ref.get("token_label") or "").strip() == ACTION_VIDEO_LABEL:
            return
    next_step = len(ir.get("references", [])) + 1
    ir.setdefault("references", []).append({
        "slots": [ACTION_VIDEO_LABEL],
        "type": "video",
        "placeholder": f"【{ACTION_VIDEO_LABEL}】",
        "asset_id": "",
        "seedance_role": "reference_video",
        "token_label": ACTION_VIDEO_LABEL,
        "prep_step": next_step,
        "prep_prompt": (
            "用户自行在生成平台上传原片作为参考视频；"
            "本段 1:1 复刻动作的约束依赖它"
        ),
    })


ACTION_VIDEO_LABEL = "参考动作视频"
"""动作类片子里"参考视频"的句柄标签 —— 正文写成 ``[参考动作视频1]``。"""


# ----------------------------------------------------------------------
# 清洗
# ----------------------------------------------------------------------
_PLACEHOLDER_VALUES = frozenset({
    "", "unknown", "Unknown", "UNKNOWN", "未知", "无", "none", "None", "null", "N/A", "n/a",
})

_UNKNOWN_SEGMENT = re.compile(
    r"(?:^|[，,、；;：:\s])(?:unknown|Unknown|UNKNOWN|未知)(?:[，,、；;：:\s]|$)"
)
"""```unknown``` 作为**独立小句**出现在句子中间的情形。

实测踩过：模型把表情字段写成「unknown，面部朝向侧对镜头，看不清具体表情」——
整串不等于 unknown，``_text`` 的判据放它过去，于是交付物里出现
「……走向前台，**unknown**，面部朝向侧对镜头」。内部哨兵绝不能进交付物。
"""


def _text(value: Any) -> str:
    """把模型的"空值表述"统一成空串。

    模型经常把 unknown 写成中文"未知"或"无"。这些**不是内容**，
    必须当成空处理 —— 否则会拼出"的未知环境"这种垃圾句子，
    而且会让闸门以为该维度已填充。

    整串是哨兵词的直接判空；**混在句子中间的哨兵小句单独摘掉**
    （见 ``_UNKNOWN_SEGMENT``）。
    """
    text = str(value or "").strip()
    if text in _PLACEHOLDER_VALUES:
        return ""
    if "unknown" in text.lower() or "未知" in text:
        text = _UNKNOWN_SEGMENT.sub("，", text)
        text = re.sub(r"[，,]{2,}", "，", text).strip(" ，,、；;：:")
        if text in _PLACEHOLDER_VALUES:
            return ""
    return text


def _placeholder(value: Any) -> str:
    """需要落库为 UNKNOWN 的字段。"""
    return _text(value) or UNKNOWN


def _fill_content_kind(ir: dict, report: InferenceReport) -> None:
    """落 ``meta.content_kind``：**这条片子有没有需要复刻的身体动作**。

    用户 2026-10-07 反馈：口播片里出现了【动作与物理】（无对象的空约束），
    动作片里缺 1:1 复刻动作的句子。判据不能用 genre —— 实测瑜伽带练被判成
    dialogue、舞台口播被判成 teaching。

    优先级：**模型在分析阶段的判断** > 按画面动作词的兜底判据。
    """
    from engine.core.assembler import looks_like_action

    model_kind = str(getattr(report.semantic, "content_kind", "") or "").strip().lower()
    if model_kind in ("action", "speech"):
        ir["meta"]["content_kind"] = model_kind
        return
    ir["meta"]["content_kind"] = (
        ContentKind.ACTION.value if looks_like_action(ir) else ContentKind.SPEECH.value
    )


def _fill_scene(ir: dict, report: InferenceReport) -> None:
    """场景类型 + 场景锚定句。"""
    scene_type = _text(report.semantic.scene_type if report.semantic else "")
    texture = _camera_texture(ir)
    if scene_type:
        ir["locked"]["B1_scene"]["type"] = scene_type
        ir["locked"]["B1_scene"]["anchor"] = SCENE_ANCHOR_WITH_TYPE.format(
            场景句柄="【场景参考图】", 场景类型=scene_type, 拍摄质感=texture
        )
    else:
        # 反推不出场景类型：留空 + 锚定句不带场景名，**不写"未知环境"这种垃圾句子**
        ir["locked"]["B1_scene"]["type"] = UNKNOWN
        ir["locked"]["B1_scene"]["anchor"] = SCENE_ANCHOR_NO_TYPE.format(
            场景句柄="【场景参考图】", 拍摄质感=texture
        )
    ir["locked"]["B1_scene"]["cross_shot_consistency"] = True


def _camera_texture(ir: Mapping[str, Any]) -> str:
    """按画质档位取"拍摄质感"措辞。

    ``_fill_quality`` 在 ``_fill_scene`` 之后跑，但 ``empty_ir`` 已把
    preset 的默认值填好了，所以这里读得到；档位是枚举外的值时走兜底。
    """
    preset = str(ir.get("locked", {}).get("B5_quality", {}).get("preset") or "")
    return CAMERA_TEXTURE_BY_PRESET.get(preset, CAMERA_TEXTURE_FALLBACK)


# 复刻场景里"人物"的上限。
# 超过这个数的多半是模型把群体（学员/观众/顾客）当成了独立个体 ——
# 实测有一条带练视频被抽出 10 个"人物"，于是生成 10 张多视图，
# 直接撞爆 Seedance 2.0 的 9 张图配额（G2 阻断）。
MAX_REVERSE_ENGINEERED_PERSONS = 4

# 群体词：出现这些的"人物"标签应视为群体，不进人物清单（PRD D2 群体防细化）
_GROUP_WORDS = ("观众", "学员", "顾客", "路人", "人群", "大家", "其他人", "群众", "粉丝")

# 位置词：**相对视角**，同一人在不同镜头里会被标成不同位置。
# 实测：同一个人在 5 个镜头里被叫成「左侧蓝衣男子」「右侧蓝衣男子」「后方蓝衣男子」——
# 完全无法作为人物身份依据，必须从标签里剥掉。
# 位置信息另有来源：CV 人体框中心 x（确定性量测），比模型描述准得多。
_POSITION_WORDS = (
    "画面", "左侧", "右侧", "左边", "右边", "左半", "右半",
    "左方", "右方", "前方", "后方", "左前", "右前", "左后", "右后",
    "前景", "背景", "居中", "中间", "中央",
)
"""位置词：**相对视角**，同一人在不同镜头里会被标成不同位置，必须剥掉。

**只要通用词，不要组合词**（如「画面左侧」）。替换是逐词 `str.replace`，
通用词能覆盖全部组合：「画面左侧紫衣女子」剥掉「画面」「左侧」后得到「紫衣女子」。
写成固定组合反而会漏 —— 实测漏过「画面左侧」只留「画面」，又漏过「左」，
把同一个人拆成两个身份键。
"""

# 通用人称词：剥掉位置与外观特征后如果**只剩这些**，说明这个标签没有承载身份。
# 「人物」「背影人物」「女子」都不是身份 —— 一条片子里可以有两个"人物"，
# 拿它当名字会让用户以为要准备两张多视图。
_GENERIC_PERSON_WORDS = (
    "人物", "人影", "背影", "侧影", "剪影", "身姿", "人像",
    "男子", "女子", "男生", "女生", "男人", "女人", "男性", "女性",
    "男士", "女士", "小孩", "老人", "长者",
)

# 稳定的外观特征词。**这些才是人物身份的依据**：服装颜色在一段片子里不变，
# 而位置会变、性别是模型的推断（PRD 红线 5 禁止添加未提供的性别信息）。
_APPEARANCE_WORDS = (
    "红", "橙", "黄", "绿", "青", "蓝", "紫", "黑", "白", "灰", "粉", "棕", "褐",
    "西装", "衬衫", "外套", "大衣", "裙", "T恤", "毛衣", "运动服", "制服", "围裙",
    "长发", "短发", "卷发", "马尾", "光头", "眼镜",
)


def _find_color(text: str) -> str:
    """从标签里找颜色名，找不到返回空串。

    **取文本里最靠左的那个颜色**，不是"词表里排最前的那个"（实测踩过）：
    原来是按颜色词表的顺序找"哪个出现了"，于是「紫白色上衣女子」认成白色、
    「黑白色上衣女子」认成白色、「白紫拼色上衣男子」认成紫色 ——
    谁是主色完全取决于词表顺序，而不是这句话怎么写。

    两级匹配：先找完整色名（`深棕色`），再退回单字色（`深棕`→`棕色`）——
    模型的写法不统一，只认完整词会漏掉一半。同位置命中多个时取更长的那个。
    """
    best_index = -1
    best = ""
    for candidate in (*_COLORS, *_BARE_COLOR):
        index = text.find(candidate)
        if index < 0:
            continue
        if best_index < 0 or index < best_index or (
            index == best_index and len(candidate) > len(best)
        ):
            best_index = index
            best = _BARE_COLOR.get(candidate, candidate)
    return best


def canonical_person_key(label: str) -> str:
    """从人物标签里提取**身份键**，用于跨镜头合并同一个人。

    判定顺序：

    1. 剥掉位置词与装饰性细节（发型/配饰），留下**颜色 + 服装**作为键；
    2. 都没有时再剥通用人称词（``人物``/``背影``/``女子``…）；
    3. 仍有剩余（如真人名「王姐」「魏巍老师」）→ 用剩余部分做键；
    4. 什么都不剩 → **返回空串**，表示"这个标签不足以认定身份"。

    **为什么键里必须带服装**（实测踩过）：只用颜色当键会把两个人并成一个 ——

        「深棕短袖上衣盘发女子」与「棕色无袖上衣披发女子」
        → 两者颜色都归到"棕色" → 合并成一个人

    而画面里明明是两个人。服装类型（短袖/无袖/背心/连衣裙）在同一视频里
    通常不重复，加上它就能分开；同一人换机位时服装不变，也不会被拆开。

    **为什么只取一对「颜色 + 服装」**：模型的写法常常前后不一致
    （同一人一处写"深棕短袖上衣"、另一处写"棕色无袖上衣"），
    要求完全一致会把同一人拆成两个；而只靠颜色又会把两个人并成一个。
    取**同一片段里的那一对**是这两个错误之间的折中；
    一个人写了两件衣服时优先取上身（见 ``_appearance_pair``）。

    位置词绝不当身份：它随镜头变，也不是**人**的属性。
    返回空串是诚实的选择 —— 让上游如实登记"认不出是谁"，而不是造个假名字。
    """
    cleaned = _strip_position_words(label)
    color, garment = _appearance_pair(cleaned)
    if not (color or garment):
        color, garment = _appearance_pair(str(label))
    color = color or _find_color(cleaned) or _find_color(str(label))
    features: list[str] = []
    if color:
        features.append(color)
    if garment:
        features.append(garment)
    if features:
        return "+".join(features)
    for word in _GENERIC_PERSON_WORDS:
        cleaned = cleaned.replace(word, "")
    return cleaned.strip()


def dedupe_person_labels(labels: Sequence[str]) -> list[str]:
    """按身份键合并同一个人，返回去重后的规范名。

    同一身份出现多次时，选**出现次数最多**的标签；次数相同则选最短的
    （通常最干净，少带位置修饰）。

    两条纪律：

    1. 身份键为空的标签**直接丢弃**：纯位置词不是身份（见 ``canonical_person_key``）。
    2. 保留下来的名字里**不留位置词**：身份是"紫衣女子"，而"画面中央紫衣女子"
       里的"画面中央"是**这张画面**的属性。带着它，用户改机位后名字就自相矛盾，
       而且素材位会被写成「画面中央紫衣女子多视图」。
       剥掉位置词后如果几个身份撞名，就退回"人物1/人物2"（不编外观特征，红线 5）。
    """
    from collections import Counter

    groups: dict[str, Counter] = {}
    for label in labels:
        cleaned = _text(label)
        if not cleaned:
            continue
        if any(word in cleaned for word in _GROUP_WORDS):
            continue
        key = canonical_person_key(cleaned)
        if not key:
            continue
        groups.setdefault(key, Counter())[_strip_position_words(cleaned)] += 1

    names: list[str] = []
    for key in sorted(groups):          # 排序保证同一输入得到同一结果
        counter = groups[key]
        best = max(counter.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
        best = best.strip()
        if not best:
            best = key                     # 只剩身份键（如"紫"）也不能交白卷
        # 压成「颜色+服装+性别」—— 模型给的名字常拖着发型/配饰细节，
        # 用户要的是"一眼认出是谁"，不是外观清单（实测有过
        # 「白色衬衫女子（灰白色短卷发、玳瑁色墨镜）」这种名字）
        best = compact_person_name(best) or best
        names.append(best)

    # 撞名就按序号重命名 —— 宁可叫"人物1"，也不要两个角色同名让用户分不清
    if len(set(names)) != len(names):
        names = [f"人物{i}" for i in range(1, len(names) + 1)]
    return names


_COLOR_MODIFIERS = ("深灰", "深", "浅", "淡", "亮", "暗", "灰", "亮色", "暗色")
_GARMENT_CORE = (
    "连衣裙", "连帽外套", "运动外套", "短袖上衣", "长袖上衣", "无袖上衣", "运动上衣",
    "印花T恤", "T恤", "背心", "开衫", "外套", "衬衫", "西装", "卫衣", "毛衣",
    "运动装", "长裤", "短裤", "阔腿裤", "半身裙", "上衣", "衣服", "服装",
)


def _base_color(color: str) -> str:
    """把「深蓝色」「浅蓝色」「灰米色」这类**带强度修饰的色名**归到基础色。

    模型的写法前后不一致是常态：同一个人在 4 个镜头里被写成
    「深蓝色运动外套女子」「深蓝色上衣人物」「深灰蓝色上衣女子」——
    不归一就会被当成 3 个人。
    **「灰色」本身就是基础色**（实测踩过）：单字「灰」原来在修饰词表里，
    于是 ``_base_color("灰色") == ""`` —— 所有穿灰衣服的人身份键都成了**空色**，
    而一个压根没写颜色的「运动外套女子」键也是空色，
    两个不同的人会算成同一个键、被并成一个人。
    现在剥完修饰词什么都不剩时，原文里有「灰」就认成灰色。
    """
    text = color
    for modifier in _COLOR_MODIFIERS:
        text = text.replace(modifier, "")
    text = text.replace("色", "")
    if text:
        return text
    return "灰" if "灰" in color else ""


def _garment_core(garment: str) -> str:
    """服装归一：把「运动外套」「连帽外套」「短袖上衣」都收敛到「上衣/外套」这类**部位**。

    为什么按部位而不是按款式：款式写法最不稳定（同一件外套会被写成
    「运动外套」「连帽外套」「外套」），而"上身/下身/连体"这个粒度
    足以区分同一画面里的两个人，且对写法变化鲁棒。
    """
    if not garment:
        return ""
    if any(word in garment for word in ("连衣裙", "运动装", "服装", "衣服")):
        return "衣装"
    if any(word in garment for word in ("裤", "半身裙")):
        return "下装"
    return "上衣"


def _identity_key(label: str) -> str:
    """**归一到基础色 + 部位**的身份键。比 ``canonical_person_key`` 更宽松，
    专门用于"同一个人的不同写法"合并。
    """
    color = _find_color(_strip_position_words(label)) or _find_color(str(label))
    garment = ""
    cleaned = _strip_position_words(label)
    color, garment = _appearance_pair(cleaned)
    if not (color or garment):
        color, garment = _appearance_pair(str(label))
    # 成对片段里没带颜色时仍要**独立取一次色**（与 canonical_person_key 同一口径）。
    # 实测踩过：「白紫拼色上衣男子」的成对片段只能取到「上衣」（颜色被"拼色"隔开了），
    # 少了这一行它的身份键是「+上衣」、而压缩名是「紫色上衣男子」的键是「紫+上衣」
    # —— 同一个人算不出同一个键，别名表就是空的，位置证据整条链路失效。
    color = color or _find_color(cleaned) or _find_color(str(label))
    base_color = _base_color(color) if color else ""
    core = _garment_core(garment)
    if base_color or core:
        return f"{base_color}+{core}"
    return canonical_person_key(label)


def _split_extra_persons(
    names: list[str],
    expected: int,
) -> list[str]:
    """人物数超过"画面里同时最多几个人"时，把**写法相近的合并**。

    判据是画面证据，不是猜：视觉模型逐镜头给了 people_labels，
    一条片子里同时出现的人数上限（``expected``）就是画面的物理上限。
    识别出的人物数超过它，说明是同一个人的不同写法被拆开了 ——
    实测踩过：一条双人视频被拆成 4 个人
    （「米色上衣女子」「灰米色上衣棕裤女子」「深蓝色运动外套女子」「深蓝色上衣人物」），
    而画面里从头到尾只有 2 个人。

    合并只在**基础色 + 部位**相同、或其中一方特征更全时进行；
    合并后仍超上限就保留，宁可多列出让人删，也不静默丢掉。
    """
    if expected <= 0 or len(names) <= expected:
        return names

    groups: list[tuple[str, list[str]]] = []
    for name in names:
        key = _identity_key(name)
        for index, (group_key, members) in enumerate(groups):
            if group_key == key or (
                group_key and key
                and (group_key.startswith(key + "+") or key.startswith(group_key + "+"))
            ):
                members.append(name)
                break
        else:
            groups.append((key, [name]))

    merged: list[str] = []
    for _, members in groups:
        # 取**最短**的那个写法当名字（通常最干净），但优先带颜色的
        with_color = [m for m in members if _find_color(m)]
        pool = with_color or members
        merged.append(min(pool, key=len))
    return merged


def canonical_speaker(speaker: str, people: Sequence[str]) -> str:
    """把台词里的说话人收敛到**人物清单里的那个名字**。

    为什么要这一步：台词归属模型是按画面描述人（"蓝色运动T恤男子"），
    而人物清单是去重压缩后的名字（"蓝色T恤男子"）——
    两者写法不一致，用户在上面看到「人物1〔蓝色T恤男子〕」、
    下面台词却写「蓝色运动T恤男子说」，就对不上号了。

    匹配用**基础色 + 部位**（与人物合并同一套判据），
    匹配不上就原样返回 —— 宁可保留一个陌生名字，也不要硬塞给别人。
    """
    text = str(speaker or "").strip()
    if not text or not people:
        return text
    key = _identity_key(text)
    for name in people:
        if name == text:
            return name
    if key:
        for name in people:
            if _identity_key(name) == key:
                return name
    return text


def _strip_position_words(label: str) -> str:
    cleaned = label
    for word in _POSITION_WORDS:
        cleaned = cleaned.replace(word, "")
    return cleaned.strip()


# 服装类型词（用于压缩人物名）
_GARMENT_WORDS = (
    "连衣裙", "连帽外套", "运动外套", "短袖上衣", "长袖上衣", "无袖上衣", "运动上衣",
    "印花T恤", "T恤", "背心", "开衫", "外套", "衬衫", "西装", "卫衣", "毛衣",
    "运动装", "运动服", "长裤", "短裤", "阔腿裤", "半身裙", "上衣", "衣服", "服装",
)
_GARMENT_TOKENS = _GARMENT_WORDS + ("衣",)
"""词表 + 单字「衣」——「红衣者」「紫衣女子」这种简写，衣就是上衣。

**必须排在最后**：正则的候选按顺序匹配，短的放前面会把「上衣」「衣服」切碎。
"""
# 修长人物名时要丢掉的装饰性细节（发型、配饰）——它们对"区分是谁"没有帮助，
# 却让名字变得很难读（实测出现过「白色衬衫女子（灰白色短卷发、玳瑁色墨镜）」）
_DECOR_WORDS = (
    "长卷发", "短发", "长发", "卷发", "扎马尾", "马尾", "扎发髻", "发髻", "盘发",
    "丸子头", "扎丸子头", "配", "墨镜", "眼镜", "玳瑁色", "灰白色",
)
_COLORS = (
    "深蓝色", "浅蓝色", "深棕色", "浅棕色", "灰白色", "淡紫色", "酒红色",
    "深灰色", "浅灰色", "米白色",
    "白色", "黑色", "红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色",
    "粉色", "灰色", "棕色", "褐色", "米色", "银色", "金色",
)
"""颜色词表。**长词在前** —— 匹配取第一个命中的词，
所以"深棕色"必须排在"棕色"前面，否则「深棕短袖」会被归成"棕色"，
与同一视频里另一件"棕色无袖"撞在一起。"""


_BARE_COLOR = {
    "蓝": "蓝色", "紫": "紫色", "红": "红色", "绿": "绿色", "粉": "粉色",
    "灰": "灰色", "棕": "棕色", "黄": "黄色", "白": "白色", "黑": "黑色",
    "青": "青色", "橙": "橙色", "褐": "褐色", "米": "米色",
}

_MODIFIER_CHARS = "深灰浅淡亮暗"
"""颜色强度修饰字（「深灰」的「深」、「浅蓝」的「浅」）。

必须跟颜色一起匹配：只认颜色词的话，「深灰运动外套女子」里只会认出
「灰运动外套女子」，替换完正文里会剩下一个孤零零的「深」。
"""

_PERSON_SUFFIX_WORDS = (
    "男子", "男士", "女子", "女士", "人物", "老师", "先生", "太太", "姐", "哥", "者",
)

_PERSON_MENTION_RE = re.compile(
    f"[{_MODIFIER_CHARS}]*"
    "(?:" + "|".join(_COLORS) + "|" + "|".join(_BARE_COLOR) + ")?"
    "(?:" + "|".join(_GARMENT_TOKENS) + ")"
    "(?:" + "|".join(_PERSON_SUFFIX_WORDS) + ")?"
)
"""从自由文本里认出"提到某个人"的片段（强度前缀 + 颜色 + 服装 + 人称后缀）。

「红衣者」「浅紫色运动服者」这类**简写指代**也必须认出来 ——
模型在同一段描述里常常先写全名、后面就用简写，认不出就会
在正文里留下「红衣者」和「人物1」并存的场面（实测踩过）。
"""

_APPEARANCE_RE = re.compile(
    "(?P<color>" + "|".join(_COLORS) + "|" + "|".join(_BARE_COLOR) + ")?"
    "(?P<garment>" + "|".join(_GARMENT_TOKENS) + ")"
)


def _appearance_pair(text: str) -> tuple[str, str]:
    """取**成对**的「颜色 + 服装」—— 两者必须来自同一个片段。

    为什么必须成对（实测踩过）：模型的标签会一次写两件衣服。
    「红衣黑色短裤者」里独立取色会取到「黑色」（那是短裤的色），
    独立取服装按词表顺序会取到「短裤」，拼出来是「黑色短裤人物」——
    这个人的**红上衣**在交付物里彻底消失，用户认不出是谁。

    优先取**上身**：上衣颜色才是跨镜头认人的主要依据，
    下装（尤其黑短裤/黑长裤）常常两个人一样，拿它命名等于没命名。

    只剩颜色、没有服装时仍返回颜色（退回旧的兜底能力）——
    否则「紫色女子」这种写法会因为"配不成对"而整个丢掉身份。
    """
    lower: tuple[str, str] = ("", "")
    for match in _APPEARANCE_RE.finditer(text):
        garment = match.group("garment")
        if garment == "衣":
            garment = "衣服"          # 「红衣者」= 红色上衣
        # 单字色要补全：「蓝」→「蓝色」，否则会拼出「蓝衣服男子」
        color = _BARE_COLOR.get(match.group("color") or "", match.group("color") or "")
        if _garment_core(garment) != "下装":
            return color, garment
        if not any(lower):
            lower = (color, garment)
    if any(lower):
        return lower
    return _find_color(text), ""


def person_alias_map(
    observed: Sequence[str],
    people: Sequence[str],
    *,
    texts: Sequence[str] = (),
) -> dict[str, str]:
    """``模型写法 → 规范名``。只收**认得出是同一个身份**的写法。

    两个来源缺一不可：

    1. 模型逐镜头给的 ``people`` 清单（``observed``）；
    2. **自由文本本身**。实测踩过：镜头描述里写了「深灰运动外套女子」，
       而 people 清单里从头到尾没有这个写法 —— 只信清单就会漏掉它，
       正文里就会一边写「人物1」一边冒出「深灰运动外套女子」。
    """
    out: dict[str, str] = {}

    def consider(label: str) -> None:
        cleaned = _text(label)
        if not cleaned or cleaned in out:
            return
        canon = canonical_speaker(cleaned, people)
        if canon and canon != cleaned:
            out[cleaned] = canon

    for label in observed:
        consider(label)

    keys = {canonical_person_key(name) for name in people}
    keys.discard("")
    if keys:
        for text in texts:
            for match in _PERSON_MENTION_RE.finditer(str(text or "")):
                if canonical_person_key(match.group(0)) in keys:
                    consider(match.group(0))
    return out


def canonical_person_mentions(text: str, aliases: Mapping[str, str]) -> str:
    """把自由文本里的人称写法换成人物清单里的**规范名**。

    为什么必须做：视觉模型在同一个人的写法上前后不一致。实测同一条片子里
    同一个人被写成「灰色运动外套女子」和「深灰运动外套女子」，
    而 ``blocking`` 是自由文本 —— 不统一的话，正文里会一边写「人物1」、
    一边又冒出一个「深灰运动外套女子」，用户没法把它们对上号。

    **长写法先换**：短写法是长写法的子串时，先换长的那次才不会被切碎。
    判据与 ``canonical_speaker`` 同一套（基础色 + 服装部位），
    认不出来就原样保留 —— 宁可留一个陌生名字，也不要硬塞给别人。
    """
    if not text or not aliases:
        return text
    for label in sorted(aliases, key=len, reverse=True):
        if label in text:
            text = text.replace(label, aliases[label])
    return text


def compact_person_name(label: str) -> str:
    """把模型给的人物名压成「颜色+服装+性别」这种一眼能认的形式。

    用户要求的是「蓝色衣服男老师」那种命名 —— 关键信息是**颜色 + 服装**，
    因为它们在整条片子里不变，是跨镜头认人的唯一依据。

    实测模型给的名字往往拖着无用的修饰：
        白色衬衫女子（灰白色短卷发、玳瑁色墨镜）  → 白色衬衫女子
        紫色连衣裙长卷发女子                      → 紫色连衣裙女子
        蓝色T恤灰色短裤男子                       → 蓝色T恤男子
        灰色上衣深色长裤女子                      → 灰色上衣女子
        浅蓝运动装女子                            → 蓝色运动装女子（"浅蓝"归一到"蓝色"）

    压不出颜色或服装时**原样返回** —— 宁可名字长一点，也不要丢信息。
    """
    text = _strip_position_words(str(label or "")).replace("（", "(").split("(")[0].strip()
    if not text:
        return ""

    # 颜色与服装**成对**取（见 _appearance_pair）：
    # 一次写两件衣服时，「红衣黑色短裤者」必须取到「红衣」这一对，
    # 而不是各自独立取成「黑色」+「短裤」—— 那样这个人的红上衣就丢了。
    # 成对片段没带颜色时退回 _find_color（**与身份键同一口径**：
    # 判据分叉的话，同一个人会算不出同一个键，别名表直接失效）。
    color, garment = _appearance_pair(text)
    color = color or _find_color(text)

    if not (color or garment):
        return text

    gender = ""
    for word in ("男子", "女子", "男士", "女士", "男性", "女性", "男人", "女人", "老师"):
        if word in text:
            gender = word
            break
    if not gender:
        # 「者」这类中性结尾：按外观词里有没有明确的男女倾向来定，
        # 定不了就用"人物"（不编性别，红线 5）
        if "西装" in text:
            gender = "男子"
        elif "女" in text:
            gender = "女子"
        elif "男" in text:
            gender = "男子"
        else:
            gender = "人物"

    core = f"{color}{garment}" if garment else f"{color}衣服"
    return f"{core}{gender}"


def people_structure_note(ir: Mapping[str, Any], report: InferenceReport) -> str:
    """人物结构是"模型判断的"还是"代码推断的"，如实写进报告。

    用户 2026-10-08：「我下次又给个超出框架的视频，你又开始乱反推，怎么办？
    我总不能次次都和你说改，**一点自适应都没有吗**？」

    自适应来自两件事：

    1. **结构判断交给模型**（``report.semantic.people_structure``）——
       它见过画面，比我写的任何关键词表都更能判断"谁是主角、谁是群像"；
    2. **代码兜底时要如实说** —— 推断出来的清单可能不准，用户得知道
       这条不是"看清了"，而是"猜的"，并且知道一句话就能改。

    返回空串表示这次是模型判断的（不需要提醒）。
    """
    source = str((ir.get("meta") or {}).get("people_structure_source") or "")
    if source != "code":
        return ""
    observed: list[str] = []
    for understanding in report.understandings:
        observed.extend(understanding.people)
    names = dedupe_person_labels(observed)
    if len(names) < 2:
        return ""
    locked = len(ir.get("locked", {}).get("B2_person") or [])
    return (
        f"这次**没拿到模型对人物结构的判断**（它没答「谁是主角、谁是群像」），"
        f"人物清单是按画面标签推断的：画面里出现 {len(names)} 位不同的人，"
        f"锁定了 {locked} 位。如果锁错了人，用改写功能说一句就行"
        f"（例如「只锁主角，其他都算群像」）。"
    )


def unnamed_person_note(report: InferenceReport) -> str:
    """有检测到人、却一个身份都认不出来时的诚实说明。

    这条不能省：交付物里会因此**没有人物条目**，连带眨眼节拍与皮肤质感
    两条固定模块一起消失（它们挂在人物条目下面）。不说明的话，
    用户只会看到"少了东西"，不知道是自己要补还是系统坏了。
    """
    observed: list[str] = []
    for understanding in report.understandings:
        observed.extend(understanding.people)
    if not observed:
        return ""
    if dedupe_person_labels(observed):
        return ""
    labels = "、".join(dict.fromkeys(_text(x) for x in observed if _text(x)))[:60]
    return (
        f"画面里识别到人物（{labels}），但视觉模型只给了位置描述、没给稳定特征"
        f"（服装颜色/发型），因此**无法确定身份**。没有人物条目就没有多视图素材位、"
        f"也没有眨眼节拍与皮肤质感句。请补上角色名与外观描述。"
    )


def _fill_blocking(ir: dict, report: InferenceReport) -> None:
    """归纳出稳定站位，写进 LOCK 位。站位是**致命级**维度（错了人物关系错乱）。

    **两种口径分开写，不混成一个**：

    - ``composition``（构图偏侧）：整个画面的重心。双人一左一右 → 居中。
    - ``blocking``（逐人左右）：谁在左、谁在右。多人时才有意义。
    实测踩过：只写"人物始终在画面左侧"会丢掉"两人一左一右"这个真正要保的关系。

    **证据分两级，模型优先**：

    1. `_model_layout_sentences` —— 模型逐镜头说的位置（``红衣者位于画面偏左``）。
       带身份、能分辨"挡住的人"与"屏幕里的人"；
    2. `_cv_layout_sentences` —— CV 检测框量测（``检测框中心 x≈0.29``）。
       只在模型一个位置都没给时兜底 —— 框没有身份，也会把大屏/倒影里的人算进来
       （实测：演播厅单人示范片被判成「有角色位于画面右侧 x≈0.816」，
       而那个框是背景屏上的照片）。
    """
    parts = _model_layout_sentences(ir, report) or _cv_layout_sentences(report)

    # 人群规模：1 人还是 2 人在画面里，也是"复刻像不像"的一部分。
    # **不能只在有检测框时才写** —— 检出数只是下限，模型逐镜头的人数同样是证据。
    ensemble = ir["locked"].get("B2b_ensemble")
    if isinstance(ensemble, Mapping) and (ensemble.get("label") or ensemble.get("look")):
        # **有群体时不写"固定 N 位人物出镜，不做增减"**（用户 2026-10-07）：
        # 学员是交给生成模型自己生成的，锁死人数既做不到、也和群体句自相矛盾。
        arrangement = "主体人物固定出镜、不做增减，其余为群体角色按画面需要自然生成"
    else:
        arrangement = _arrangement_sentence(report)
    if not parts and not arrangement:
        ir["locked"]["B3_blocking"]["positions"] = []
        return
    if arrangement:
        parts.append(arrangement)
    parts.append("机位与人物相对位置全程禁止变动")
    ir["locked"]["B3_blocking"]["positions"] = parts


def _model_layout_sentences(ir: Mapping[str, Any], report: InferenceReport) -> list[str]:
    """由**模型逐镜头说的位置**归纳构图与左右；模型没给位置就返回空。"""
    people = [str(p.get("name") or "").strip() for p in ir["locked"]["B2_person"]]
    positions = _model_positions(report, [name for name in people if name])
    if not positions:
        return []

    sides = {_side_of_position(text) for text in positions.values()}
    sides.discard("")
    if not sides:
        return []

    parts: list[str] = []
    # 所有人都说同一侧 → 构图就是那一侧；否则（含中央/左右都有）按"居中"说。
    parts.append(_layout_sentence(sides.pop() if len(sides) == 1 else "center"))
    for side, label in (("left", "左侧"), ("right", "右侧")):
        if side in _sides_of(positions.values()):
            parts.append(f"有角色位于画面{label}")
    return parts


def _sides_of(positions: Iterable[str]) -> set[str]:
    return {_side_of_position(text) for text in positions} - {""}


def _side_of_position(text: str) -> str:
    """位置文本（``画面右后方``）→ left/right/center。认不出返回空串。"""
    if "左" in text:
        return "left"
    if "右" in text:
        return "right"
    if "中" in text:
        return "center"
    return ""


def _cv_layout_sentences(report: InferenceReport) -> list[str]:
    """检测框量测出的构图与左右（**兜底**）。

    正文里的 ``检测框中心 x≈0.29`` 是几何量测，可追溯；但它**没有身份**，
    也分不清真人、倒影和屏幕上的照片 —— 所以只在模型没给位置时用。
    """
    per_shot = [s.blocking for s in report.shots if s.blocking]
    if not per_shot:
        return []
    parts: list[str] = []

    def dominant(side: str) -> tuple[int, float]:
        hits = [b.x_ratio for shot in per_shot for b in shot if b.side == side]
        if not hits:
            return 0, 0.0
        return len(hits), round(sum(hits) / len(hits), 3)

    threshold = max(1, int(len(per_shot) * 0.6))

    # 一、构图：整个画面的重心偏哪一侧
    compositions = [s.composition for s in report.shots if s.composition]
    if compositions:
        counts: dict[str, int] = {}
        for side in compositions:
            counts[side] = counts.get(side, 0) + 1
        layout = max(counts, key=lambda k: counts[k])
        if counts[layout] >= threshold:
            parts.append(_layout_sentence(layout))

    # 二、逐人左右
    # **不写「检测框中心 x≈0.29」这类内部量测标记**（用户 2026-10-07 明确要求）：
    # 那是我们自己的坐标，生成平台看不懂；量测值留在 provenance 里可追溯即可。
    for side, label in (("left", "左侧"), ("right", "右侧")):
        count, _average = dominant(side)
        if count >= threshold:
            parts.append(f"有角色位于画面{label}")

    if not parts:
        # 既没有稳定构图也没有稳定左右：单人成片最常见。
        # （PRD 把站位列为致命级维度，不能因为"不左不右"就丢掉。）
        count, _average = dominant("center")
        if count and count >= threshold:
            parts.append("人物始终位于画面中央")
    return parts


def _arrangement_sentence(report: InferenceReport) -> str:
    """按镜头里的人数归纳"画面上有几个人"。

    只在**多数镜头人数一致**时才写 —— 人数本身在变的片子（有人入画出画），
    写一句"全程 2 人"就是编造。

    **以模型逐镜头看到的为准，CV 检出数兜底**（两个方向的实测反例都有）：

    - CV **少检**：双人片子里后排的人被前排挡住，YOLO 4 个镜头里 3 个只检出
      1 个框 → 写出「画面中固定 1 位人物出镜」，与人物清单自相矛盾；
    - CV **多检**：演播厅单人示范片的背景大屏上印着 4 张人物照片，YOLO 把
      屏幕里的人也算成真人（检出 3）→ 写出「画面中固定 3 位人物出镜」，
      而画面里明明只有 1 个人（模型逐镜头都说 1 人，还注明了那是屏上画面）。

    两个反例里模型都是对的：它能分辨"挡住的人"和"屏幕里的人"。
    CV 只在模型一个身份都给不出时兜底。

    **人数在各镜头不一样时必须说清楚**（自检回路实测抓到的真缺陷）：
    某双人片子的镜头1 只有 1 个人（另一位还没入画），多数镜头是 2 人，
    于是写出了全局的「画面中固定 2 位人物出镜，不做增减」——
    与镜头1 的画面事实矛盾，而这一句是**永不可裁**的强约束。
    现在把逐镜头差异一并写出来，模型才不会被一个错误的全局数字误导。
    """
    per_shot = [len(u.people) for u in report.understandings if u.people]
    counts = _majority_count(per_shot)
    if not counts:
        counts = _majority_count([s.people_count for s in report.shots])
    if not counts:
        return ""
    sentence = f"画面中固定 {counts} 位人物出镜，不做增减"
    # 逐镜头差异只按**模型**逐镜头看到的说（CV 检出数会漏检被挡住的人）；
    # 模型一个身份都没给出时不做差异描述，避免拿检测框的错数去"纠正"整句。
    odd = (
        [
            (u.shot_index, len(u.people))
            for u in report.understandings
            if u.people and len(u.people) != counts
        ]
        if per_shot
        else []
    )
    if odd:
        detail = "、".join(f"镜头{index} 为 {count} 位" for index, count in odd[:4])
        sentence = f"{sentence}（{detail}，其余镜头 {counts} 位）"
    return sentence


def _majority_count(counts: Sequence[int]) -> int:
    """在 ≥60% 的镜头里都一致的那个人数；没有一致值就返回 0（不写）。"""
    values = [c for c in counts if c > 0]
    if not values:
        return 0
    common = max(set(values), key=values.count)
    if values.count(common) < max(1, int(len(values) * 0.6)):
        return 0
    return common


_LAYOUT_TEXT = {
    "center": "人物位于画面中央、构图居中（左右两侧留白基本对称）",
    "left": "人物整体偏画面左侧（构图重心在左）",
    "right": "人物整体偏画面右侧（构图重心在右）",
}


def _layout_sentence(layout: str) -> str:
    return _LAYOUT_TEXT.get(layout, f"人物构图偏侧：{layout}")


_CROWD_WORDS = {
    "学员": "学员们",
    "观众": "观众们",
    "听众": "听众们",
    "队员": "队员们",
    "粉丝": "粉丝们",
    "顾客": "顾客们",
    "会员": "会员们",
    "人群": "人群",
    "群众": "群众",
    "大家": "大家",
    "其他人": "其他人物",
}
"""人群词 → 群体称呼。**观众与学员都算群体**（用户 2026-10-07 两次强调）。

第一次（带练视频）：「其他学员只需要给她们定位成**学员们**，然后穿着、年龄定位好」；
第二次（舞台互动）：「台下的**观众们**是 AI 生成的群像，那么多人都定义为观众们，
只需要定义年龄和穿着即可」。

"跟随"与"旁观"是两种行为，不能混（把观众写成"跟着老师做同一套动作"就错了）：

- ``_CROWD_FOLLOWS``：跟着主体做（学员/队员）；
- 其余（观众/听众/粉丝/顾客…）：**在旁边看**。
"""

_CROWD_FOLLOWS = ("学员", "队员")


def _crowd_label(text: str) -> str:
    """从描述里认出这是一群什么人（认不出返回空串）。"""
    for word, label in _CROWD_WORDS.items():
        if word in text:
            return label
    return ""


def _video_subject(report: InferenceReport) -> str:
    """认出**全片的主角**。

    不能简单取"各镜头 subject 的众数"（实测踩过，用户 2026-10-07 报的）：
    舞台互动那条片子里，镜1 的主角是**台上讲话的男老师**，
    镜2 是观众席特写、模型把**举手机的观众**当成了那一镜的主角 ——
    两者各出现一次，取众数就会随机挑一个（那次的实现用 ``set`` 去重，
    结果挑中了观众，用户看到的就是"老师被认成深蓝色上衣女子"）。

    判据按优先级：

    1. **说话人**（有台词时，说话的人就是主角，这是最硬的证据）；
    2. 出现在**最多镜头**里的 subject；
    3. 平票时取**最早镜头**的那个 —— 开场镜头给主角，是稳定的影视惯例。
    """
    subjects = [
        (u.shot_index, str(u.subject or "").strip())
        for u in report.understandings
        if str(u.subject or "").strip()
    ]
    if not subjects:
        return ""

    speakers = [
        str(segment.speaker or "").strip()
        for segment in report.transcript
        if str(segment.speaker or "").strip()
    ]
    for _index, name in sorted(subjects, key=lambda item: item[0]):
        key = _identity_key(name)
        if any(_identity_key(speaker) == key for speaker in speakers):
            return name

    counts: dict[str, list[int]] = {}
    for index, name in subjects:
        counts.setdefault(_identity_key(name), []).append(index)
    best_key = min(
        counts,
        key=lambda key: (-len(counts[key]), min(counts[key])),
    )
    return next(name for _index, name in sorted(subjects) if _identity_key(name) == best_key)


def _ensemble_from_report(
    report: InferenceReport,
    deduped: Sequence[str],
) -> tuple[str, dict]:
    """认出**主角**与**群体**，返回 ``(主体名, 群体条目或{})``。

    判据全部基于画面事实：

    1. 主角见 :func:`_video_subject`；
    2. **触发群体化的条件**：画面事实里出现人群词（观众/学员/队员/听众/粉丝…），
       或者单镜头里同时 ≥5 个人（真正的"人群"）。
       ⚠️ 原来这里额外要求"必须是动作类片子"——**那是我的错误假设**：
       舞台互动的观众同样是群像（用户 2026-10-07 第二次指出）。已去掉。
    3. 群体条目 = 各镜头 ``group`` 里最长的那条（年龄 + 穿着概况，一句话）；
    4. **不编**：没有人群线索就返回空群体，保持逐人清单（宁可不合并也不乱合并）。
    """
    descriptions = " ".join(
        str(u.blocking or "") + str(u.description or "") + str(u.group or "")
        for u in report.understandings
    )
    label = _crowd_label(descriptions)
    ceiling = max((len(u.people) for u in report.understandings), default=0)
    if not label:
        if ceiling < 5:
            return _video_subject(report), {}
        label = "人群"

    groups = [
        str(u.group or "").strip()
        for u in report.understandings
        if str(u.group or "").strip()
    ]
    look = max(groups, key=len) if groups else ""
    subject = _video_subject(report)
    members = [name for name in deduped if name != subject]
    if not subject and not look:
        return "", {}
    follows = "人物1" if subject and any(w in descriptions for w in _CROWD_FOLLOWS) else ""
    return subject, {
        "label": label,
        "look": look,
        "follows": follows,
        # 不做动作的人群（观众）由渲染层写成"在旁边看"
        "mode": "follow" if follows else "watch",
        "members": members,
    }


def disambiguate_display_names(people: Sequence[str]) -> list[str]:
    """给每个人一条**互不相同的**显示名。

    为什么必须有（用户 2026-10-07 那个 bug 的另一半）：显示名是
    「颜色+服装+性别」压出来的，**两个人可能压成同一个**；一旦同名，
    别名映射（name → 人物N）就会把后来那条的引用指到第一个人身上 ——
    实测出现过「人物2〔白色T恤女子〕：**人物1**形象参考[图2]」。

    规则：撞名的那几个退回**完整名**（带发型等区分信息）；完整名还撞，
    就退回「人物N」（不编外观特征，红线 5）。
    """
    displays = [compact_person_name(name) or name for name in people]
    duplicated = {d for d in displays if displays.count(d) > 1}
    if not duplicated:
        return displays
    out: list[str] = []
    used: set[str] = set()
    for index, (name, display) in enumerate(zip(people, displays), start=1):
        candidate = name if display in duplicated else display
        if candidate in used:
            candidate = f"人物{index}"
        used.add(candidate)
        out.append(candidate)
    return out


MAX_REFERENCE_PERSONS = 2
"""**需要"人物参考"的人物上限 —— 2 个**（用户 2026-10-07 反复强调的原话）：

> 「很多时候，**需要使用人物参考提示词的人物一般都不会超过 2 个**，很多其他的人物都是群像！
>  你理不理解群像 —— 难道出现十个人物，我都要全部固定人物形象吗？
>  我**只要固定主体人物的形象**就行了！其他的给 Seedance 自己生成就好了，
>  你只需要给他定位一下年龄和穿着。」

所以这是一条**硬规则**，不是启发式：超过 2 个的人一律进群像。
以前靠"人群关键词命中"来决定要不要折叠 —— 模型换个说法（"其余数人"/"背景中多名女性"）
就失效，于是又变成一堆人物参考（用户第三次因此发火）。
"""

_PRINCIPAL_HINTS = ("说话人", "主角", "主体")


def _match_person(label: str, names: Sequence[str]) -> str:
    """把模型给的一个写法（subject / 说话人）对应到人物清单里的**那个人**。

    为什么不能只按身份键查表（实测踩过，用户 2026-10-07 第三次发火的直接原因）：
    ``_identity_key`` 是"宽松键"，**会撞**——

        「紫色短袖上衣女子」→ 紫+上衣
        「紫色背心女子」    → 紫+上衣      ← 同键！

    于是"按身份键查 dict"时后者覆盖了前者，``subject`` 被对应到了**另一个女人**身上，
    真正的主角被换掉。所以：

    1. 先按**最长公共子串**认（模型写法常常只差一两个修饰词：
       「浅紫色短袖上衣女子」 vs 「紫色短袖上衣女子」 → 公共串 7 个字）；
    2. 只有在键**唯一**时才用身份键兜底 —— 撞键时宁可返回空（不认），
       也不要把人认错：认错的代价是"主角变成别人"，比留空严重得多。
    """
    # 先比**去掉通用人称词**之后的部分（「丁女子」与「紫色长裙女子」的公共串只有
    # 「女子」两个字，那是通用后缀，不能当作同一个人 —— 实测踩过：
    # 这样会把主角认成清单里任意一个「…女子」）。
    def distinctive(text: str) -> str:
        cleaned = str(text)
        for word in _GENERIC_PERSON_WORDS:
            cleaned = cleaned.replace(word, "")
        return cleaned.strip()

    target = distinctive(label)
    best, best_score = "", 0
    for name in names:
        candidate = distinctive(name)
        if not target or not candidate:
            continue
        score = _common_run(target, candidate)
        if score >= 2 and score > best_score:
            best, best_score = name, score
    if best:
        return best
    # 再比整串，但门槛更高（≥3），避免"…女子"这种通用后缀把人认错
    for name in names:
        score = _common_run(str(label), name)
        if score >= 3 and score > best_score:
            best, best_score = name, score
    if best:
        return best
    key = _identity_key(label)
    same = [name for name in names if _identity_key(name) == key]
    return same[0] if len(same) == 1 else ""


def _common_run(a: str, b: str) -> int:
    """最长公共**连续**子串长度（认人用；写法常只差几个修饰词）。"""
    if not a or not b:
        return 0
    best = 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def _prefer_specific(model_label: str, matched: str) -> str:
    """名字用**更具体**的那个写法。

    实测：视觉模型给的是「粉色上衣**人物**」，而结构归纳给的是
    「粉色上衣**女子**」—— 后者读起来才像个人。模型的说法更具体时就用它的。

    ``matched`` 为空（**模型答的名字在画面标签里对不上**）时返回空串 ——
    那是"它看错了"，绝不能用它的写法当人名（否则会凭空造出一个画面里没有的人）。
    """
    if not matched:
        return ""
    generic = ("人物", "人影", "背影", "侧影", "剪影", "身姿", "人像")
    if any(word in matched for word in generic):
        return compact_person_name(model_label) or matched
    return matched


def _from_model_structure(
    report: InferenceReport,
    deduped: Sequence[str],
) -> tuple[list[str], dict] | None:
    """**优先用模型给的人物结构**（用户 2026-10-08：「我做的不是 agent 吗？」）。

    模型（多模态结构归纳那一次调用）会回答：主角是谁、第二位要锁定的是谁、
    哪些人各自有独立身份、其余那些人作为群像的称呼/年龄穿着/行为。
    这里只做**校验**，不重猜：

    - 名字必须能在画面标签里对上（对不上就当作模型看错了，整块不用）；
    - 锁定人数仍受 ``MAX_REFERENCE_PERSONS`` 约束；
    - **说话人必须在人物清单里**（他要开口，不能被塞进群像）。

    返回 ``None`` 表示"模型没给/给的东西对不上"→ 调用方退回代码兜底。
    """
    structure = dict(getattr(report.semantic, "people_structure", {}) or {}) if report.semantic else {}
    if not structure:
        return None

    lead_label = str(structure.get("lead") or "")
    second_label = str(structure.get("second") or "")
    lead = _prefer_specific(lead_label, _match_person(lead_label, deduped))
    second = _prefer_specific(second_label, _match_person(second_label, deduped))
    distinct = [
        name for name in (
            _match_person(str(item), deduped) for item in (structure.get("distinct_others") or [])
        ) if name
    ]
    locked = [name for name in (lead, second) if name]
    for name in distinct:
        if len(locked) >= MAX_REFERENCE_PERSONS:
            break
        if name not in locked:
            locked.append(name)

    # 说话人一个都不能少（哪怕模型没把他列进主角/第二位）
    for segment in report.transcript:
        speaker = str(segment.speaker or "").strip()
        if not speaker:
            continue
        name = _match_person(speaker, deduped)
        if name and name not in locked:
            locked.append(name)

    if not locked:
        # 模型只答了群像、没答主角（或名字都对不上）：退回代码兜底
        return None

    members = [name for name in deduped if name not in locked]
    crowd_label = str(structure.get("crowd_label") or "").strip()
    crowd_look = str(structure.get("crowd_look") or "").strip()
    behavior = str(structure.get("crowd_behavior") or "").strip()
    if not members and not crowd_look:
        return locked, {}
    ensemble = {
        "label": crowd_label or "背景人物",
        "look": crowd_look or _appearance_summary(members),
        "follows": "人物1" if behavior == "follow" and locked else "",
        "mode": "follow" if behavior == "follow" and locked else "watch",
        # 群像被折叠了谁，用来记账（落 IR 前会删掉）
        "members": members,
    }
    if behavior == "own_action":
        # 各做各的（后景学员在练瑜伽）：行为写在照顾不到的地方，交给节拍逐镜描述
        ensemble["behavior_note"] = "own_action"
    return locked, ensemble


def _principal_persons(
    report: InferenceReport,
    deduped: Sequence[str],
) -> tuple[list[str], dict]:
    """选出**真正需要人物参考的人**（≤2），其余全部归为群像。

    判据（都基于画面/台词事实）：

    1. **说话人必须是人**（他要开口说话，必须锁定形象 + 绑音色）；
    2. **当过主角的人**（在某个镜头里是 ``subject``）按出现镜头数排序；
    3. 到这里还不够 2 个，就从剩下的人里按"身份键稳定度"补 —— 但**最多 2 个**；
    4. **其余全部进群像**：只留年龄与穿着（用户要的就是这个）。

    返回 ``(人物名列表, 群像条目或{})``。
    """
    speakers = [
        str(segment.speaker or "").strip()
        for segment in report.transcript
        if str(segment.speaker or "").strip()
    ]
    if not deduped:
        # 视觉标签一个都认不出身份（纯位置词被过滤掉了），但**说话人还在** ——
        # 那是"谁在说"的直接证据，不能因为画面认不出人就把他删掉。
        unique: list[str] = []
        for speaker in speakers:
            if speaker not in unique:
                unique.append(speaker)
        return unique[: max(MAX_REFERENCE_PERSONS, len(unique))], {}

    speakers = [
        str(segment.speaker or "").strip()
        for segment in report.transcript
        if str(segment.speaker or "").strip()
    ]
    # ① 说话人：**一个都不能少**（他们要开口，必须锁定形象并绑音色）。
    #    注意：ASR 给的说话人可能**不在视觉标签里**（真人名「王姐」），
    #    那时直接用说话人的写法当人名 —— 这是"谁在说"的直接证据，不能丢。
    speakers = [
        str(segment.speaker or "").strip()
        for segment in report.transcript
        if str(segment.speaker or "").strip()
    ]
    must: list[str] = []
    for speaker in speakers:
        name = _match_person(speaker, deduped) or speaker
        if name not in must:
            must.append(name)

    # ② **只认一个主角**：当过主角次数最多、平票取最早镜头的那位。
    #    为什么不是"每个镜头的主角都算"：实测那条舞台互动里，
    #    镜2 是观众席特写、模型把举手机的观众当成了那一镜的主角 ——
    #    若把各镜主角都收进来，观众又变成"人物2"（用户第三次因此发火）。
    subject_hits: dict[str, list[int]] = {}
    for understanding in report.understandings:
        label = str(understanding.subject or "").strip()
        if not label:
            continue
        name = _match_person(label, deduped)
        if name:
            subject_hits.setdefault(name, []).append(int(understanding.shot_index or 0))
    optional: list[str] = []
    if subject_hits:
        best = sorted(
            subject_hits, key=lambda k: (-len(subject_hits[k]), min(subject_hits[k]))
        )[0]
        optional.append(best)

    # ③ **画面里本来就没几个人**（≤2）时，剩下的也是真人（双人对话/双人采访），
    #    否则不去"凑满第二个名额" —— 用户要的是"只固定主体"，
    #    凑数会把背景里某个人莫名变成人物2（实测就是这样又给了一个多余的人物参考）。
    if len(deduped) <= MAX_REFERENCE_PERSONS:
        for name in deduped:
            if name not in must and name not in optional:
                optional.append(name)

    # 上限是 2；但**说话人不受上限约束** —— 三个人都说话就必须都锁定，
    # 否则没被锁定的那位台词会找不到说话人（G6 直接不过）。
    limit = max(MAX_REFERENCE_PERSONS, len(must))
    kept = (must + [name for name in optional if name not in must])[:limit]
    leftovers = [name for name in deduped if name not in kept]
    if not leftovers:
        return kept, {}
    return kept, _ensemble_from(report, leftovers)


def _ensemble_from(report: InferenceReport, members: Sequence[str]) -> dict:
    """把"其余的人"写成一个群体条目（称呼 + 年龄穿着）。

    **只要画面里还有别人就生成** —— 不再等关键词命中（实测踩过：
    模型写「其余数人」「背景中多名女性」时关键词一个都不中，于是 4 个人都成了人物参考）。
    """
    descriptions = " ".join(
        str(u.blocking or "") + str(u.description or "") + str(u.group or "")
        for u in report.understandings
    )
    label = _crowd_label(descriptions) or "背景人物"
    follows = "人物1" if any(w in descriptions for w in _CROWD_FOLLOWS) else ""

    # 年龄与穿着：模型给的 group 文本最准；没有就用被折叠人物的标签合成
    groups = [
        str(u.group or "").strip() for u in report.understandings if str(u.group or "").strip()
    ]
    look = max(groups, key=len) if groups else ""
    if not look:
        look = _appearance_summary(list(members))
    return {
        "label": label,
        "look": look,
        "follows": follows,
        "mode": "follow" if follows else "watch",
        "members": list(members),
    }


def _appearance_summary(members: Sequence[str]) -> str:
    """用人物标签合成一句"年龄+穿着"（合成不出来就留空，不编）。

    只做**已有信息的归纳**：把「粉色背心女子」「黑色上衣女子」这类标签
    写成「穿各色瑜伽背心与上衣的多位女性」不需要编造任何没看到的东西。
    """
    if not members:
        return ""
    colors: list[str] = []
    garments: list[str] = []
    gender = ""
    for name in members:
        color, garment = _appearance_pair(str(name))
        color = color or _find_color(str(name))
        if color and color not in colors:
            colors.append(color)
        if garment and garment not in garments:
            garments.append(garment)
        if not gender:
            for word in ("女子", "男子", "女士", "男士", "女性", "男性"):
                if word in str(name):
                    gender = word
                    break
    parts: list[str] = []
    if len(colors) > 1:
        parts.append("穿各色" + "、".join(garments[:3]) if garments else "穿各色服装")
    elif garments:
        parts.append("穿" + "、".join(garments[:3]))
    if gender:
        parts.append(gender)
    return "，".join(parts)


def _fill_persons(ir: dict, report: InferenceReport) -> None:
    """人物清单：**只留真正需要人物参考的人（≤2）**，其余写成群像。

    用户 2026-10-07 反复强调的规则（第三次因为同一件事发火）：

    > 「需要使用人物参考提示词的人物**一般都不会超过 2 个**，很多其他的人物都是群像！
    >  难道出现十个人物，我都要全部固定人物形象吗？我只要固定主体人物的形象就行了！
    >  其他的给 Seedance 自己生成就好了，你只需要给他定位一下年龄和穿着。」

    所以选人只认三条硬证据：**说话人** → **当过主角的人（按镜头数）** → 补足到 2 个；
    剩下的一律进群像（年龄 + 穿着，一句话）。判定见 :func:`_principal_persons`。
    """
    observed: list[str] = []
    for understanding in report.understandings:
        observed.extend(understanding.people)
    deduped = dedupe_person_labels(observed)

    # ---- 人物结构：**先听模型的**，它给不出才由代码兜底 ----
    # 用户 2026-10-08：「我做的不是 agent 吗？画面理解呢？」——
    # 三次"一堆人物参考"都是代码猜结构猜错的，所以结构判断归模型。
    structured = _from_model_structure(report, deduped)
    if structured is not None:
        selected, ensemble = structured
        ir["meta"]["people_structure_source"] = "model"
    else:
        selected, ensemble = _principal_persons(report, deduped)
        ir["meta"]["people_structure_source"] = "code"
    if ensemble:
        ensemble.pop("members", None)
        ensemble.pop("behavior_note", None)
        ir["locked"]["B2b_ensemble"] = ensemble
    people = list(selected)

    blink = BLINK_BY_GENRE.get(
        report.semantic.genre if report.semantic else Genre.DIALOGUE,
        BlinkBeat.FIVE,
    )
    if not people:
        ir["locked"]["B2_person"] = []
        return

    # 人物可能带画面位置：**模型逐镜头说的位置优先**，CV 检测框兜底。
    positions = _person_positions(report, people)
    displays = disambiguate_display_names(people)
    ir["locked"]["B2_person"] = [
        {
            "name": name,
            "display_name": displays[index],
            "position": positions[index] if index < len(positions) else "",
            "multi_view_ref": f"{name}多视图",
            "lock_strength": "standard",
            "wardrobe": "",
            "skin": "brief",
            "blink": blink.value,
        }
        for index, name in enumerate(people)
    ]


_SIDE_TEXT = {"left": "画面左侧", "right": "画面右侧", "center": "画面中央"}

_SIDE_HINTS = (("左", "画面左侧"), ("右", "画面右侧"))
"""模型说法 → 左右。**长词在前**由 ``_position_text`` 的顺序保证。"""

_CENTER_HINTS = ("中央", "中间", "居中")
_DEPTH_HINTS = (("后", "后"), ("背", "后"), ("前", "前"))


def _position_text(where: str) -> str:
    """把模型说的一小段位置话（「画面偏左前方」「其右后方」）收敛成位置文本。

    认不出左右就返回空串 —— 宁可这一项留空，也不要硬塞一个位置。
    能认出前后时一并带上（「画面右后方」）：前景/后景关系是复刻时
    真正容易做错的东西，而模型逐镜头本来就说了。
    """
    hint = where or ""
    if any(word in hint for word in _CENTER_HINTS) and not any(
        key in hint for key, _ in _SIDE_HINTS
    ):
        return "画面中央"
    side = next((text for key, text in _SIDE_HINTS if key in hint), "")
    if not side:
        return ""
    depth = next((text for key, text in _DEPTH_HINTS if key in hint), "")
    return f"{side[:-1]}{depth}方" if depth else side


def _model_positions(report: InferenceReport, people: Sequence[str]) -> dict[str, str]:
    """**模型逐镜头说的位置** → 逐人位置文本。

    为什么优先用模型的话（实测踩过）：CV 检测框**没有身份**，只能把
    "最左的那个框"按序号配给"第一个人"。一条双人片子里后面那位被前面
    那位挡住，YOLO 只检出 1 个框，于是「画面左侧」被安到了错误的人头上 ——
    画面里明明是男的在左前、女的在右后，交付物却写成「紫色运动服人物 · 画面左侧」。
    而模型逐镜头本来就写清了「红衣者位于画面偏左，浅紫色运动服者位于其右后方」，
    那是**带身份**的位置证据，比框的序号可靠得多。

    判据与站位一致：同一个人被提到的镜头里，多数（≥60%）说同一侧才采用。
    """
    from collections import Counter

    aliases = person_alias_map(
        [label for understanding in report.understandings for label in understanding.people],
        people,
        texts=[understanding.blocking for understanding in report.understandings],
    )

    votes: dict[str, Counter[str]] = {}
    for understanding in report.understandings:
        text = canonical_person_mentions(understanding.blocking or "", aliases)
        for person in people:
            if not person:
                continue
            # 模型的说法不止「位于」一种：实测出现过
            # 「白紫拼色上衣男子站立于演播室地面中央偏左位置」——
            # 只认「位于」会把整条片子的位置证据全丢掉，退回没有身份的检测框。
            pattern = re.compile(
                re.escape(person) + r"(?:站立|站|坐|蹲|卧|躺|位|处|身|出现在)?于([^，。；、]{1,12})"
            )
            for match in pattern.finditer(text):
                hint = _position_text(match.group(1))
                if hint:
                    votes.setdefault(person, Counter())[hint] += 1

    out: dict[str, str] = {}
    for person, counter in votes.items():
        hint, hits = counter.most_common(1)[0]
        if hits >= max(1, int(sum(counter.values()) * 0.6)):
            out[person] = hint
    return out


def _person_positions(report: InferenceReport, people: Sequence[str]) -> list[str]:
    """逐人位置：**先用模型逐镜头说的，CV 检测框兜底**。

    ``people`` 是人物清单里的规范名（顺序即人物1/人物2…）。
    """
    by_model = _model_positions(report, people)
    by_cv = _cv_positions(report, len(people)) if _counts_agree(report) else [""] * len(people)
    return [
        by_model.get(name) or (by_cv[index] if index < len(by_cv) else "")
        for index, name in enumerate(people)
    ]


def _counts_agree(report: InferenceReport) -> bool:
    """检测框数出来的多数人数，与模型看到的人数**一致**吗？

    不一致就说明检测框这一路不可信 —— 它会**漏检**被挡住的人（双人片子里
    后排那位），也会**多检**屏幕/倒影/照片里的人（演播厅大屏上的人物图）。
    这种情况下按"序号 → 人物"配位置就是把位置安到错误的人头上，
    所以宁可不给这个字段（红线 7：不编）。
    """
    model_count = _majority_count([len(u.people) for u in report.understandings])
    cv_count = _majority_count([s.people_count for s in report.shots])
    return bool(model_count) and model_count == cv_count


def _cv_positions(report: InferenceReport, count: int) -> list[str]:
    """CV 检测框量出来的位置（**兜底**，模型没说位置时才用）。

    诚实边界：检测框没有身份，"第 1 个人在左、第 2 个人在右"只保证数量级正确，
    不保证与人物名一一对应 —— 所以它现在只当兜底，不再当主判据。
    """
    from collections import Counter

    per_shot: list[list[float]] = []
    for signal in report.shots:
        xs = sorted(b.x_ratio for b in signal.blocking)
        if xs:
            per_shot.append(xs)
    if not per_shot:
        return [""] * count

    sizes = Counter(len(xs) for xs in per_shot)
    common_size, common_count = sizes.most_common(1)[0]
    if common_count < max(1, int(len(per_shot) * 0.6)):
        return [""] * count

    if common_size == 1:
        avg = sum(xs[0] for xs in per_shot if len(xs) == 1) / common_count
        return [_SIDE_TEXT[side_of_ratio(avg)]] + [""] * (count - 1)

    # 多人：用**最左框的均值**与**最右框的均值**当左右锚点，
    # 不要按 slot 序号取平均 —— 实测镜4/5 只检出中央一个大框，
    # 按 slot 算会把右侧的人算成"画面中央"。
    lefts = [xs[0] for xs in per_shot if len(xs) == common_size]
    rights = [xs[-1] for xs in per_shot if len(xs) == common_size]
    left_anchor = sum(lefts) / len(lefts)
    right_anchor = sum(rights) / len(rights)

    slots = [left_anchor]
    if common_size > 2:
        slots.append((left_anchor + right_anchor) / 2)
    slots.append(right_anchor)

    result = [_SIDE_TEXT[side_of_ratio(x)] for x in slots]
    while len(result) < count:
        result.append("")
    return result[:count]


def side_of_ratio(x_ratio: float) -> str:
    """画面横向位置 → left/center/right（与 CV 层同一套判据）。"""
    from ..providers.cv import side_of

    return side_of(x_ratio)


def person_overflow_note(report: InferenceReport) -> str:
    """人物数超上限时的说明 —— 由调用方放进 degradations，如实告知。

    统计的是**去重后**的人物数，否则同一个人被标成三个位置就会误报超限。
    """
    observed: list[str] = []
    for understanding in report.understandings:
        observed.extend(understanding.people)
    names = dedupe_person_labels(observed)
    if len(names) <= MAX_REVERSE_ENGINEERED_PERSONS:
        return ""
    return (
        f"识别到 {len(names)} 个人物，已截取前 {MAX_REVERSE_ENGINEERED_PERSONS} 个"
        f"（其余可能是次要人物）：{'、'.join(names[MAX_REVERSE_ENGINEERED_PERSONS:])}。"
        f"请按实际需要增删角色，每多一个角色就多占一张参考图配额。"
    )


def _fill_camera(ir: dict, report: InferenceReport) -> None:
    camera = ir["locked"]["B4_camera"]
    framings = [
        signal.framing.kind
        for signal in report.shots
        if signal.framing is not None and signal.framing.kind != "unknown"
    ]
    camera["framing_set"] = framings

    # 机位优先取 CV 的确定性结论；CV 给不出（亮度序列分不清镜头动与
    # 被摄体动）就退回模型判断。留空是最次的选择 —— 机位是高影响级维度。
    # **按多数镜头投票**，不取第一个镜头（见 infer.vote_rig）。
    from .infer import vote_rig

    rig, _source, _index = vote_rig(report.shots, report.understandings)
    camera["rig"] = rig or UNKNOWN
    # ``transition_logic``（镜头衔接逻辑）**留空**：它是"镜头之间怎么接"的字段，
    # 不是记判定来源的便签。原来往里写「机位判定来源：CV」是字段语义错位，
    # 哪天渲染它就会把内部说明写进正文。判定来源现在挂在 provenance 的 note 上。


def _fill_quality(ir: dict, report: InferenceReport) -> None:
    semantic = report.semantic
    ir["locked"]["B5_quality"]["preset"] = (
        semantic.quality_preset.value if semantic else QualityPreset.P1_PHONE.value
    )
    ir["locked"]["B5_quality"]["skin_text"] = "brief"


def _fill_beats(ir: dict, report: InferenceReport) -> None:
    """逐镜头填节拍。台词来自 ASR（经台词归属切分），**不做任何代写**。

    台词有两路来源，优先级从高到低：

    1. **``report.dialogue_turns``** —— 文本模型按画面证据切好的逐句台词，
       带说话人与所属镜头。这是主路：ASR 不给说话人和时间戳时，
       没有它双人对话会被压成一句话、全片只有最后一个镜头有台词（实测踩过）。
    2. **``_distribute_transcript``** —— 退路。只在归属不可用时使用，
       行为是"整段归给重叠最多的镜头"。
    """
    by_index = {u.shot_index: u for u in report.understandings}
    hook_index = report.semantic.hook_index if report.semantic else 0
    sole_speaker = (
        ir["locked"]["B2_person"][0]["name"]
        if len(ir["locked"]["B2_person"]) == 1 else ""
    )

    turns_by_shot = _turns_by_shot(report)
    lines_by_shot = {} if turns_by_shot else _distribute_transcript(report)
    known_people = [
        str(p.get("display_name") or p.get("name") or "").strip()
        for p in ir["locked"]["B2_person"]
    ]
    # 逐镜头描述里的人物写法先统一到规范名（见 canonical_person_mentions）。
    # 组装器再把规范名换成「人物N」—— 两步分开是因为"哪几个写法是同一身份"
    # 要靠这里的词表判据，而"正文用什么称呼"是组装器的事。
    aliases = person_alias_map(
        [label for u in report.understandings for label in u.people],
        known_people,
        texts=[
            text
            for u in report.understandings
            for text in (u.blocking, u.expression)
        ],
    )

    def _shot_text(value: str) -> str:
        return canonical_person_mentions(_text(value), aliases)

    beats: list[dict] = []
    for signal in report.shots:
        shot = signal.boundary
        understanding = by_index.get(shot.index)
        dialogue = (
            _dialogue_from_turns(turns_by_shot.get(shot.index, ()), known_people)
            if turns_by_shot
            else _dialogue_for(lines_by_shot.get(shot.index, ()), sole_speaker)
        )

        beats.append({
            "shot_id": shot.index,
            "framing": (
                _placeholder(signal.framing.kind) if signal.framing else UNKNOWN
            ),
            "camera_angle": _placeholder(
                understanding.camera_angle if understanding else ""
            ),
            "camera": _placeholder(understanding.camera if understanding else ""),
            "location": _text(understanding.location if understanding else ""),
            "beat_role": _beat_role(shot.index, hook_index, len(report.shots)),
            "duration_sec": round(shot.duration_sec, 3),
            "blocking": _shot_text(understanding.blocking if understanding else ""),
            "expression": _shot_text(understanding.expression if understanding else ""),
            "emotion": _text(understanding.emotion if understanding else ""),
            "dialogue": dialogue,
        })
    ir["creative"]["C2_beats"] = beats


def _turns_by_shot(report: InferenceReport) -> dict[int, list]:
    """归属好的台词按镜头分组。同一镜头多句按原顺序拼成一句。"""
    out: dict[int, list] = {}
    for turn in report.dialogue_turns:
        if str(turn.text).strip():
            out.setdefault(turn.shot_index, []).append(turn)
    return out


def _dialogue_from_turns(
    turns: Sequence, known_people: Sequence[str] = ()
) -> dict[str, Any]:
    """把一个镜头的多句台词合成一条。

    同一镜头里可能有多句（一问一答都在这个镜头里）。合成时**保留说话人**：
    说话人不同就写成"甲：… 乙：…"的形式，用户一眼能看出谁说什么。
    台词本身一个字都不改（红线 3：不代写台词）。

    说话人会**收敛到人物清单里的名字** —— 否则同一人在人物段叫
    「蓝色T恤男子」、在台词里叫「蓝色运动T恤男子」，用户对不上号。
    """
    usable = [t for t in turns if str(t.text).strip()]
    if not usable:
        return _empty_dialogue()
    speakers = [canonical_speaker(str(t.speaker).strip(), known_people) for t in usable]
    unique = [s for s in dict.fromkeys(speakers) if s]
    if len(unique) <= 1:
        line = "".join(str(t.text).strip() for t in usable)
        return {
            "speaker": unique[0] if unique else "",
            "line": line,
            "tone": str(usable[0].tone or "").strip(),
            "lip_sync": "",
            "pause_at": [],
        }
    # 多人对话：逐句带说话人，保留话轮
    parts: list[str] = []
    for turn, speaker in zip(usable, speakers):
        text = str(turn.text).strip()
        parts.append(f"{speaker}：{text}" if speaker else text)
    return {
        "speaker": "",
        "line": " ".join(parts),
        "tone": "",
        "lip_sync": "",
        "pause_at": [],
    }


def _beat_role(index: int, hook_index: int, total: int) -> str:
    """节拍功能（LOCK 位）。只做**结构推断**，不臆造内容。"""
    if hook_index and index == hook_index:
        return "hook"
    if index == 1:
        return "hook"
    if index == total:
        return "cta"
    if index == total - 1:
        return "proof"
    return "demo"


def _distribute_transcript(report: InferenceReport) -> dict[int, list]:
    """把 ASR 片段分配到镜头。

    两种情形分开处理，这是本模块最容易出错的地方：

    - **时间戳来自 ASR 词/句对齐**（``has_timestamps=True``）：按真实时间重叠
      分配。一句话横跨两个镜头是正常的；
    - **区间只是音频总时长推导的**（``has_timestamps=False``）：把整段归给
      **时序重叠最多**的那个镜头，其余镜头留空。

    为什么第二类不做比例切分：按比例切是在**编造对齐关系**，
    而 PRD 红线 7 要求"低置信度宁可留空也不猜"。硬切还会产生
    "你这气色也太好了吧，是 / 不是偷偷做项目啦？" 这种半句话。
    整段归位是诚实的：它只说"这段时间在说话"，切割交给 M2 的对齐环节由用户定。
    """
    precise = [s for s in report.transcript if s.has_timestamps and s.text.strip()]
    if precise:
        out: dict[int, list] = {}
        for signal in report.shots:
            out[signal.boundary.index] = [
                seg for seg in precise if _overlaps(seg.span, signal.boundary.span)
            ]
        return out

    coarse = [s for s in report.transcript if s.text.strip()]
    if not coarse or not report.shots:
        return {}

    out: dict[int, list] = {}
    for segment in coarse:
        # 只有一条镜头时没有歧义，直接归它
        if len(report.shots) == 1:
            target = report.shots[0]
        else:
            target = max(
                report.shots,
                key=lambda x: _overlap_ms(segment.span, x.boundary.span),
            )
        out.setdefault(target.boundary.index, []).append(segment)
    return out


def _overlap_ms(a: Span, b: Span) -> int:
    return max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


def _dialogue_for(lines: Sequence, sole_speaker: str = "") -> dict[str, Any]:
    """从候选片段里选置信度最高的一条落地。**低置信度不填**。

    ASR 多数情况下不返回说话人。此时若全片只有一个人物，
    说话人不存在歧义 —— 这是**推导**（从"只有一个角色"推出"就是他说的"），
    不是猜测，所以可以填。多于一个角色时留空，交给用户对齐。
    """
    candidates = [seg for seg in lines if str(seg.text).strip()]
    if not candidates:
        return _empty_dialogue()
    best = max(candidates, key=lambda s: s.confidence)
    if best.confidence < CONFIDENCE_REVIEW:
        # 低置信度：宁可留空让用户填，也不猜
        return _empty_dialogue()
    return {
        "speaker": best.speaker or sole_speaker,
        "line": best.text.strip(),
        "tone": "",
        "lip_sync": "",
        "pause_at": [],
    }


def _empty_dialogue() -> dict[str, Any]:
    return {"speaker": "", "line": "", "tone": "", "lip_sync": "", "pause_at": []}


def _overlaps(a, b) -> bool:
    if a.end_ms <= a.start_ms:  # 无时间戳的整段
        return False
    return a.start_ms < b.end_ms and b.start_ms < a.end_ms


def _fill_logline(ir: dict, report: InferenceReport) -> None:
    logline = report.semantic.logline if report.semantic else ""
    ir["creative"]["C1_logline"] = logline or ""


def _fill_audio(ir: dict, report: InferenceReport) -> None:
    """音色绑定清单。

    说话人有三路来源，必须**都收进来**：

    1. ASR 返回的说话人（声纹分离或 ``说话人：内容`` 格式）；
    2. 台词里已经落地的说话人 —— 包括"全片只有一个人物"时推导出来的那个。
       漏掉这一路会导致：台词有说话人、音色列表却是空的，
       闸门 G6 阻断、G13 缺音色维度、素材清单也少了音频素材；
    3. **有台词但没识别出说话人** —— 退回到已识别的人物清单。

    第 3 条曾经漏了：原来的回退条件写成 "没有任何台词时"，
    结果 ASR 出词但没给说话人、且模型也没在台词上写 speaker 时，
    B6_audio 是**空的** —— 交付物缺音色句、G6 阻断。
    实际素材里这种情况很常见（单人口播、ASR 不做声纹分离）。

    用户 2026-09-29 明确要求：**只要有人物说话，音色句就是固定要出现的**。
    反过来说：**没人说话就没有音色句** —— 静音片子里没有声音可以克隆。
    """
    speakers: list[str] = []
    for segment in report.transcript:
        if segment.speaker and segment.speaker not in speakers:
            speakers.append(segment.speaker)

    for beat in ir.get("creative", {}).get("C2_beats", []):
        dialogue = beat.get("dialogue") or {}
        speaker = str(dialogue.get("speaker") or "").strip()
        if str(dialogue.get("line") or "").strip() and speaker and speaker not in speakers:
            speakers.append(speaker)

    has_dialogue = any(
        str((b.get("dialogue") or {}).get("line") or "").strip()
        for b in ir.get("creative", {}).get("C2_beats", [])
    )
    # 有台词却一个说话人都没定下来 → 按人物清单补齐（第 3 条来源）
    if has_dialogue and not speakers:
        for person in ir.get("locked", {}).get("B2_person", []):
            name = str(person.get("name") or "").strip()
            if name:
                speakers.append(name)
    # 连人物清单也空（反推不出人）时，至少留一个说话人位，让音色句出现。
    # **用「说话人」而不是「人物1」**（实测踩过）：交付物里会写出
    # 「人物1声音音色参考[音频1]」，可【关键一致性】里一个角色都没有 ——
    # 凭空多出一个从未定义的人物标号。用泛称既不悬空，也仍然告诉用户要准备音色素材。
    if has_dialogue and not speakers:
        speakers.append("说话人")
    # **完全没有台词时不预留音色位**（实测踩过）：参考视频里没人说话
    # = 没有声音可以克隆。给一条静音片子派发「XX音频」素材位，
    # 等于让用户去准备一份根本不存在的音频，交付物里还会多出一段
    # 声称"百分百还原某人音色"的假约束。
    # （用户 2026-09-29 的要求是"只要**有人物说话**，音色句就要出现"。）
    # 之后要加配音时，把台词写进去再重新组装，音色句自然会出现。

    ir["locked"]["B6_audio"] = [
        {"speaker": name, "audio_ref": f"{name}音频", "volume_mode": "normal"}
        for name in speakers
    ]


def _fill_forbidden(ir: dict, report: InferenceReport) -> None:
    """禁止项**不反推，直接套 D0 全库**（框架 §8 关键原则）。"""
    from engine.core import phrases as ph

    library = ph.load_library()
    ir["forbidden"] = library.forbidden_items()


def _fill_delivery(ir: dict, report: InferenceReport) -> None:
    shot_mode = ir["meta"]["shot_mode"]
    strategy = {
        ShotMode.MULTI_SHOT.value: "S2",
        ShotMode.SAME_ANGLE_SEGMENTS.value: "S4",
    }.get(shot_mode, "S1")
    ir["delivery"]["strategy"] = strategy
    ir["delivery"]["mode"] = "phase1_manual"
    # 参数声明行默认**不写**（用户 2026-09-29 明确要求去掉）：
    # 比例/时长/分辨率在生成平台上是独立控件，写进提示词正文只是冗余。
    ir["delivery"]["include_param_header"] = False
    ir["delivery"]["first_frame_mode"] = "multimodal_reference"


# ----------------------------------------------------------------------
# 素材清单：指名需要哪些素材（Agent 不生成素材，只指名）
# ----------------------------------------------------------------------
def reference_manifest(ir: Mapping[str, Any]) -> list[dict[str, Any]]:
    """列出这条片子需要哪些素材，供上游照单准备（PRD F5.1）。"""
    manifest: list[dict[str, Any]] = []
    step = 0
    for person in ir.get("locked", {}).get("B2_person", []):
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        step += 1
        manifest.append({
            "slots": [f"{name}多视图"],
            "type": RefType.MULTI_VIEW.value,
            "placeholder": f"【{name}多视图】",
            "asset_id": None,
            "seedance_role": "reference_image",
            "prep_step": step,
            "prep_prompt": "人物多视图：正面 / 左右 3/4 侧 / 侧面，五官清晰无遮挡",
        })
    step += 1
    manifest.append({
        "slots": ["场景参考图"],
        "type": RefType.SCENE.value,
        "placeholder": "【场景参考图】",
        "asset_id": None,
        "seedance_role": "reference_image",
        "prep_step": step,
        "prep_prompt": "场景图：先生成人物位置关系图，再去掉人物生成场景图效果更稳定",
    })
    for track in ir.get("locked", {}).get("B6_audio", []):
        speaker = str(track.get("speaker") or "").strip()
        if not speaker:
            continue
        step += 1
        manifest.append({
            "slots": [f"{speaker}音频"],
            "type": RefType.AUDIO.value,
            "placeholder": f"【{speaker}音频】",
            "asset_id": None,
            "seedance_role": "reference_audio",
            "prep_step": step,
            "prep_prompt": "音色克隆样本：任意完整一段，2~15 秒，环境安静无混响",
        })
    return manifest
