"""框架契约 v1：Framework IR 的唯一真相源。

设计要点（这是我为了避免"屎山"做的第一个架构决定）：

    **Schema 即数据。** 字段定义本身就是一张可遍历的表（``FIELDS``），
    字段级锁定规则、四类位置标注、应用位置追踪全部由它推导，
    不再各写一份。

于是"哪个字段能改"只有一个答案；改契约只需改一处。

契约已冻结为 v1（PRD §8）。任何结构调整 = 版本升级，不是"小改一下"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from .types import FRAMEWORK_VERSION, PositionCategory

# --------------------------------------------------------------------------
# 字段种类
# --------------------------------------------------------------------------
# TEXT        单行文本
# LONGTEXT    多行文本（会组装进正文）
# ENUM        白名单枚举 —— 天然 PARAM 位
# ENUM_LIST   枚举数组（景别序列等）
# FLAG        布尔开关
# NUMBER      数值（整数）
# OBJECT      嵌套结构
# LIST        对象数组
# REF_LIST    参考物料数组（唯一的写入路径来自"素材绑定"这一动作）
#
# 约定：带 choices 的字段是 PARAM；未标注 category 的，由 note 说明，
# 不写 category 的字段一律视为 FREE（可改）—— 与框架 §7.1 逐条对齐。

KIND_TEXT = "text"
KIND_LONGTEXT = "longtext"
KIND_ENUM = "enum"
KIND_ENUM_LIST = "enum_list"
KIND_FLAG = "flag"
KIND_NUMBER = "number"
KIND_OBJECT = "object"
KIND_LIST = "list"
KIND_REF_LIST = "ref_list"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """一个可寻址字段的完整定义。"""

    path: str
    label: str
    kind: str
    category: PositionCategory
    risk: str = "low"
    apply_section: str = ""
    """该字段落在输出提示词的哪个段落（框架 §10.3 应用位置）。"""
    choices: tuple[str, ...] = ()
    whitelist_addable: bool = False
    """forbidden 专用：可增不可减 —— 减是 LOCK 级违规。"""
    note: str = ""
    children: tuple["FieldSpec", ...] = ()

    @property
    def is_container(self) -> bool:
        return self.kind in (KIND_OBJECT, KIND_LIST, KIND_REF_LIST)

    @property
    def is_editable(self) -> bool:
        return self.category in (PositionCategory.FREE, PositionCategory.PARAM)

    def walk(self) -> Iterator["FieldSpec"]:
        yield self
        for child in self.children:
            yield from child.walk()


def _f(
    path: str,
    label: str,
    kind: str,
    category: PositionCategory,
    risk: str = "low",
    apply_section: str = "",
    choices: Sequence[str] = (),
    whitelist_addable: bool = False,
    note: str = "",
    children: Sequence[FieldSpec] = (),
) -> FieldSpec:
    return FieldSpec(
        path=path,
        label=label,
        kind=kind,
        category=category,
        risk=risk,
        apply_section=apply_section,
        choices=tuple(choices),
        whitelist_addable=whitelist_addable,
        note=note,
        children=tuple(children),
    )


LOCK = PositionCategory.LOCK
PARAM = PositionCategory.PARAM
FREE = PositionCategory.FREE
AUTO = PositionCategory.AUTO

# --------------------------------------------------------------------------
# 输出段落：与框架 §9.1 输出模板一一对应
# --------------------------------------------------------------------------
SEC_PARAMS = "参数声明"
SEC_REFS = "参考素材"
SEC_LOCKED = "关键一致性"
SEC_LOGLINE = "剧情"
SEC_BEATS = "镜头节拍"
SEC_DIALOGUE = "对白与口型"
SEC_FORBIDDEN = "禁止项"


# ==========================================================================
# meta —— 结构参数。决定输出格式与拆条策略。
# ==========================================================================
_META = _f("meta", "结构参数", KIND_OBJECT, PARAM, risk="critical", children=(
    _f("meta.genre", "场景类型", KIND_ENUM, PARAM, risk="medium",
       choices=("dialogue", "teaching", "studio", "street_interview", "replacement")),
    _f("meta.shot_mode", "镜头模式", KIND_ENUM, PARAM, risk="critical",
       note="决定拆条策略与提示词条数 —— 改动风险最高",
       choices=("one_take", "multi_shot", "same_angle_segments")),
    _f("meta.aspect_ratio", "画面比例", KIND_ENUM, PARAM,
       choices=("9:16", "16:9", "1:1"), apply_section=SEC_PARAMS),
    _f("meta.total_duration_sec", "全片时长", KIND_NUMBER, AUTO, risk="critical",
       note="ffprobe 是唯一可信来源，不靠模型估"),
    _f("meta.target_model", "目标模型", KIND_ENUM, PARAM, apply_section=SEC_PARAMS,
       choices=("seedance-2.0", "seedance-2.5")),
    _f("meta.replacement_level", "替换级别", KIND_ENUM, PARAM, risk="medium",
       note="由参考物料种类推导，不作为独立输入",
       choices=("L0", "L1", "L2", "L3")),
))

# ==========================================================================
# references —— A 区：参考物料。素材绑定是唯一可写路径。
# ==========================================================================
_REFERENCES = _f("references", "参考物料", KIND_REF_LIST, FREE, risk="high",
                 apply_section=SEC_REFS, children=(
    _f("references[].slots", "素材位名称", KIND_ENUM_LIST, FREE),
    _f("references[].type", "素材类型", KIND_ENUM, PARAM, risk="high",
       choices=("multi_view", "scene", "blocking", "pose_sheet", "video", "audio")),
    _f("references[].placeholder", "原文占位符", KIND_TEXT, AUTO,
       note="解析期识别，生成期必须全部绑定 —— 交付物中【】残留必须为 0"),
    _f("references[].asset_id", "绑定素材句柄", KIND_TEXT, FREE, risk="high"),
    _f("references[].seedance_role", "Seedance 归属", KIND_ENUM, AUTO,
       choices=("reference_image", "reference_video", "reference_audio")),
    _f("references[].prep_step", "物料生成链顺序", KIND_NUMBER, AUTO),
    _f("references[].prep_prompt", "物料生成建议", KIND_LONGTEXT, AUTO),
))

# ==========================================================================
# locked —— B 区：关键一致性。决定"像不像原片"。
# ==========================================================================
_LOCKED = _f("locked", "关键一致性", KIND_OBJECT, LOCK, risk="critical",
             apply_section=SEC_LOCKED, children=(
    _f("locked.B1_scene", "场景一致性", KIND_OBJECT, LOCK, children=(
        _f("locked.B1_scene.type", "场景类型", KIND_ENUM, PARAM, risk="medium",
           apply_section=SEC_LOCKED,
           choices=("菜市场", "茶楼雅间", "小区花园步行道", "商场走廊", "室内客厅",
                    "户外公园", "瑜伽室", "演播厅舞台", "演播厅访谈区", "客厅沙发区")),
        _f("locked.B1_scene.anchor", "场景锚定", KIND_LONGTEXT, LOCK, risk="high",
           apply_section=SEC_LOCKED),
        _f("locked.B1_scene.lighting", "光线", KIND_ENUM, PARAM,
           apply_section=SEC_LOCKED,
           choices=("场景光线明亮通透", "室内光自然通透", "阴天柔和漫射自然光")),
        _f("locked.B1_scene.cross_shot_consistency", "跨镜头一致", KIND_FLAG, LOCK),
    )),
    _f("locked.B2_person", "人物一致性", KIND_LIST, LOCK, risk="high",
       apply_section=SEC_LOCKED, children=(
        _f("locked.B2_person[].name", "角色名", KIND_TEXT, FREE),
        _f("locked.B2_person[].multi_view_ref", "多视图引用", KIND_TEXT, FREE, risk="high"),
        _f("locked.B2_person[].lock_strength", "五官锁定强度", KIND_ENUM, PARAM,
           risk="high", apply_section=SEC_LOCKED, choices=("standard", "extreme")),
        _f("locked.B2_person[].wardrobe", "服装约束", KIND_LONGTEXT, LOCK,
           apply_section=SEC_LOCKED),
        _f("locked.B2_person[].skin", "皮肤质感句", KIND_ENUM, PARAM,
           apply_section=SEC_LOCKED, choices=("full", "brief")),
        _f("locked.B2_person[].blink", "眨眼节拍", KIND_ENUM, PARAM,
           apply_section=SEC_LOCKED, choices=("3-4s", "5s", "3s")),
    )),
    _f("locked.B3_blocking", "空间与站位", KIND_OBJECT, LOCK, risk="critical",
       apply_section=SEC_LOCKED, children=(
        _f("locked.B3_blocking.positions", "左右站位", KIND_ENUM_LIST, LOCK,
           risk="critical", apply_section=SEC_LOCKED),
        _f("locked.B3_blocking.occlusion", "遮挡关系", KIND_LONGTEXT, LOCK,
           risk="high", apply_section=SEC_LOCKED),
        _f("locked.B3_blocking.eye_line", "视线方向", KIND_LONGTEXT, LOCK,
           risk="high", apply_section=SEC_LOCKED),
        _f("locked.B3_blocking.off_screen", "画外对象", KIND_LONGTEXT, LOCK,
           risk="high", apply_section=SEC_LOCKED),
    )),
    _f("locked.B4_camera", "镜头与运镜", KIND_OBJECT, LOCK, children=(
        _f("locked.B4_camera.shot_mode", "镜头模式", KIND_ENUM, PARAM, risk="critical",
           choices=("one_take", "multi_shot", "same_angle_segments")),
        _f("locked.B4_camera.rig", "机位", KIND_ENUM, PARAM, risk="high",
           apply_section=SEC_LOCKED,
           choices=("固定", "斯坦尼康跟拍", "稳定推近", "固定平视", "侧前方缓慢推近",
                    "手持自拍", "固定偷拍", "过肩正反打", "固定机位全程不变")),
        _f("locked.B4_camera.framing_set", "景别序列", KIND_ENUM_LIST, FREE,
           apply_section=SEC_BEATS),
        _f("locked.B4_camera.ratio", "画面比例", KIND_ENUM, PARAM,
           choices=("9:16", "16:9", "1:1")),
        _f("locked.B4_camera.transition_logic", "镜头衔接逻辑", KIND_LONGTEXT, FREE,
           apply_section=SEC_BEATS),
    )),
    _f("locked.B5_quality", "质感与画质", KIND_OBJECT, LOCK, apply_section=SEC_LOCKED,
       children=(
        _f("locked.B5_quality.preset", "画质预设", KIND_ENUM, PARAM, risk="medium",
           apply_section=SEC_LOCKED, choices=("P1", "P2", "P3")),
        _f("locked.B5_quality.skin_text", "皮肤质感标准句", KIND_ENUM, PARAM,
           apply_section=SEC_LOCKED, choices=("full", "brief")),
        _f("locked.B5_quality.grain", "噪点与锐度", KIND_LONGTEXT, AUTO,
           apply_section=SEC_LOCKED),
    )),
    _f("locked.B6_audio", "声音一致性", KIND_LIST, LOCK, risk="high",
       apply_section=SEC_LOCKED, children=(
        _f("locked.B6_audio[].speaker", "说话人", KIND_TEXT, FREE),
        _f("locked.B6_audio[].audio_ref", "音色参考绑定", KIND_TEXT, FREE, risk="high"),
        _f("locked.B6_audio[].volume_mode", "音量模式", KIND_ENUM, PARAM, risk="medium",
           apply_section=SEC_LOCKED, choices=("normal", "whisper")),
    )),
))

# ==========================================================================
# creative —— C 区：创作区。用户唯一能改的区域。
# ==========================================================================
_CREATIVE = _f("creative", "创作区", KIND_OBJECT, FREE, children=(
    _f("creative.C1_logline", "剧情主线", KIND_LONGTEXT, FREE, risk="medium",
       apply_section=SEC_LOGLINE, note="全片语义锚，用户最先要改的一句话"),
    _f("creative.C2_beats", "镜头节拍", KIND_LIST, FREE, risk="high",
       apply_section=SEC_BEATS, children=(
        _f("creative.C2_beats[].shot_id", "镜头号", KIND_NUMBER, AUTO, risk="high",
           note="结构位，不可改"),
        _f("creative.C2_beats[].framing", "景别", KIND_ENUM, PARAM, risk="high",
           apply_section=SEC_BEATS,
           choices=("全景", "中全景", "中景", "中近景", "近景", "脸部特写")),
        _f("creative.C2_beats[].camera", "机位运镜", KIND_ENUM, PARAM, risk="high",
           apply_section=SEC_BEATS,
           choices=("平视", "正面", "过肩", "俯拍", "仰拍", "固定", "推近", "跟拍",
                    "横摇", "手持")),
        _f("creative.C2_beats[].location", "场景", KIND_TEXT, PARAM,
           apply_section=SEC_BEATS),
        _f("creative.C2_beats[].beat_role", "节拍功能", KIND_ENUM, LOCK, risk="critical",
           note="爆款结构位 —— 改了就不是这条爆款的复刻",
           choices=("hook", "setup", "demo", "proof", "cta", "transition")),
        _f("creative.C2_beats[].duration_sec", "时长", KIND_NUMBER, AUTO, risk="high"),
        _f("creative.C2_beats[].blocking", "走位动作", KIND_LONGTEXT, FREE,
           apply_section=SEC_BEATS),
        _f("creative.C2_beats[].expression", "表情", KIND_LONGTEXT, FREE,
           apply_section=SEC_BEATS),
        _f("creative.C2_beats[].emotion", "情绪标签", KIND_TEXT, FREE,
           apply_section=SEC_BEATS),
        _f("creative.C2_beats[].dialogue", "对白", KIND_OBJECT, FREE,
           apply_section=SEC_DIALOGUE, children=(
            _f("creative.C2_beats[].dialogue.speaker", "说话人", KIND_TEXT, FREE,
               apply_section=SEC_DIALOGUE),
            _f("creative.C2_beats[].dialogue.line", "台词正文", KIND_LONGTEXT, FREE,
               risk="medium", apply_section=SEC_DIALOGUE,
               note="最核心的创作位；Agent 绝不代写"),
            _f("creative.C2_beats[].dialogue.tone", "语气", KIND_TEXT, FREE,
               apply_section=SEC_DIALOGUE),
            _f("creative.C2_beats[].dialogue.lip_sync", "口型要求", KIND_LONGTEXT, FREE,
               apply_section=SEC_DIALOGUE),
            _f("creative.C2_beats[].dialogue.pause_at", "停顿点", KIND_ENUM_LIST, FREE,
               apply_section=SEC_DIALOGUE),
        )),
    )),
))

# ==========================================================================
# forbidden —— D 区：禁止项。可增不可减。
# ==========================================================================
_FORBIDDEN = _f("forbidden", "禁止项", KIND_LIST, PARAM, apply_section=SEC_FORBIDDEN,
                whitelist_addable=True,
                note="可增不可减；通用禁止项全库逐字，不可删",
                children=(
    _f("forbidden[].id", "条目编号", KIND_TEXT, AUTO),
    _f("forbidden[].group", "分组", KIND_ENUM, AUTO,
       choices=("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")),
    _f("forbidden[].text", "条文", KIND_LONGTEXT, PARAM, apply_section=SEC_FORBIDDEN),
))

# ==========================================================================
# delivery / plan —— 交付与拆条。不属于框架原文，但组装必须用。
# ==========================================================================
_DELIVERY = _f("delivery", "交付形态", KIND_OBJECT, PARAM, children=(
    _f("delivery.mode", "交付模式", KIND_ENUM, PARAM, risk="medium",
       choices=("phase1_manual", "phase2_api"),
       note="决定参数声明是否写进正文 —— 同一提示词两种模式正文不同"),
    _f("delivery.include_param_header", "写参数声明", KIND_FLAG, PARAM,
       apply_section=SEC_PARAMS),
    _f("delivery.strategy", "拆条策略", KIND_ENUM, PARAM, risk="critical",
       choices=("S1", "S2", "S3", "S4")),
    _f("delivery.first_frame_mode", "互斥模式", KIND_ENUM, PARAM, risk="critical",
       apply_section=SEC_PARAMS,
       choices=("multimodal_reference", "first_frame", "first_last_frame"),
       note="首帧/首尾帧/多模态参考三者互斥，不可混用"),
))

_SCHEMA: tuple[FieldSpec, ...] = (
    _META, _REFERENCES, _LOCKED, _CREATIVE, _FORBIDDEN, _DELIVERY,
)

# 展开成 path -> FieldSpec 的索引
FIELDS: dict[str, FieldSpec] = {}
for _root in _SCHEMA:
    for _spec in _root.walk():
        FIELDS[_spec.path] = _spec

ROOT_FIELDS: tuple[FieldSpec, ...] = _SCHEMA


# --------------------------------------------------------------------------
# 路径寻址
# --------------------------------------------------------------------------
_INDEX_SUFFIX = "[]"
_INDEX_TAIL = __import__("re").compile(r"\[(\d+)\]")


def concrete_path(template: str, index: int) -> str:
    """``locked.B2_person[].name`` + 0 → ``locked.B2_person[0].name``"""
    return template.replace(_INDEX_SUFFIX, f"[{index}]", 1)


def spec_for(path: str) -> FieldSpec | None:
    """按具体路径取字段定义。会自动把 ``[0]`` 还原成模板 ``[]``。"""
    if path in FIELDS:
        return FIELDS[path]
    normalized = normalize_path(path)
    return FIELDS.get(normalized)


def normalize_path(path: str) -> str:
    """``locked.B2_person[0].name`` → ``locked.B2_person[].name``

    位置规则、维度映射、diff 比对都按**模板路径**工作，
    具体实例路径必须先过这里。
    """
    out: list[str] = []
    i = 0
    while i < len(path):
        if path[i] == "[":
            end = path.find("]", i)
            if end == -1:
                out.append(path[i:])
                break
            out.append(_INDEX_SUFFIX)
            i = end + 1
            continue
        out.append(path[i])
        i += 1
    return "".join(out)


def _strip_indexes(path: str) -> str:  # 兼容旧调用
    return normalize_path(path)


def iter_specs(prefix: str = "") -> Iterator[FieldSpec]:
    """遍历字段定义；给前缀可只看某个子树。"""
    for path, spec in FIELDS.items():
        if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "["):
            yield spec


def category_of(path: str) -> PositionCategory:
    spec = spec_for(path)
    return spec.category if spec else PositionCategory.FREE


def label_of(path: str) -> str:
    spec = spec_for(path)
    return spec.label if spec else path


def apply_section_of(path: str) -> str:
    spec = spec_for(path)
    return spec.apply_section if spec else ""


# --------------------------------------------------------------------------
# IR 树：用纯 Python 结构承载，schema 负责校验与定位
# --------------------------------------------------------------------------
IR = dict[str, Any]


def empty_ir(
    *,
    genre: str = "dialogue",
    shot_mode: str = "one_take",
    aspect_ratio: str = "9:16",
    target_model: str = "seedance-2.0",
) -> IR:
    """产出一份结构完整、值为空的 IR。

    刻意不用 `None` 填充：**未知用 UNKNOWN 占位**，
    与"用户明确留空"区分开 —— 这是 PRD 红线 7「不猜」的落地基础。
    """
    from .types import UNKNOWN

    return {
        "framework_version": FRAMEWORK_VERSION,
        "meta": {
            "genre": genre,
            "shot_mode": shot_mode,
            "replacement_level": "L0",
            "aspect_ratio": aspect_ratio,
            "total_duration_sec": 0,
            "target_model": target_model,
        },
        "references": [],
        "locked": {
            "B1_scene": {
                "type": UNKNOWN,
                "anchor": "",
                "lighting": UNKNOWN,
                "cross_shot_consistency": True,
            },
            "B2_person": [],
            "B3_blocking": {
                "positions": [],
                "occlusion": "",
                "eye_line": "",
                "off_screen": "",
            },
            "B4_camera": {
                "shot_mode": shot_mode,
                "rig": UNKNOWN,
                "framing_set": [],
                "ratio": aspect_ratio,
                "transition_logic": "",
            },
            "B5_quality": {"preset": "P1", "skin_text": "brief", "grain": ""},
            "B6_audio": [],
        },
        "creative": {
            "C1_logline": "",
            "C2_beats": [],
        },
        "forbidden": [],
        "delivery": {
            "mode": "phase1_manual",
            "include_param_header": True,
            # 空 = 交给自动推荐（由 shot_mode 推导）。用户显式指定才会覆盖。
            "strategy": "",
            "first_frame_mode": "multimodal_reference",
        },
    }


def get_in(ir: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """按 ``locked.B2_person[0].name`` 形式读值。"""
    node: Any = ir
    for token, index in _parse_path(path):
        if not isinstance(node, Mapping) or token not in node:
            return default
        node = node[token]
        if index is not None:
            if not isinstance(node, Sequence) or index >= len(node):
                return default
            node = node[index]
    return node


def set_in(ir: IR, path: str, value: Any) -> None:
    """按路径写值。路径必须已存在（容器由反推/组装阶段建好）。"""
    tokens = _parse_path(path)
    node: Any = ir
    for token, index in tokens[:-1]:
        node = node[token]
        if index is not None:
            node = node[index]
    token, index = tokens[-1]
    if index is None:
        node[token] = value
    else:
        node[index] = value


def _parse_path(path: str) -> list[tuple[str, int | None]]:
    tokens: list[tuple[str, int | None]] = []
    for part in path.split("."):
        if part.endswith("]") and "[" in part:
            name, _, rest = part.partition("[")
            tokens.append((name, int(rest[:-1])))
        else:
            tokens.append((part, None))
    return tokens


def flatten(ir: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """把 IR 摊平成 ``(具体路径, 值)`` 序列。位置标注与 diff 都靠它。

    两类数组要分清，这直接决定路径能不能对上契约：

    - **对象数组**（``B2_person`` / ``C2_beats``）：逐个成员摊成
      ``locked.B2_person[0].name``，与契约模板 ``locked.B2_person[].name`` 对应；
    - **标量数组**（``positions`` / ``framing_set``）：是**一个逻辑值**，
      整体产出在 ``locked.B3_blocking.positions`` —— 拆成 ``positions[0]``
      会让契约里声明的那条规则永远匹配不上（G10 就会漏检站位改动）。
    """
    for key, value in ir.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            if not value:
                yield path, value
            else:
                yield from flatten(value, path)
        elif isinstance(value, list):
            if not value or not any(isinstance(item, Mapping) for item in value):
                # 空数组与标量数组都整体产出
                yield path, value
                continue
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                if isinstance(item, Mapping):
                    yield from flatten(item, item_path)
                else:
                    yield item_path, item
        else:
            yield path, value


# --------------------------------------------------------------------------
# 写入门禁：字段级锁定规则（框架 §7.1）的唯一执行点
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WriteDecision:
    allowed: bool
    reason: str = ""
    requires_whitelist: bool = False

    def __bool__(self) -> bool:  # pragma: no cover - 语法糖
        return self.allowed


def can_write(path: str, *, value: Any = None, current: Any = None) -> WriteDecision:
    """判断某个位置能否被写入。

    **UI 与 API 两层都必须过这里**（PRD F2.3 位置级编辑门禁）。
    """
    spec = spec_for(path)
    if spec is None:
        return WriteDecision(False, f"契约中不存在该位置：{path}")

    if spec.whitelist_addable:
        # forbidden：可增不可减
        if isinstance(current, Sequence) and isinstance(value, Sequence):
            if len(value) < len(current):
                return WriteDecision(False, "禁止项可增不可减：删条会破坏崩坏兜底")
        return WriteDecision(True)

    if spec.category is PositionCategory.LOCK:
        return WriteDecision(
            False,
            f"「{spec.label}」是锁定位，决定复刻是否成立，只读。",
        )
    if spec.category is PositionCategory.AUTO:
        return WriteDecision(False, f"「{spec.label}」是自动位，由系统从视频分析得出。")
    if spec.category is PositionCategory.PARAM:
        if value is None or not spec.choices:
            return WriteDecision(True, requires_whitelist=bool(spec.choices))
        if value not in spec.choices:
            return WriteDecision(
                False,
                f"「{spec.label}」只能从白名单切换，收到：{value}",
                requires_whitelist=True,
            )
        return WriteDecision(True, requires_whitelist=True)
    return WriteDecision(True)


def schema_digest() -> list[dict[str, Any]]:
    """导出契约清单，供前端渲染与文档生成。"""
    return [
        {
            "path": spec.path,
            "label": spec.label,
            "kind": spec.kind,
            "category": spec.category.value,
            "risk": spec.risk,
            "apply_section": spec.apply_section,
            "choices": list(spec.choices),
            "note": spec.note,
        }
        for spec in FIELDS.values()
    ]


# --------------------------------------------------------------------------
# 级联改写：改一个位置要带动一串引用
# --------------------------------------------------------------------------
def cascade_person_rename(ir: IR, index: int, old: str, new: str) -> list[str]:
    """改人物名时，同步所有以该名字派生的引用。

    **为什么必须级联**：人物名不是孤立字段，它被四处引用 ——
    音色绑定的说话人、多视图素材的槽位名、占位符文本、台词说话人。
    只改名字不同步，会造成两种坏结果，实测都踩过：

    1. 交付物自相矛盾：「王姐形象参考[图1]」和「穿紫色连衣裙的女子声音音色参考[音频1]」
       同时出现，用户不知道为什么冒出个没听过的角色名；
    2. 闸门误报：G6 发现"王姐缺音色绑定"而阻断，其实音色是有的，
       只是还挂在旧名字下。

    **按名字匹配而不是按索引**：``B6_audio`` 的顺序来自台词说话人，
    与 ``B2_person`` 的下标并不保证一致，按下标改是脆的。
    ``index`` 只用于提示，不参与匹配。

    返回被同步修改的路径列表，便于前端提示"哪些地方跟着变了"。
    """
    touched: list[str] = []
    if not old or not new or old == new:
        return touched

    for audio_index, track in enumerate(ir.get("locked", {}).get("B6_audio", [])):
        if str(track.get("speaker") or "") == old:
            track["speaker"] = new
            touched.append(f"locked.B6_audio[{audio_index}].speaker")
        ref = str(track.get("audio_ref") or "")
        if old in ref:
            track["audio_ref"] = ref.replace(old, new)
            touched.append(f"locked.B6_audio[{audio_index}].audio_ref")

    for ref_index, ref in enumerate(ir.get("references", [])):
        slots = [str(s) for s in ref.get("slots", [])]
        if any(old in slot for slot in slots):
            ref["slots"] = [slot.replace(old, new) for slot in slots]
            touched.append(f"references[{ref_index}].slots")
        placeholder = str(ref.get("placeholder") or "")
        if old in placeholder:
            ref["placeholder"] = placeholder.replace(old, new)
            touched.append(f"references[{ref_index}].placeholder")

    # **必须同步 multi_view_ref**：它是句柄解析的索引键。
    # 漏了它会导致 ``形象参考[王姐多视图]`` 解析不出 [图N]，
    # 退化成字面文本 —— 交付物直接不可用（实测踩过）。
    for person_index, person in enumerate(ir.get("locked", {}).get("B2_person", [])):
        ref = str(person.get("multi_view_ref") or "")
        if old in ref:
            person["multi_view_ref"] = ref.replace(old, new)
            touched.append(f"locked.B2_person[{person_index}].multi_view_ref")

    for beat_index, beat in enumerate(ir.get("creative", {}).get("C2_beats", [])):
        dialogue = beat.get("dialogue") or {}
        if str(dialogue.get("speaker") or "") == old:
            dialogue["speaker"] = new
            touched.append(f"creative.C2_beats[{beat_index}].dialogue.speaker")

    _ = index  # 保留参数以便调用方表达意图，但匹配不依赖它
    return touched


def referenced_by(ir: Mapping[str, Any], path: str) -> list[str]:
    """反查：改这个位置会影响哪些别的位置。用于 UI 提示。"""
    spec = spec_for(path)
    if spec is None or not spec.path.endswith("B2_person[].name"):
        return []
    match = _INDEX_TAIL.search(path)
    if not match:
        return []
    index = int(match.group(1))
    out = [f"locked.B6_audio[{index}].speaker", f"locked.B6_audio[{index}].audio_ref"]
    out.append("references[].slots / placeholder（按名字匹配）")
    out.append("creative.C2_beats[].dialogue.speaker（按名字匹配）")
    return out
