"""修改位置与作用位置（框架 §10）。

这是本产品的**核心操作定义**：对框架的每一处，同时回答两件事——
「用户能不能改」（修改位置）与「改了会影响成片的什么」（作用位置）。

三个概念必须分清，混在一起就是屎山的起点：

    PositionCategory  四类位置（LOCK/PARAM/FREE/AUTO）—— 工程视角
    Color             三种颜色（蓝/橙/灰）—— 用户视角，由上面单向映射
    Provenance        反推溯源（置信度 + 时间码证据）—— 可信度视角

映射规则集中在本模块的 ``color_for``，是**唯一**的着色实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from . import framework as fw
from .types import (
    Color,
    PositionCategory,
    Provenance,
    RiskLevel,
    Span,
)

# --------------------------------------------------------------------------
# 颜色规则的例外表
# --------------------------------------------------------------------------
# 默认规则（见 color_for）已经能覆盖绝大多数位置。这里只登记**例外**，
# 让"为什么这个位置是灰的"永远能在代码里找到一句话解释。

_GRAY_OVERRIDES: frozenset[str] = frozenset({
    # 枚举档位，但不落在正文里 —— 用户改了看不到正文变化，没必要给橙色
    "meta.genre",
    "meta.shot_mode",
    "meta.replacement_level",
    "meta.target_model",
    "locked.B4_camera.shot_mode",
    "locked.B4_camera.ratio",
    "locked.B6_audio[].volume_mode",
    "references[].type",
    "forbidden[].group",
    # 结构位与开关：改了会破坏可比性，属于"别动"
    "locked.B1_scene.cross_shot_consistency",
    "delivery.mode",
    "delivery.include_param_header",
    "delivery.first_frame_mode",
})

_ORANGE_OVERRIDES: frozenset[str] = frozenset({
    # 自由文本，但落点不在正文，而是驱动素材绑定动作
    "references[].asset_id",
    "locked.B2_person[].multi_view_ref",
    "locked.B6_audio[].audio_ref",
    # 自由文本，落在参数声明区（用户看得到、也常改）
    "meta.aspect_ratio",
})


def color_for(path: str) -> Color:
    """位置 → 颜色。用户视角只有三种状态，绝不出现第四种（PRD F2.2）。"""
    if path.startswith("references"):
        # A 区的素材绑定就是蓝位：点一下 = 换素材
        return Color.BLUE

    spec = fw.spec_for(path)
    if spec is None:
        return Color.GRAY

    if path in _GRAY_OVERRIDES:
        return Color.GRAY
    if path in _ORANGE_OVERRIDES:
        return Color.ORANGE

    if spec.category is PositionCategory.FREE:
        # 纯文本创作位：剧情、走位、表情、台词、角色名
        if spec.kind in (fw.KIND_TEXT, fw.KIND_LONGTEXT, fw.KIND_ENUM_LIST):
            return Color.ORANGE if spec.apply_section else Color.GRAY
        return Color.GRAY

    if spec.category is PositionCategory.PARAM and spec.kind is fw.KIND_ENUM:
        # 档位位：落在正文里就给橙色（景别/机位/光线/皮肤句…），否则灰色
        return Color.ORANGE if spec.apply_section else Color.GRAY

    return Color.GRAY


# --------------------------------------------------------------------------
# 位置表
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Position:
    """一个具体位置。

    ``apply_target`` 是框架 §10.3 要求的"应用位置"：改了它，输出提示词的哪一段会变。
    """

    path: str
    label: str
    category: PositionCategory
    color: Color
    kind: str
    risk: RiskLevel
    value: Any = None
    choices: tuple[str, ...] = ()
    editable: bool = False
    apply_target: str = ""
    provenance: Provenance | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "category": self.category.value,
            "color": self.color.value,
            "kind": self.kind,
            "risk": self.risk.value,
            "value": self.value,
            "choices": list(self.choices),
            "editable": self.editable,
            "apply_target": self.apply_target,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "note": self.note,
        }

    @property
    def must_confirm(self) -> bool:
        return bool(self.provenance and self.provenance.must_confirm)


@dataclass(slots=True)
class PositionTable:
    """某个 IR 的完整位置表。UI 渲染与改写门禁都读它。"""

    positions: list[Position] = field(default_factory=list)

    def by_path(self, path: str) -> Position | None:
        for pos in self.positions:
            if pos.path == path:
                return pos
        return None

    def of_color(self, color: Color) -> list[Position]:
        return [p for p in self.positions if p.color is color]

    def of_category(self, category: PositionCategory) -> list[Position]:
        return [p for p in self.positions if p.category is category]

    @property
    def counts(self) -> dict[str, int]:
        return {
            Color.BLUE.value: len(self.of_color(Color.BLUE)),
            Color.ORANGE.value: len(self.of_color(Color.ORANGE)),
            Color.GRAY.value: len(self.of_color(Color.GRAY)),
        }

    @property
    def pending_confirm(self) -> list[Position]:
        """必须用户确认的位置（置信度 < 0.70）。"""
        return [p for p in self.positions if p.must_confirm]

    def blue_slots(self) -> list[Position]:
        """蓝位：素材清单就是从这里长出来的（PRD F5.1）。"""
        return [
            p for p in self.positions
            if p.color is Color.BLUE and p.path.endswith("].asset_id")
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "positions": [p.to_dict() for p in self.positions],
        }

    def __iter__(self) -> Iterator[Position]:
        return iter(self.positions)

    def __len__(self) -> int:
        return len(self.positions)


# --------------------------------------------------------------------------
# 组装位置表
# --------------------------------------------------------------------------
def build_positions(
    ir: Mapping[str, Any],
    provenance: Mapping[str, Provenance] | None = None,
) -> PositionTable:
    """遍历 IR 叶子，逐个贴上类别、颜色、门禁与溯源。"""
    prov = provenance or {}
    table = PositionTable()
    for path, value in fw.flatten(ir):
        spec = fw.spec_for(path)
        if spec is None:
            # 契约外的字段：不给颜色，但在表里留痕便于发现契约漂移
            table.positions.append(Position(
                path=path,
                label=path,
                category=PositionCategory.FREE,
                color=Color.GRAY,
                kind="unknown",
                risk=RiskLevel.LOW,
                value=value,
                note="契约外字段 —— 说明写入方绕过了契约，需排查",
            ))
            continue

        decision = fw.can_write(path)
        category = spec.category
        # 锁定位即使"是空容器"也保持 LOCK，避免前端误判成可填
        if category is PositionCategory.LOCK and value in ([], {}):
            category = PositionCategory.LOCK
        table.positions.append(Position(
            path=path,
            label=spec.label,
            category=category,
            color=color_for(path),
            kind=spec.kind,
            risk=RiskLevel(spec.risk),
            value=value,
            choices=spec.choices,
            editable=decision.allowed and (
                bool(spec.apply_section) or category is PositionCategory.FREE
            ),
            apply_target=spec.apply_section,
            provenance=prov.get(path),
            note=spec.note,
        ))
    return table


def blue_slot_manifest(ir: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从参考物料生成"给上游照单准备"的素材清单（PRD F5.1）。

    Agent 不生成素材，只做三件事：指名、接收绑定、正确引用。
    """
    from .types import RefType, SeedanceRole

    role_of: dict[RefType, SeedanceRole] = {
        RefType.MULTI_VIEW: SeedanceRole.IMAGE,
        RefType.SCENE: SeedanceRole.IMAGE,
        RefType.BLOCKING: SeedanceRole.IMAGE,
        RefType.POSE_SHEET: SeedanceRole.IMAGE,
        RefType.VIDEO: SeedanceRole.VIDEO,
        RefType.AUDIO: SeedanceRole.AUDIO,
    }
    out: list[dict[str, Any]] = []
    for index, ref in enumerate(ir.get("references", [])):
        ref_type = ref.get("type", "")
        try:
            role = role_of[RefType(ref_type)]
        except (ValueError, KeyError):
            role = SeedanceRole.IMAGE
        out.append({
            "index": index,
            "slots": list(ref.get("slots", [])),
            "type": ref_type,
            "placeholder": ref.get("placeholder", ""),
            "asset_id": ref.get("asset_id"),
            "bound": bool(ref.get("asset_id")),
            "seedance_role": role.value,
            "prep_step": ref.get("prep_step"),
            "prep_hint": ref.get("prep_prompt") or "",
        })
    return out


def apply_targets(ir: Mapping[str, Any]) -> dict[str, list[str]]:
    """反查：输出提示词的每一段，由哪些位置决定。

    这是"给出应用位置"的另一半 —— 用户改某段前能看到它被什么约束着。
    """
    out: dict[str, list[str]] = {}
    for path, _ in fw.flatten(ir):
        section = fw.apply_section_of(path)
        if section:
            out.setdefault(section, []).append(path)
    return out


def spans_of(
    provenance: Mapping[str, Provenance],
    path_prefix: str = "",
) -> list[tuple[str, Span]]:
    """收集带时间码证据的位置，供"点回原片"用（PRD F1.8）。"""
    result: list[tuple[str, Span]] = []
    for path, prov in provenance.items():
        if not path_prefix or path.startswith(path_prefix):
            if prov.span is not None:
                result.append((path, prov.span))
    result.sort(key=lambda item: (item[1].start_ms, item[0]))
    return result
