"""短语工程：约束有效性分级、强约束降级检测、必写维度覆盖。

PRD §6.3 说得很清楚：

    只有结构能力，Agent 是"会填槽的执行者"；
    加上短语能力，它才是"会写提示词的专家"。

本模块是那套"手感"的可执行形式。三条铁律：

1. **不可以把强约束降级成弱约束**（把 ``1:1复刻`` 改成 ``参考一下`` 会直接导致人物漂移）；
2. **必然有 E 级（泛化）与 F 级（愿望式）表述** —— 出现即阻断；
3. **必写维度缺失即告警** —— 眨眼与皮肤在 100% 样本中出现，不是习惯，是模型默认会做错的地方。

资产本身在 ``assets/phrases/phrases.json``，本模块只做判定，不存数据。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import framework as fw
from .types import CONSTRAINT_GRADE_RANK, WEAK_GRADES, Grade, UNKNOWN

DEFAULT_ASSET_PATH = Path(__file__).resolve().parents[2] / "assets" / "phrases" / "phrases.json"


# --------------------------------------------------------------------------
# 资产目录
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PhraseAsset:
    """一条短语资产。``tier`` 决定改写自由度。"""

    key: str
    label: str
    dimension: str
    tier: str
    source: str = ""
    sentence: str = ""
    variants: Mapping[str, str] | None = None
    mandatory_fragments: tuple[str, ...] = ()
    note: str = ""

    @property
    def is_fixed(self) -> bool:
        """必须逐字：任何改动都是违规。"""
        return self.tier == "fixed"

    def render(self, **slots: str) -> str:
        """填槽。缺槽会显式抛错，绝不静默留空。"""
        if not self.sentence:
            raise ValueError(f"{self.key} 没有可渲染句式（tier={self.tier}）")
        missing = [
            name for name in re.findall(r"\{([^}]+)\}", self.sentence)
            if name not in slots
        ]
        if missing:
            raise ValueError(f"{self.key} 缺少槽位：{missing}")
        return self.sentence.format(**slots)

    def variant(self, name: str) -> str:
        if not self.variants or name not in self.variants:
            raise KeyError(f"{self.key} 无变体 {name}；可选：{list(self.variants or {})}")
        return self.variants[name]


class PhraseLibrary:
    """短语资产库的门面。索引一次，供全应用复用。"""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.raw = raw
        self.version: str = raw.get("phrase_lib_version", "")
        self.framework_version: str = raw.get("framework_version", "")
        self.capacity: Mapping[str, Any] = raw.get("capacity", {})
        self.speech_rate: Mapping[str, Any] = raw.get("speech_rate", {})

        self.templates: dict[str, PhraseAsset] = {
            key: _asset_from_template(key, value)
            for key, value in raw.get("templates", {}).items()
        }
        self.dimensions: dict[str, dict[str, Any]] = {
            dim["key"]: dim for dim in raw.get("dimensions", [])
        }
        self.mandatory_dimensions: tuple[str, ...] = tuple(raw.get("mandatory_dimensions", []))
        self.forbidden_groups: Mapping[str, Any] = raw.get("forbidden_groups", {})
        self.downgrade_rules: tuple[Mapping[str, Any], ...] = tuple(raw.get("downgrade_rules", []))
        self._grade_patterns = [
            (Grade(item["grade"]), re.compile(item["pattern"]))
            for item in raw.get("grade_patterns", [])
        ]
        self._compiled_downgrades = [
            (
                rule,
                re.compile(
                    _alternation(rule["downgrades"])
                ),
            )
            for rule in self.downgrade_rules
        ]

    # ---- 禁止项 ----
    def forbidden_items(self, groups: Sequence[str] | None = None) -> list[dict[str, str]]:
        """取禁止项条文。不给 groups 时取默认全选组（D0 全选）。"""
        selected = groups
        if selected is None:
            selected = [
                key for key, group in self.forbidden_groups.items() if group.get("default_on")
            ]
        items: list[dict[str, str]] = []
        for group_key in selected:
            group = self.forbidden_groups.get(group_key)
            if not group:
                continue
            for index, text in enumerate(group["items"], start=1):
                items.append({
                    "id": f"{group_key}.{index}",
                    "group": group_key,
                    "text": text,
                })
        return items

    def default_forbidden_ids(self) -> list[str]:
        return [item["id"] for item in self.forbidden_items()]

    # ---- 维度 ----
    def dimension(self, key: str) -> Mapping[str, Any]:
        try:
            return self.dimensions[key]
        except KeyError as exc:  # pragma: no cover - 配置错误
            raise KeyError(f"短语库中不存在维度 {key}") from exc

    def vocabulary(self, key: str) -> tuple[str, ...]:
        return tuple(self.dimension(key).get("vocab", ()))

    def capacity_for(self, model: str) -> Mapping[str, Any]:
        models = self.capacity.get("models", {})
        if model not in models:
            raise KeyError(f"能力矩阵中不存在模型 {model}")
        return models[model]

    # ---- 判定 ----
    def grade(self, text: str) -> tuple[Grade, str]:
        """约束表述有效性分级。返回 (级别, 命中片段)。

        规则有三条，缺一条就会判歪：

        1. **先判弱**：E/F 级一旦出现，整句就是弱的 —— 一句泛化表述会污染
           整条约束，不能因为别处有量化就被"平均"掉。
        2. **P0 级别优先**（PRD §6.3.4）：命中 D 级枚举时即便同时出现
           ``100%``，也判 D —— 人写的是纯禁止清单，没有正向目标态。
        3. **取最强**：其余情况取声明强度最高的那一级。
        """
        if not text or not text.strip():
            return Grade.E_GENERIC, ""
        probe = strip_quoted(text)
        hits: dict[Grade, str] = {}
        for grade, pattern in self._grade_patterns:
            match = pattern.search(probe)
            if match:
                hits[grade] = match.group(0)
        for grade in (Grade.F_WEAK, Grade.E_GENERIC):
            if grade in hits:
                return grade, hits[grade]
        if Grade.D_ENUMERATED in hits:
            return Grade.D_ENUMERATED, hits[Grade.D_ENUMERATED]
        if hits:
            best = max(hits, key=lambda g: CONSTRAINT_GRADE_RANK[g])
            return best, hits[best]
        return Grade.D_ENUMERATED, ""

    def is_weak(self, text: str) -> bool:
        return self.grade(text)[0] in WEAK_GRADES

    def detect_downgrades(self, texts: Iterable[str]) -> list["PhraseIssue"]:
        """强约束降级检测（PRD F3.5）。

        判据是**「弱化替换出现在非受保护语境」**。两个容易踩的坑：

        1. 原文还在（``1:1复刻…参考一下也行``）算并存描述，不报；
        2. ``参考`` 这类短词必须加边界：``形象参考[图1]``、``音色参考[音频1]``
           是**规范强制写法**，绝不是降级。所以替换词按"后面不能再跟汉字"匹配。
        """
        issues: list[PhraseIssue] = []
        for text in texts:
            if not text:
                continue
            for rule, pattern in self._compiled_downgrades:
                hit = pattern.search(text)
                if not hit:
                    continue
                if rule["protected"] in text:
                    continue
                issues.append(PhraseIssue(
                    code="F3.5",
                    severity="block",
                    title=f"强约束被降级：{rule['protected']} → {hit.group(0)}",
                    detail=rule["why"],
                    suggestion=f"改回逐字表述「{rule['protected']}」；该短语在 ≥4 份样本中逐字一致。",
                    evidence=[text, hit.group(0)],
                ))
        return _dedupe(issues)

    def suggest_upgrade(self, text: str) -> str:
        """把 E/F 级表述升级成 A–D 级的具体建议。"""
        if not text:
            return "补写一条带具体对象的约束"
        grade, hit = self.grade(text)
        if grade not in WEAK_GRADES:
            return ""
        if hit:
            return (
                f"「{hit}」属于{'泛化' if grade is Grade.E_GENERIC else '愿望式'}表述，"
                f"约束无效：改成带具体对象或量化的写法"
                f"（如「{text.replace(hit, '100%相同')}」）"
            )
        return "改写为带具体对象或量化的表述"


@dataclass(frozen=True, slots=True)
class PhraseIssue:
    """短语级问题。``severity``: block / warn / info。"""

    code: str
    severity: str
    title: str
    detail: str = ""
    suggestion: str = ""
    path: str = ""
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "suggestion": self.suggestion,
            "path": self.path,
            "evidence": list(self.evidence),
        }


_POSSESSIVE = "我你他她它这那该本此"


def _boundary_pattern(word: str) -> str:
    """给替换词加边界，避免误报（误报一次，用户就再也不信这个闸门）。

    1. **后面不能直接跟指代词**：``参考一下我的动作`` 是在说"参照我的动作"，
       不是把 ``1:1复刻`` 降级；``参考一下原片`` 才是降级；
    2. **短词结尾不能再跟汉字**：``类似`` 不该命中 ``类似度``。
    """
    tail = rf"(?!\s*[{_POSSESSIVE}])"
    if len(word) >= 4:
        return re.escape(word) + tail
    return re.escape(word) + r"(?![\u4e00-\u9fff])" + tail


def _alternation(words: Iterable[str]) -> str:
    """拼正则交替。

    **必须按长度降序**：``参考|参考一下`` 里的短项会先匹配上，
    长项永远轮不到 —— 导致 ``动作参考一下原片`` 被漏检
    （它恰好是最典型的降级写法）。
    """
    ordered = sorted(set(words), key=len, reverse=True)
    return "|".join(_boundary_pattern(word) for word in ordered)


def _asset_from_template(key: str, raw: Mapping[str, Any]) -> PhraseAsset:
    variants = raw.get("variants")
    return PhraseAsset(
        key=key,
        label=raw.get("label", key),
        dimension=raw.get("dimension", ""),
        tier=raw.get("tier", "template"),
        source=raw.get("source", ""),
        sentence=raw.get("sentence", ""),
        variants=dict(variants) if variants else None,
        mandatory_fragments=tuple(raw.get("mandatory_fragments", ())),
    )


def _dedupe(issues: Sequence[PhraseIssue]) -> list[PhraseIssue]:
    seen: set[tuple[str, str]] = set()
    out: list[PhraseIssue] = []
    for issue in issues:
        marker = (issue.code, issue.title)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(issue)
    return out


# 被引号/方括号包起来的内容是**提及**而非**主张**。
# 例：D4 那条禁止项写作「禁止补充「服装保持一致」「参考图延续」等泛化一致性描述」
# —— 它在*引用*要被禁掉的说法，本身不是泛化约束，不能误报。
#
# **ASCII 双引号必须在列表里**（实测踩过）：组装器包台词用的就是 `"…"`
# （``assembler.ensure_quoted``，方舟官方建议的写法）。不认它的话，
# 台词里的「你照着做就行」会被当成正文主张，G14 直接 block：
# 「强约束被降级：1:1复刻 → 照着做就行」——用户一句正常台词就让整条片子不可交付。
_QUOTED_SPAN = re.compile(
    r"「[^」]*」|“[^”]*”|『[^』]*』|\[[^\]]*\]|【[^】]*】|\"[^\"]*\""
)


def strip_quoted(text: str) -> str:
    return _QUOTED_SPAN.sub("", text)


# --------------------------------------------------------------------------
# 必写维度覆盖（框架 §11 G13）
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DimensionCoverage:
    key: str
    label: str
    covered: bool
    paths: tuple[str, ...]
    missing_fragments: tuple[str, ...] = ()
    suggestion: str = ""
    implied: bool = False
    """靠素材句柄隐含成立，而非逐字写出（例如人物一致性由「形象参考[图1]」兜住）。

    这类维度在 UI 上要提示用户补逐字表述 —— 它是**最弱**的成立方式。
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "covered": self.covered,
            "implied": self.implied,
            "paths": list(self.paths),
            "missing_fragments": list(self.missing_fragments),
            "suggestion": self.suggestion,
        }


# 素材句柄会隐含这些维度的成立。判据是 PRD 对"必写维度"的定义：
# **逐字表述必须显式钉住**，只靠引用素材不算合格。
IMPLICIT_FRAGMENTS: Mapping[str, tuple[str, ...]] = {
    "person_consistency": ("形象参考",),
    "audio": ("音色参考",),
    "action": ("1:1复刻", "复刻"),
    "scene": ("场景始终为", "参考"),
}


def _missing_fragments(
    mandatory: Sequence[Any],
    haystack: str,
) -> tuple[str, ...]:
    """检查必写片段。

    每个元素有两种形态：

    - ``"锁定人物五官脸型"``：字符串 → **必须出现**；
    - ``["一镜到底", "机位全程不变"]``：列表 → **命中任一即可**。

    为什么需要第二种：原文里 ``一镜到底`` / ``机位全程不变`` / ``镜头无跳切``
    是等价表述（取自不同样本），要求三个都出现是错的。
    """
    missing: list[str] = []
    for entry in mandatory:
        if isinstance(entry, (list, tuple)):
            if not any(f in haystack for f in entry):
                missing.append("／".join(entry))
        elif entry not in haystack:
            missing.append(str(entry))
    return tuple(missing)


def check_mandatory_dimensions(
    ir: Mapping[str, Any],
    library: PhraseLibrary,
    *,
    prompt_text: str | None = None,
) -> list[DimensionCoverage]:
    """逐维度检查必写覆盖。**只做检查，不代写**（红线 3）。

    ``prompt_text`` 是**组装后的提示词正文**。这一点很关键：
    IR 里 ``blink`` 存的是键名 ``"5s"``，逐字表述（``自然快速``、
    ``每次约0.1秒``）是组装时才展开的。所以片段检查必须审最终文本，
    只审 IR 会把合规的提示词判成缺维度。

    判据有两条，缺一不可：

    1. **逐字片段出现在最终提示词里**；
    2. **该维度在 IR 上确实有内容** —— 否则没有 mandatory 片段的维度
       （景别/机位/情绪）会在空 IR 上被判成"覆盖"。

    例外（第 3 条）：**有逐字片段、且片段全部命中时，不再要求 IR 有内容**。
    为什么（实测踩过）：眨眼节拍与皮肤质感是**无条件固定句**，没有人物条目时
    正文里照样写着，而它们的 IR 内容挂在 ``locked.B2_person[]`` 下 ——
    只看 IR 就会报「缺：眨眼节拍 / 整条未反推到」，可交付物里明明有这两句。
    交付物才是用户真正拿去生成的东西，逐字要求以它为准。
    """
    leaves = dict(fw.flatten(ir))
    haystack = prompt_text if prompt_text is not None else "\n".join(_collect_text(ir))
    out: list[DimensionCoverage] = []
    for key in library.mandatory_dimensions:
        dim = library.dimension(key)
        mandatory = tuple(dim.get("mandatory", ()))
        missing = _missing_fragments(mandatory, haystack)
        has_content = _dimension_has_content(leaves, dim)
        if mandatory and not missing:
            # 逐字表述全在交付物里 → 覆盖成立（见 docstring 第 3 条例外）
            covered, implied = True, False
        elif has_content and not missing:
            covered, implied = True, False
        elif not has_content:
            covered, implied = False, False
        else:
            implicit = IMPLICIT_FRAGMENTS.get(key, ())
            implied = any(f in haystack for f in implicit)
            covered = implied
        out.append(DimensionCoverage(
            key=key,
            label=dim["label"],
            covered=covered,
            paths=tuple(dim.get("ir_paths", ())),
            missing_fragments=missing,
            suggestion=_dimension_suggestion(dim, missing, implied, has_content),
        ))
    return out


def _dimension_has_content(
    leaves: Mapping[str, Any],
    dim: Mapping[str, Any],
) -> bool:
    """该维度在自己的 IR 路径上是否真的有值。

    排除 UNKNOWN：那是"没反推出来"，不是"写了"（红线 7「不猜」）。
    """
    templates = [fw.normalize_path(p) for p in dim.get("ir_paths", [])]
    for path, value in leaves.items():
        normalized = fw.normalize_path(path)
        matched = normalized in templates or any(
            normalized.startswith(t + ".") or normalized.startswith(t + "[")
            for t in templates
        ) or any(t.startswith(normalized + ".") for t in templates)
        if not matched:
            continue
        for text in _collect_text(value):
            if text.strip() and text.strip() != UNKNOWN:
                return True
    return False


def _collect_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [] if value == UNKNOWN else [value]
    if isinstance(value, Mapping):
        out: list[str] = []
        for item in value.values():
            out.extend(_collect_text(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_collect_text(item))
        return out
    return []


def _dimension_suggestion(
    dim: Mapping[str, Any],
    missing: Sequence[str],
    implied: bool,
    has_content: bool,
) -> str:
    if not missing and has_content:
        return ""
    note = dim.get("note", "跨样本一致性 ≥4 份")
    coverage = dim.get("evidence_hit") or "—"
    if not has_content:
        return f"「{dim['label']}」整条未反推到。样本覆盖率 {coverage}。依据：{note}"
    if implied:
        return (
            f"「{dim['label']}」目前只靠素材引用隐含成立，缺逐字必写表述："
            f"{'、'.join(missing)}。补齐后约束才真正钉住。依据：{note}"
        )
    return f"「{dim['label']}」缺逐字必写表述：{'、'.join(missing)}。依据：{note}"


# --------------------------------------------------------------------------
# 全文扫描：把闸门 G14 需要的输入一次算齐
# --------------------------------------------------------------------------
CONSTRAINT_SECTIONS: frozenset[str] = frozenset({
    "参数声明", "参考素材", "关键一致性", "镜头节拍", "禁止项",
})
"""E/F 级模糊判据（``尽量``/``保持一致``）只体检这些段落。

**刻意排除** 剧情与对白：那是用户的创作表达，
在台词里说"你先参考一下我的动作"是正常说话，不是把 ``1:1复刻`` 降级。
把创作区拿去做模糊判据体检会产生大量误报，最终让用户不再信任闸门。

注意**镜头节拍要留下**：``1:1复刻``、``不出现任何其他动作`` 这类
强约束正是写在节拍里的，排除掉就会漏检最关键的降级。
"""


def scan_texts(
    library: PhraseLibrary,
    lines: Iterable[str] | Iterable[tuple[str, str]],
    *,
    strict_sections_only: bool = True,
) -> list[PhraseIssue]:
    """对提示词做短语级体检：E/F 级 + 强约束降级。

    两类判据的误报风险不同，所以覆盖面也应该不同：

    - **降级检测**用的是精确短语 + 指代词/汉字边界，误报风险低，
      因此扫**全部**段落 —— 否则节拍里的 ``1:1复刻`` 降级会被漏掉；
    - **E/F 级**用 ``尽量``/``保持一致`` 这类模糊词，在台词里是正常语言，
      只扫约束段落。
    """
    materialized: list[tuple[str, str]] = []
    for item in lines:
        section, text = item if isinstance(item, tuple) else ("", str(item))
        if text and text.strip():
            materialized.append((section, text))

    issues = library.detect_downgrades([strip_quoted(text) for _, text in materialized])
    for section, text in materialized:
        if strict_sections_only and section and section not in CONSTRAINT_SECTIONS:
            continue
        grade, hit = library.grade(strip_quoted(text))
        if grade in WEAK_GRADES:
            issues.append(PhraseIssue(
                code="F3.4",
                severity="block",
                title=f"出现 {grade.value} 级约束表述：{hit or text[:20]}",
                detail="E 级（泛化）与 F 级（愿望式）在成功样本中不出现，等于没写约束。",
                suggestion=library.suggest_upgrade(text),
                path=section,
                evidence=[text],
            ))
    return _dedupe(issues)


# --------------------------------------------------------------------------
# 单例装载
# --------------------------------------------------------------------------
_cache: dict[str, PhraseLibrary] = {}


def load_library(path: str | Path | None = None) -> PhraseLibrary:
    """装载短语库。默认读 assets/phrases/phrases.json，结果缓存。"""
    target = Path(path) if path else DEFAULT_ASSET_PATH
    key = str(target)
    if key not in _cache:
        raw = json.loads(target.read_text(encoding="utf-8"))
        _cache[key] = PhraseLibrary(raw)
    return _cache[key]
