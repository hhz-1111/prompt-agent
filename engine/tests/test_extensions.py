"""扩展段：框架外信息的落脚点。

用户确认的第 ④ 批。留出集实测：取证已经能读到屏上大字
（「美体普拉提 收紧身形 练出好气质」），但契约里没有位置安放它 —— 读完就丢。
"""

from __future__ import annotations

import pytest

from engine.core import assembler, framework as fw, phrases as ph
from engine.core.contracts import Extension, ReviewNote
from engine.core.review import apply_extensions
from engine.tests.test_core import _sample_ir


@pytest.fixture(scope="module")
def lib():
    return ph.load_library()


def _extension(**kwargs) -> Extension:
    base = {
        "title": "屏上文案",
        "lines": ("大屏写着「美体普拉提」", "下方小字：收紧身形 练出好气质"),
        "evidence": "t=1.2s 取证看到大屏文字",
    }
    base.update(kwargs)
    return Extension(**base)


class TestApplyExtensions:
    def test_accepts_a_grounded_extension(self):
        updated, notes = apply_extensions(_sample_ir(), [_extension()])
        assert notes[0].applied
        stored = updated["creative"]["C3_extensions"]
        assert stored[0]["title"] == "屏上文案"
        assert stored[0]["evidence"].startswith("t=1.2s")
        # 原 IR 不被改动（调用方拿它做回滚）
        assert "C3_extensions" not in _sample_ir()["creative"]

    def test_quoted_screen_text_is_allowed(self):
        """**带引号的屏上文案是正当用途** —— 曾经有一条"不许写引号"的规则，
        被测试当场否掉：它恰好挡住了扩展段最典型的用法。"""
        _updated, notes = apply_extensions(
            _sample_ir(), [_extension(lines=("大屏写着「美体普拉提」",))]
        )
        assert notes[0].applied

    def test_requires_evidence(self):
        _updated, notes = apply_extensions(_sample_ir(), [_extension(evidence="")])
        assert not notes[0].applied
        assert "证据" in notes[0].detail

    def test_rejects_fixed_section_titles(self):
        _updated, notes = apply_extensions(
            _sample_ir(), [_extension(title=fw.SEC_LOCKED)]
        )
        assert not notes[0].applied
        assert "重名" in notes[0].detail

    def test_caps_count_and_length(self):
        extensions = [_extension(title=f"段{i}") for i in range(5)]
        updated, notes = apply_extensions(_sample_ir(), extensions)
        assert len(updated["creative"]["C3_extensions"]) == 3
        assert sum(1 for n in notes if n.applied) == 3
        assert any("上限" in n.detail for n in notes if not n.applied)

        long_one = _extension(title="长段", lines=tuple(["x" * 200] * 5))
        updated, _ = apply_extensions(_sample_ir(), [long_one])
        stored = updated["creative"]["C3_extensions"][0]
        assert len(stored["lines"]) == 3
        assert all(len(line) <= 80 for line in stored["lines"])

    def test_same_title_is_not_duplicated(self):
        updated, _ = apply_extensions(_sample_ir(), [_extension(), _extension()])
        assert len(updated["creative"]["C3_extensions"]) == 1


class TestRendering:
    def test_renders_after_action_and_before_forbidden(self, lib):
        updated, _ = apply_extensions(_sample_ir(), [_extension()])
        # 样例是对话类、没有【动作与物理】，所以显式标成动作类来验证**相对顺序**
        updated["meta"]["content_kind"] = "action"
        doc = assembler.assemble(updated, lib)
        titles = [section.title for section in doc.sections]
        assert fw.SEC_EXTENSIONS in titles
        assert titles.index(fw.SEC_ACTION) < titles.index(fw.SEC_EXTENSIONS)
        assert titles.index(fw.SEC_EXTENSIONS) < titles.index(fw.SEC_FORBIDDEN)

    def test_the_evidence_parenthetical_is_gone(self, lib):
        """**括号里的依据不要出现在交付物里**（用户 2026-10-08）。

        用户原话：「目前你给出的信息太复杂，看不懂 …… 目前你给出的括号里的内容就不要了。」
        依据仍然留在 `evidence`（报告里可追溯），只是不进提示词。
        """
        updated, _ = apply_extensions(_sample_ir(), [_extension()])
        text = assembler.assemble(updated, lib).full_text()
        assert "［扩展］" in text
        assert "依据：" not in text
        assert "定向取证" not in text or "证据" not in text.split("［扩展］")[1][:80]

    def test_no_prefix_before_screen_text(self, lib):
        """屏上文字**直接写在扩展段里**，前面不加任何前缀（用户 2026-10-08 追加）。

        「以下画面文字不要生成（用来理解原片）：」这句也去掉了 ——
        D0 是【禁止项】里的全局约束（「全程不生成任何字幕文字，100%透明字幕。」），
        在扩展段里重复一遍只是噪音。
        """
        updated, _ = apply_extensions(_sample_ir(), [_extension()])
        text = assembler.assemble(updated, lib).full_text()
        assert "以下画面文字不要生成" not in text
        assert "用来理解原片" not in text
        # 文字本身照旧写出来（带「」），而且照样提醒人工确认
        assert "「美体普拉提」" in text
        assert "请人工确认后修改" in text

    def test_the_forbidden_section_is_not_touched(self, lib):
        """**别动【禁止项】** —— 用户 2026-10-08 明确说过（他指的是扩展段那句前缀）。"""
        ir = _sample_ir()
        before = [dict(item) for item in ir.get("forbidden", [])]
        updated, _ = apply_extensions(ir, [_extension()])
        assert [dict(item) for item in updated.get("forbidden", [])] == before

    def test_our_own_analysis_is_filtered_out(self, lib):
        """**我们自己的判断口径不进交付物**（用户看着"看不懂"的正是这些）。

        实测那条扩展段里混着「经取证为完整单画面，屏幕属场景内元素，非字幕、分屏或画中画」。
        """
        extension = _extension()
        extension = type(extension)(
            title=extension.title,
            lines=extension.lines + ("经取证为完整单画面，屏幕属场景内元素，非字幕、分屏或画中画。",),
            evidence=extension.evidence,
        )
        updated, _ = apply_extensions(_sample_ir(), [extension])
        text = assembler.assemble(updated, lib).full_text()
        assert "经取证" not in text
        assert "属场景内元素" not in text
        assert "分屏" not in text

    def test_unreadable_parts_ask_the_human_to_confirm(self, lib):
        """**没看清的部分要说清是什么、为什么，并请人工确认**（用户 2026-10-08）。"""
        from engine.core.contracts import Extension

        updated, _ = apply_extensions(_sample_ir(), [Extension(
            title="背景屏文字",
            lines=("背景屏上写着黄橙色大字「5天普拉提训练营」。",),
            evidence="3 秒处 top 区域取证",
            unreadable="屏幕右下角还有一行小字，画面分辨率不够、字太小，辨认不出内容",
        )])
        text = assembler.assemble(updated, lib).full_text()
        assert "5天普拉提训练营" in text
        assert "辨认不出内容" in text
        assert "请人工确认后修改" in text

    def test_no_extensions_means_no_section(self, lib):
        doc = assembler.assemble(_sample_ir(), lib)
        assert fw.SEC_EXTENSIONS not in [s.title for s in doc.sections]


class TestContract:
    def test_extension_fields_are_declared(self):
        for path in (
            "creative.C3_extensions",
            "creative.C3_extensions[].title",
            "creative.C3_extensions[].lines",
            "creative.C3_extensions[].evidence",
        ):
            assert fw.spec_for(path) is not None, path

    def test_extension_is_writable_by_the_user(self):
        """扩展段属于**创作区**：用户能改能删（它不是框架的硬约束）。"""
        spec = fw.spec_for("creative.C3_extensions")
        assert spec.category.value == "FREE"

    def test_review_notes_are_reported(self):
        _updated, notes = apply_extensions(_sample_ir(), [_extension(evidence="")])
        assert isinstance(notes[0], ReviewNote)
        assert notes[0].action == "extend"
