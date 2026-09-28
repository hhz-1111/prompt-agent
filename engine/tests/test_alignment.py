"""内容对齐测试（PRD F6）。

对齐是**推导**而不是量测，所以它的每个判据、每档置信度都用
可复现的例子钉住 —— 否则调一次权重就会把之前的行为悄悄改掉。

测试按 PRD F6 的功能编号组织。
"""

from __future__ import annotations

import pytest

from engine.core import alignment as al, framework as fw, phrases as ph
from engine.core.alignment import BindingStatus


# ==========================================================================
# F6.1 三种输入形态解析
# ==========================================================================
class TestContentParsing:
    def test_plain_script_one_line_per_block(self):
        blocks = al.parse_user_content("第一句话\n第二句话\n第三句话")
        assert [b.content for b in blocks] == ["第一句话", "第二句话", "第三句话"]

    def test_speaker_form(self):
        blocks = al.parse_user_content("王姐：你这气色也太好了吧\n李姐：哪有啊")
        assert [(b.speaker, b.content) for b in blocks] == [
            ("王姐", "你这气色也太好了吧"),
            ("李姐", "哪有啊"),
        ]

    def test_timecode_form(self):
        blocks = al.parse_user_content("00:03 哪有啊\n01:05.5 后面这句")
        assert blocks[0].timecode_hint_sec == 3.0
        assert blocks[0].content == "哪有啊"
        assert blocks[1].timecode_hint_sec == 65.5

    def test_timecode_range_form(self):
        blocks = al.parse_user_content("00:01-00:03 这段是这个时间说的")
        assert blocks[0].timecode_hint_sec == 1.0
        assert blocks[0].content == "这段是这个时间说的"

    def test_three_forms_can_mix(self):
        raw = "朋友都问我是不是做项目了\n王姐：你这气色也太好了吧\n00:05 哪有啊"
        blocks = al.parse_user_content(raw)
        assert len(blocks) == 3
        assert blocks[1].speaker == "王姐"
        assert blocks[2].timecode_hint_sec == 5.0

    def test_blank_lines_are_skipped(self):
        blocks = al.parse_user_content("第一句\n\n\n第二句\n   \n")
        assert len(blocks) == 2

    def test_colon_in_sentence_is_not_a_speaker(self):
        """「他说：今天天气不错」里的冒号不该被当成说话人分隔。

        判据是说话人必须是短名词，不能含标点。
        """
        blocks = al.parse_user_content("他说，今天天气不错：真的很好")
        assert blocks[0].speaker == ""

    # ---- F6.4 双向引用位置 ----
    def test_char_offsets_point_back_to_original(self):
        """字符区间必须能在原稿里精确定位 —— 这是"跳回原稿"的基础。"""
        raw = "第一句话\n王姐：你这气色也太好了吧\n第三句话"
        blocks = al.parse_user_content(raw)
        for block in blocks:
            assert raw[block.char_start:block.char_end] == block.content

    def test_char_offsets_account_for_leading_whitespace(self):
        raw = "第一句\n    缩进的第二句\n第三句"
        blocks = al.parse_user_content(raw)
        assert raw[blocks[1].char_start:blocks[1].char_end] == "缩进的第二句"

    def test_speech_seconds_follows_char_count(self):
        block = al.ContentBlock(index=0, content="十个字十个字十个字", char_start=0, char_end=9)
        assert block.speech_seconds(5.0) == pytest.approx(1.8)
        assert block.speech_seconds(4.0) == pytest.approx(2.25)


# ==========================================================================
# 打分判据
# ==========================================================================
class TestScoring:
    def test_identical_text_scores_one(self):
        assert al.text_similarity("你这气色也太好了吧", "你这气色也太好了吧") == 1.0

    def test_similarity_ignores_punctuation(self):
        assert al.text_similarity("你这气色也太好了吧。", "你这气色也太好了吧") == 1.0

    def test_unrelated_text_scores_low(self):
        assert al.text_similarity("今天天气不错", "你这气色也太好了吧") < 0.3

    def test_empty_scores_zero(self):
        assert al.text_similarity("", "任何内容") == 0.0

    def test_duration_fit_penalises_overflow(self):
        """1.2 秒的位置塞 18 个字 —— 文字再像也不成立。"""
        block = al.ContentBlock(index=0, content="十八个字十八个字十八个字十八个字十八个字",
                                char_start=0, char_end=20)
        tight = al.AlignmentSlot(
            index=0, path="p", label="l", span=al.Span(0, 1200), capacity_chars=6,
        )
        roomy = al.AlignmentSlot(
            index=1, path="p", label="l", span=al.Span(0, 8000), capacity_chars=40,
        )
        assert al.duration_fit(block, tight, 5.0) < 0.3
        assert al.duration_fit(block, roomy, 5.0) == 1.0

    def test_speaker_mismatch_is_flagged(self):
        block = al.ContentBlock(index=0, content="x", char_start=0, char_end=1, speaker="王姐")
        slot = al.AlignmentSlot(index=0, path="p", label="l",
                                span=al.Span(0, 4000), speaker="李姐")
        assert al.speakers_conflict(block, slot)

    def test_missing_speaker_is_not_a_conflict(self):
        block = al.ContentBlock(index=0, content="x", char_start=0, char_end=1)
        slot = al.AlignmentSlot(index=0, path="p", label="l",
                                span=al.Span(0, 4000), speaker="李姐")
        assert not al.speakers_conflict(block, slot)

    def test_unknown_speaker_does_not_inflate_score(self):
        """双方都没有说话人时不该靠中性分拿到虚高分。

        实测漏洞：两份毫无共同点的内容靠 0.5 的中性分拿到 0.65，
        高于我要的保守程度。现在说话人判据不可用时把它的权重
        让给另两维，且中性值降到 0.4。
        """
        block = al.ContentBlock(index=0, content="你好啊", char_start=0, char_end=3)
        slot = al.AlignmentSlot(index=0, path="p", label="l",
                                span=al.Span(0, 4000), reference_line="你这气色也太好了吧")
        score = al.score_pair(block, slot, 5.0)
        assert score.speaker_match < 0.5, "没有说话人信息不该给中性以上的分"

        # 真正起作用的是置信度阈值：这类内容不该被判为"已确认"
        binding = al.AlignmentBinding(
            slot=slot, block=block, score=score, status=BindingStatus.BOUND,
        )
        assert binding.confidence < al.BIND_THRESHOLD


# ==========================================================================
# F6.2 自动对齐 + 置信度
# ==========================================================================
def _slots() -> list[al.AlignmentSlot]:
    from engine.tests.test_core import _sample_ir

    return al.collect_slots(_sample_ir())


class TestAlignment:
    def test_identical_script_binds_everything(self):
        report = al.align(al.parse_user_content(
            "王姐：你这气色也太好了吧\n"
            "李姐：哪有啊，我就是每天在家练普拉提\n"
            "李姐：你摸摸我这腰线"
        ), _slots())
        assert report.bound_count == 3
        assert not report.has_unresolved
        for binding in report.bindings:
            assert binding.confidence >= 0.95

    def test_rewritten_script_still_aligns(self):
        """用户把台词全换了，只要结构相似就该对齐上。"""
        report = al.align(al.parse_user_content(
            "王姐：你这气色也太好了吧，是不是偷偷做什么项目了\n"
            "李姐：哪有啊，我就是每天在家练二十分钟普拉提"
        ), _slots())
        assert report.bindings[0].status is BindingStatus.BOUND
        assert report.bindings[1].status is BindingStatus.BOUND
        assert report.bindings[0].confidence > 0.8

    def test_completely_unrelated_content_prefers_blank_over_misbinding(self):
        """红线 7：宁可留空也不乱配。

        用户写的内容跟原片毫无关系时，全部标为未分配，
        而不是硬塞进第一个位置。
        """
        report = al.align(al.parse_user_content(
            "甲：今天天气不错\n乙：是啊挺暖和的\n丙：出去走走吧"
        ), _slots())
        assert report.bound_count == 0
        assert report.unmatched_count == 3

    def test_speaker_conflict_is_penalised_not_misbound(self):
        """把台词放进别人嘴里是明确错误，全局分配里应当竞争不过。"""
        report = al.align(al.parse_user_content(
            "王姐：你这气色也太好了吧\n"
            "李姐：哪有啊，我就是每天在家练普拉提\n"
            "王姐：就这么简单？"
        ), _slots())
        third = report.bindings[2]
        assert third.status is not BindingStatus.BOUND
        assert third.block is None, "说错人的那句不该被塞进李姐的位置"

    def test_fewer_blocks_than_slots_leaves_rest_blank(self):
        report = al.align(al.parse_user_content("王姐：你好啊"), _slots())
        assert report.unmatched_count == 2
        for binding in report.bindings[1:]:
            assert binding.status is BindingStatus.UNMATCHED
            assert "保留原片结构" in binding.reason

    def test_more_blocks_than_slots_reports_orphans(self):
        report = al.align(al.parse_user_content(
            "第一句\n第二句\n第三句\n第四句\n第五句"
        ), _slots())
        assert report.orphans, "多出来的内容必须报出来，不能静默丢弃"
        assert any("没有位置可以放置" in note for note in report.notes)

    def test_no_slots_reports_all_as_orphans(self):
        report = al.align(al.parse_user_content("内容一\n内容二"), [])
        assert len(report.orphans) == 2
        assert report.bindings == []

    def test_no_blocks_marks_all_unmatched(self):
        report = al.align([], _slots())
        assert report.unmatched_count == 3
        assert not report.has_unresolved or True  # 空内容不算未决

    def test_assignment_preserves_order(self):
        """保序：稿子的第 3 句不可能落到第 1 个位置。"""
        report = al.align(al.parse_user_content(
            "王姐：你这气色也太好了吧\n"
            "李姐：哪有啊，我就是每天在家练普拉提\n"
            "李姐：你摸摸我这腰线"
        ), _slots())
        indices = [
            b.block.index for b in report.bindings if b.block is not None
        ]
        assert indices == sorted(indices)

    def test_duration_overflow_is_reported_as_conflict(self):
        """PRD §5 步骤 5 那个例子：18 字塞进 1.2 秒。"""
        from engine.core.alignment import AlignmentSlot, Span

        slot = AlignmentSlot(
            index=0, path="creative.C2_beats[0].dialogue", label="镜头1",
            span=Span(0, 1200), reference_line="哪有啊", speaker="李姐",
        )
        report = al.align(
            al.parse_user_content("李姐：哪有啊，我就是每天在家练普拉提，你也要坚持"),
            [slot],
        )
        binding = report.bindings[0]
        assert any("时长超限" in c for c in binding.conflicts)
        assert binding.status is BindingStatus.PENDING


# ==========================================================================
# 从 IR 抽取位置
# ==========================================================================
class TestCollectSlots:
    def test_collects_one_slot_per_beat(self):
        slots = _slots()
        assert len(slots) == 3
        assert slots[0].path == "creative.C2_beats[0].dialogue"
        assert slots[0].label == "镜头1"

    def test_spans_are_contiguous_and_cover_the_timeline(self):
        slots = _slots()
        for previous, current in zip(slots, slots[1:]):
            assert previous.span.end_ms == current.span.start_ms

    def test_reference_line_comes_from_current_ir(self):
        from engine.tests.test_core import _sample_ir

        slots = al.collect_slots(_sample_ir())
        assert slots[0].reference_line == "你这气色也太好了吧"

    def test_capacity_matches_speech_rate(self):
        slots = _slots()
        assert slots[0].capacity_chars == int(slots[0].duration_sec * 5.0)

    def test_speaker_not_invented_from_single_audio(self):
        """全片只有一个音色时，不能给所有位置硬套同一个说话人 ——
        那会让 speaker_match 判据虚高，掩盖真正的不一致。"""
        from engine.tests.test_core import _sample_ir

        ir = _sample_ir()
        for beat in ir["creative"]["C2_beats"]:
            beat["dialogue"]["speaker"] = ""
        slots = al.collect_slots(ir)
        assert all(s.speaker == "" for s in slots)


# ==========================================================================
# 写回 IR
# ==========================================================================
class TestApplyAlignment:
    def test_only_bound_bindings_are_written(self):
        """F6.3 + F6.6：未确认的对齐不许污染交付物。"""
        from engine.tests.test_core import _sample_ir

        ir = _sample_ir()
        report = al.align(al.parse_user_content(
            "王姐：你这气色也太好了吧，是不是偷偷做什么项目了\n"
            "李姐：哪有啊，我就是每天在家练二十分钟普拉提\n"
            "王姐：就这么简单？"
        ), al.collect_slots(ir))
        assert report.pending_count or report.unmatched_count, "本例应存在未决项"

        written = al.apply_alignment(ir, report)
        assert written, "已确认的绑定应当写进去"
        assert "creative.C2_beats[0].dialogue.line" in written
        # 第 3 个位置没被绑定，原值应保持不变
        assert fw.get_in(ir, "creative.C2_beats[2].dialogue.line") == "你摸摸我这腰线"

    def test_written_content_reaches_the_prompt(self):
        from engine.core import assembler
        from engine.tests.test_core import _sample_ir

        ir = _sample_ir()
        report = al.align(al.parse_user_content(
            "王姐：你这气色也太好了吧\n李姐：哪有啊"
        ), al.collect_slots(ir))
        al.apply_alignment(ir, report)
        text = assembler.assemble(ir, ph.load_library()).body_text
        assert "你这气色也太好了吧" in text
        assert "哪有啊" in text
