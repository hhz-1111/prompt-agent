"""⑤ 结构化提问 + 致命维度多源裁决。

两件事都是"把 Agent 的判断交给人或交给第三方证据"，所以纪律特别重要：
**能取证就别问人、问人要问得能答、裁决不能偷偷改数据**。
"""

from __future__ import annotations

from engine.app.pipeline import arbitration as AR
from engine.core import review as review_core
from engine.core.contracts import (
    Conflict,
    ProbeFinding,
    Question,
    ReviewNote,
    ShotBoundary,
)


def _count_conflict(cv: str = "3 人（人体框）", model: str = "1 人（浅蓝女子）") -> Conflict:
    return Conflict(
        field=AR.COUNT_FIELD, shot_index=1, cv_value=cv, model_value=model,
        resolution="以模型为准", confidence=0.6,
    )


def _shots(count: int = 1) -> list[ShotBoundary]:
    return [
        ShotBoundary(index=i, start_ms=(i - 1) * 2000, end_ms=i * 2000)
        for i in range(1, count + 1)
    ]


class _Probe:
    def __init__(self, answer: str, *, fail: bool = False):
        self.answer = answer
        self.fail = fail
        self.asked: list = []

    def look(self, video_path, request):
        self.asked.append(request)
        if self.fail:
            raise RuntimeError("抽帧失败")
        return ProbeFinding(request=request, answer=self.answer)


class TestCriticalConflictFilter:
    def test_only_counts_are_arbitrated(self):
        """只对**人数**这种致命维度花一次取证；景别那种"两边都说得通"的不动。

        依据是本轮的一次实测：景别上"最大框"与模型的分歧，抽帧后我人工看
        也判断不了谁更对 —— 所以不浪费预算去"裁决"它。
        """
        conflicts = [
            _count_conflict(),
            Conflict(field="creative.C2_beats[].framing", shot_index=1,
                     cv_value="中景", model_value="近景", resolution="x", confidence=0.6),
        ]
        assert len(AR.critical_conflicts(conflicts)) == 1


class TestArbitration:
    def test_probe_agrees_with_model(self):
        notes: list[ReviewNote] = []
        degradations: list[str] = []
        out = AR.arbitrate_conflicts(
            [_count_conflict()], _shots(), _Probe("1"), "v.mp4",
            budget=2, notes=notes, degradations=degradations,
        )
        assert "与模型一致" in out[0].resolution
        assert notes[0].action == "arbitrate" and notes[0].applied
        assert not degradations

    def test_probe_agrees_with_cv_warns_the_user(self):
        """取证站在检测框那边时**必须提醒**：正文人数句取的是模型值，可能偏。"""
        degradations: list[str] = []
        out = AR.arbitrate_conflicts(
            [_count_conflict()], _shots(), _Probe("3"), "v.mp4",
            budget=2, notes=[], degradations=degradations,
        )
        assert "以检测框为准" in out[0].resolution
        assert any("请核对" in text for text in degradations)

    def test_three_way_disagreement_is_admitted(self):
        degradations: list[str] = []
        out = AR.arbitrate_conflicts(
            [_count_conflict()], _shots(), _Probe("2"), "v.mp4",
            budget=2, notes=[], degradations=degradations,
        )
        assert "三源不一致" in out[0].resolution
        assert any("各说各话" in text for text in degradations)

    def test_unparseable_answer_keeps_the_old_verdict(self):
        """答不出数字就保留原判据 —— **不猜**。"""
        out = AR.arbitrate_conflicts(
            [_count_conflict()], _shots(), _Probe("看不清"), "v.mp4",
            budget=2, notes=[], degradations=[],
        )
        assert out[0].resolution == "以模型为准"

    def test_probe_failure_is_not_fatal(self):
        degradations: list[str] = []
        out = AR.arbitrate_conflicts(
            [_count_conflict()], _shots(), _Probe("", fail=True), "v.mp4",
            budget=2, notes=[], degradations=degradations,
        )
        assert out[0].resolution == "以模型为准"
        assert any("取证失败" in text for text in degradations)

    def test_budget_and_no_probe_are_no_ops(self):
        conflict = _count_conflict()
        probe = _Probe("1")
        assert AR.arbitrate_conflicts([conflict], _shots(), probe, "v.mp4",
                                      budget=0, notes=[], degradations=[])[0] == conflict
        assert not probe.asked
        assert AR.arbitrate_conflicts([conflict], _shots(), None, "v.mp4",
                                      budget=2, notes=[], degradations=[])[0] == conflict

    def test_question_forbids_counting_screen_images(self):
        """问法必须排除屏上图像 —— 那正是检测框出错的根源。"""
        assert "不要" in AR.COUNT_QUESTION and "屏幕" in AR.COUNT_QUESTION

    def test_parse_count(self):
        assert AR.parse_count("3") == 3
        assert AR.parse_count("画面里有 2 个真人") == 2
        assert AR.parse_count("看不清") is None
        assert AR.parse_count("") is None


class TestQuestionValidation:
    def test_accepts_a_usable_question(self):
        accepted, notes = review_core.apply_questions({}, [Question(
            id="", path="creative.C2_beats[0].emotion", text="这一镜情绪是？",
            options=("专注", "期待", "轻松"), reason="画面看不出",
        )])
        assert len(accepted) == 1
        assert accepted[0].options == ("专注", "期待", "轻松")
        assert accepted[0].id, "要给出稳定 id，答案才能对上"
        assert not notes

    def test_rejects_unwritable_target(self):
        """答案要写回字段，落点不可写的题根本没法作答 —— 直接丢弃并记账。"""
        accepted, notes = review_core.apply_questions({}, [Question(
            id="", path="locked.B3_blocking.positions", text="站位是？",
            options=("左", "右"),
        )])
        assert not accepted
        assert "不可写" in notes[0].detail or "没法用上" in notes[0].detail

    def test_rejects_bad_option_counts(self):
        for options in (("只有一个",), ("a", "b", "c", "d", "e")):
            accepted, notes = review_core.apply_questions({}, [Question(
                id="", path="creative.C1_logline", text="t", options=options,
            )])
            assert not accepted
            assert "选项" in notes[0].detail

    def test_at_most_two_questions(self):
        """问题一多用户就不看了 —— 和告警疲劳是同一件事。"""
        questions = [
            Question(id="", path=f"creative.C2_beats[{i}].emotion", text=f"q{i}",
                     options=("a", "b"))
            for i in range(4)
        ]
        accepted, notes = review_core.apply_questions({}, questions)
        assert len(accepted) == 2
        assert any("不再打扰" in n.detail for n in notes)

    def test_same_path_is_asked_once(self):
        accepted, _ = review_core.apply_questions({}, [
            Question(id="", path="creative.C1_logline", text="q1", options=("a", "b")),
            Question(id="", path="creative.C1_logline", text="q2", options=("c", "d")),
        ])
        assert len(accepted) == 1

    def test_answer_must_be_one_of_the_options(self):
        question = {"path": "creative.C1_logline", "options": ["甲", "乙"]}
        assert review_core.answer_question(question, "甲") == ("creative.C1_logline", "甲")
        try:
            review_core.answer_question(question, "丙")
        except ValueError as exc:
            assert "不在选项里" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("不在选项里的答案必须被拒绝")
