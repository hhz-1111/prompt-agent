"""台词归属覆盖度自检：判不出说话人时回去取证再判一次。

留出集实测暴露的问题：双人示范片 4 句台词只判出第 1 句的说话人，
交付物出现「镜头2（无说话人）：…」，G6 因此不通过。
"""

from __future__ import annotations

from engine.app.pipeline import attribution as A
from engine.core.contracts import (
    DialogueTurn,
    ProbeFinding,
    ReviewNote,
    ShotBoundary,
    ShotDialogueContext,
)


def _turn(index: int, text: str, speaker: str = "") -> DialogueTurn:
    return DialogueTurn(shot_index=index, speaker=speaker, text=text)


def _contexts(count: int = 4) -> list[ShotDialogueContext]:
    return [
        ShotDialogueContext(
            shot_index=i, duration_sec=2.0, location="瑜伽室",
            people=("浅蓝瑜伽服女子", "紫色上衣女子"),
            action="紫色上衣女子张口说话，浅蓝瑜伽服女子仰卧",
            expression="紫色上衣女子神情专注",
        )
        for i in range(1, count + 1)
    ]


def _shots(count: int = 4) -> list[ShotBoundary]:
    return [
        ShotBoundary(index=i, start_ms=(i - 1) * 2000, end_ms=i * 2000)
        for i in range(1, count + 1)
    ]


class _StubProbe:
    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def look(self, video_path, request):
        self.asked.append(request)
        return ProbeFinding(request=request, answer=self.answers.get(request.reason, ""))


class TestGapDetection:
    def test_unspoken_turns_found(self):
        turns = [_turn(1, "a", "人物1"), _turn(2, "b"), _turn(3, "c", "")]
        assert [t.shot_index for t in A.unspoken_turns(turns)] == [2, 3]

    def test_unknown_counts_as_missing(self):
        assert not A.has_speaker(_turn(1, "a", "unknown"))
        assert not A.has_speaker(_turn(1, "a", "无"))
        assert A.has_speaker(_turn(1, "a", "紫色上衣女子"))


class TestEvidenceCollection:
    def test_asks_once_per_shot_and_records(self):
        turns = [_turn(1, "a"), _turn(1, "b"), _turn(2, "c")]
        probe = _StubProbe({})
        notes: list[ReviewNote] = []
        degradations: list[str] = []
        evidence = A.collect_speaker_evidence(
            turns, _shots(), _contexts(), probe, "v.mp4",
            budget=3, notes=notes, degradations=degradations,
        )
        # 镜1 两条台词只问一次
        assert len(probe.asked) == 2
        assert [n.action for n in notes] == ["probe", "probe"]
        assert evidence == {}

    def test_question_names_the_people_in_frame(self):
        question = A.speaker_question(_contexts()[0])
        assert "紫色上衣女子" in question
        assert "看不出" in question, "要给它留一条'看不出来'的退路，否则它会硬猜"

    def test_budget_limits_probes(self):
        turns = [_turn(i, f"t{i}") for i in range(1, 6)]
        probe = _StubProbe({})
        A.collect_speaker_evidence(
            turns, _shots(5), _contexts(5), probe, "v.mp4",
            budget=2, notes=[], degradations=[],
        )
        assert len(probe.asked) == 2

    def test_failure_is_recorded_not_fatal(self):
        class Broken:
            def look(self, video_path, request):
                raise RuntimeError("抽帧失败")

        notes: list[ReviewNote] = []
        degradations: list[str] = []
        A.collect_speaker_evidence(
            [_turn(1, "a")], _shots(), _contexts(), Broken(), "v.mp4",
            budget=2, notes=notes, degradations=degradations,
        )
        assert notes and not notes[0].applied and "抽帧失败" in notes[0].detail
        assert any("取证失败" in text for text in degradations)

    def test_without_probe_or_budget_it_is_a_no_op(self):
        notes: list[ReviewNote] = []
        assert A.collect_speaker_evidence(
            [_turn(1, "a")], _shots(), _contexts(), None, "v.mp4",
            budget=3, notes=notes, degradations=[],
        ) == {}
        assert not notes


class TestMerge:
    def test_fills_only_empty_speakers(self):
        turns = [_turn(1, "a", "浅蓝瑜伽服女子"), _turn(2, "b"), _turn(3, "c")]
        resolved = {1: "别的人", 2: "紫色上衣女子", 3: "紫色上衣女子"}
        known = ["浅蓝瑜伽服女子", "紫色上衣女子"]
        out = A.merge_speakers(turns, resolved, known)
        assert out[0].speaker == "浅蓝瑜伽服女子", "已有的说话人一个字都不能改"
        assert out[1].speaker == "紫色上衣女子"
        assert out[2].speaker == "紫色上衣女子"

    def test_unmatched_answer_leaves_it_empty(self):
        """取证回答认不出是哪个人物时**不填** —— 塞一个"看起来像"的名字就是猜。"""
        turns = [_turn(1, "a")]
        out = A.merge_speakers(turns, {1: "画面里有个穿绿衣服的人"}, ["紫色上衣女子"])
        assert out[0].speaker == ""

    def test_answer_must_match_a_known_person(self):
        turns = [_turn(1, "a")]
        out = A.merge_speakers(turns, {1: "说话的是紫色上衣女子"}, ["紫色上衣女子"])
        assert out[0].speaker == "紫色上衣女子"


class TestSummary:
    def test_names_the_lines_that_stayed_empty(self):
        turns = [_turn(1, "腰难受", "人物1"), _turn(2, "背难受"), _turn(3, "膝难受")]
        summary = A.attribution_summary(turns)
        assert "「背难受」" in summary and "仍然判不出说话人" in summary

    def test_all_resolved_means_no_noise(self):
        assert A.attribution_summary([_turn(1, "a", "人物1")]) == ""
