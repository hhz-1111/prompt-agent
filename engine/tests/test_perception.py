"""感知覆盖度自检 + 定向补看。

留出集实测暴露的问题：一条**框架外**片子的 4 个镜头里有 3 个只拿到「全景。」，
场景/动作/表情全空，而**整条链路没有任何环节发现这件事**。
这一层就是"发现自己没看清，然后回去补"。
"""

from __future__ import annotations

from engine.app.pipeline import perception as P
from engine.core.contracts import ReviewNote, ShotBoundary, ShotUnderstanding


def _understanding(index: int, **fields) -> ShotUnderstanding:
    return ShotUnderstanding(shot_index=index, **fields)


class TestThinDetection:
    def test_full_description_is_not_thin(self):
        item = _understanding(
            1, framing="全景", location="瑜伽室", blocking="仰卧拉伸", expression="平静",
        )
        assert not P.is_thin(item)

    def test_missing_core_field_is_thin(self):
        assert P.is_thin(_understanding(1, framing="全景", location="瑜伽室"))
        assert P.is_thin(_understanding(1, framing="全景"))

    def test_unknown_counts_as_empty(self):
        """`unknown` 是哨兵，不是内容 —— 否则空描述会被当成"写过了"。"""
        item = _understanding(1, framing="全景", location="unknown", blocking="unknown")
        assert P.is_thin(item)
        assert "location" in P.missing_fields(item)

    def test_only_framing_is_a_gap(self):
        """实测那条片子的形态：只剩「全景。」"""
        items = [
            _understanding(1, framing="全景", location="室内", blocking="示范动作",
                           expression="专注"),
            _understanding(2, framing="全景"),
            _understanding(3, framing="全景"),
            _understanding(4, framing="中全景"),
        ]
        assert [u.shot_index for u in P.perception_gaps(items)] == [2, 3, 4]


class TestMerge:
    def test_only_fills_empty_fields(self):
        """补看是"换一个问法再看一次"，同样可能看错，**不能覆盖已有内容**。"""
        item = _understanding(1, framing="全景", blocking="仰卧拉伸")
        merged = P.merge_finding(item, {
            "location": "瑜伽室", "blocking": "乱改的动作", "expression": "平静",
        })
        assert merged.location == "瑜伽室"
        assert merged.expression == "平静"
        assert merged.blocking == "仰卧拉伸", "已有内容一个字都不该被覆盖"

    def test_ignores_unknown_keys_and_empty_values(self):
        item = _understanding(1, framing="全景")
        merged = P.merge_finding(item, {"people": "乱塞", "location": "", "blocking": "站立"})
        assert merged.blocking == "站立"
        assert merged.location == ""


class _StubProbe:
    def __init__(self, answers, *, fail_on=()):
        self.answers = answers
        self.fail_on = set(fail_on)
        self.asked: list[tuple[int, tuple[str, ...]]] = []

    def describe_shot(self, video_path, shot, missing):
        self.asked.append((shot.index, tuple(missing)))
        if shot.index in self.fail_on:
            raise RuntimeError("抽帧失败")
        return dict(self.answers.get(shot.index, {}))


class TestFillGaps:
    def _shots(self, count=4):
        return [ShotBoundary(index=i, start_ms=(i - 1) * 1000, end_ms=i * 1000)
                for i in range(1, count + 1)]

    def test_fills_the_blank_shots(self):
        items = [
            _understanding(1, framing="全景", location="室内", blocking="示范", expression="专注"),
            _understanding(2, framing="全景"),
            _understanding(3, framing="全景"),
        ]
        probe = _StubProbe({
            2: {"location": "灰色墙面房间", "blocking": "仰卧屈膝", "expression": "平静"},
            3: {"location": "灰色墙面房间", "blocking": "坐姿前屈", "expression": "专注"},
        })
        notes: list[ReviewNote] = []
        degradations: list[str] = []
        out = P.fill_perception_gaps(
            items, self._shots(3), probe, "v.mp4", budget=4,
            notes=notes, degradations=degradations,
        )
        assert out[1].blocking == "仰卧屈膝"
        assert out[2].location == "灰色墙面房间"
        assert [n.applied for n in notes] == [True, True]
        assert "补到" in notes[0].detail
        assert not degradations
        assert not P.perception_gaps(out), "补完之后不该再有残缺镜头"

    def test_budget_limits_the_lookbacks(self):
        items = [_understanding(i, framing="全景") for i in range(1, 6)]
        probe = _StubProbe({i: {"blocking": f"动作{i}"} for i in range(1, 6)})
        out = P.fill_perception_gaps(
            items, self._shots(5), probe, "v.mp4", budget=2,
            notes=[], degradations=[],
        )
        assert len(probe.asked) == 2
        assert out[0].blocking and out[1].blocking
        assert not out[2].blocking and not out[3].blocking

    def test_lookback_failure_is_recorded_not_fatal(self):
        items = [_understanding(1, framing="全景")]
        probe = _StubProbe({}, fail_on={1})
        notes: list[ReviewNote] = []
        degradations: list[str] = []
        out = P.fill_perception_gaps(
            items, self._shots(1), probe, "v.mp4", budget=2,
            notes=notes, degradations=degradations,
        )
        assert out[0].blocking == ""
        assert notes and not notes[0].applied and "抽帧失败" in notes[0].detail
        assert any("补看失败" in text for text in degradations)

    def test_without_a_probe_it_is_a_no_op(self):
        items = [_understanding(1, framing="全景")]
        out = P.fill_perception_gaps(
            items, self._shots(1), None, "v.mp4", budget=3,
            notes=[], degradations=[],
        )
        assert out == items

    def test_budget_zero_is_a_no_op(self):
        items = [_understanding(1, framing="全景")]
        probe = _StubProbe({1: {"blocking": "x"}})
        out = P.fill_perception_gaps(
            items, self._shots(1), probe, "v.mp4", budget=0,
            notes=[], degradations=[],
        )
        assert out == items and not probe.asked

    def test_summary_names_the_remaining_gaps(self):
        """说明里必须写清**缺哪些字段** —— 否则事后无从核对。"""
        items = [_understanding(1, framing="全景"), _understanding(2, framing="全景")]
        summary = P.gap_summary(items)
        assert "镜头1（缺" in summary and "location" in summary
        assert P.gap_summary([_understanding(1, framing="全景", location="室内",
                                            blocking="示范", expression="专注")]) == ""

    def test_summary_is_consistent_with_filling(self):
        """补完之后那句"仍然残缺"必须消失 —— 两者不能自相矛盾。"""
        items = [_understanding(1, framing="全景"), _understanding(2, framing="全景")]
        probe = _StubProbe({
            1: {"location": "室内", "blocking": "示范", "expression": "专注"},
            2: {"location": "室内", "blocking": "拉伸", "expression": "平静"},
        })
        out = P.fill_perception_gaps(
            items, self._shots(2), probe, "v.mp4", budget=4,
            notes=[], degradations=[],
        )
        assert P.gap_summary(out) == "", P.gap_summary(out)
