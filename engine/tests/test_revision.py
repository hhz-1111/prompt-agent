"""用户意见改写（PRD F8.4「按建议重写」）的测试。

覆盖三块：

1. **可改范围与校验**（``engine/core/revision.py``）：哪些能改、哪些是红线、
   增删人物与镜头时的机械补齐、封闭枚举、改动明细；
2. **编排与授权策略**（``engine/app/pipeline/revision.py``）：
   用户意志优先（判定变差也保留）但**交付物坏了要回滚**；
3. **记账落盘**：``RevisionRecord.to_dict/from_dict`` 成对 ——
   第七轮踩过"加了字段没加进 to_dict，记账静默消失"，
   这里补一条回归测试守着（用户看不到改了什么 = 这个功能白做）。
"""

from __future__ import annotations

import pytest

from engine.core import phrases as ph
from engine.core import revision as rv
from engine.core.contracts import ReviewNote, RevisionOp, RevisionRecord, RevisionRefusal
from engine.tests.test_core import _sample_ir


@pytest.fixture(scope="module")
def lib():
    return ph.load_library()


# ==========================================================================
# 可改范围
# ==========================================================================
class TestRevisableRange:
    def test_user_named_places_are_revisable(self):
        """用户 2026-10-08 点名的五类意见：换场景/加人物/改站位/改台词/加镜头。

        **注意这里和 ``framework.can_write`` 不同**：那一层是"反推产线不许改锁定位"，
        这一层是"用户明确要求偏离复刻"。两套判据并存是有意的 ——
        所以锁定位在改写通道里**可写**，但每次都会带一条"这就不是 1:1 复刻了"的提醒。
        """
        for path in (
            "locked.B1_scene.anchor",          # 换场景
            "locked.B2_person",                # 加人物（列表本身，用于 add）
            "locked.B3_blocking.positions",    # 优化站位
            "creative.C2_beats[0].dialogue.line",   # 改台词
            "creative.C2_beats",               # 加镜头（列表本身，用于 add）
            "creative.C2_beats[0].framing",    # 调景别
            "creative.C2_beats[0].camera",     # 调拍摄方式
            "creative.C2_beats[0].blocking",   # 调运动轨迹
        ):
            assert rv.is_revisable(path), path

    def test_red_lines_stay_out(self):
        """素材绑定与交付形态**任何情况下**不许 AI 动（红线 1）。"""
        assert not rv.is_revisable("references[0].asset_id")
        assert not rv.is_revisable("delivery.mode")
        assert not rv.is_revisable("forbidden[0].text")

    def test_asset_binding_is_refused_with_a_reason(self):
        ir = _sample_ir()
        ops = [RevisionOp(op="set", path="references[0].asset_id", value="偷偷绑上")]
        _cand, changes, notes = rv.apply_ops(ir, ops)
        assert not changes
        assert not notes[0].applied
        assert "红线 1" in notes[0].detail


# ==========================================================================
# 校验与应用
# ==========================================================================
class TestApplyOps:
    def test_set_scene_anchor(self):
        ir = _sample_ir()
        ops = [RevisionOp(op="set", path="locked.B1_scene.anchor",
                          value="场景始终为【场景参考图】的社区咖啡馆，靠窗座位", reason="换场景")]
        cand, changes, notes = rv.apply_ops(ir, ops)
        assert cand["locked"]["B1_scene"]["anchor"].startswith("场景始终为【场景参考图】的社区咖啡馆")
        assert ir["locked"]["B1_scene"]["anchor"] != cand["locked"]["B1_scene"]["anchor"], \
            "原 IR 不能被改动"
        assert changes[0].label == "场景锚定"
        assert notes[0].applied

    def test_add_person_fills_plumbing(self):
        """加人物必须把**管道字段**补齐，否则正文里会冒出解析不了的假句柄。"""
        cand, changes, _notes = rv.apply_ops(
            _sample_ir(),
            [RevisionOp(op="add", path="locked.B2_person",
                        value={"name": "紫衣女子", "position": "画面右侧"}, reason="加一个人")],
        )
        person = cand["locked"]["B2_person"][-1]
        assert person["name"] == "紫衣女子"
        assert person["display_name"] == "紫衣女子"       # 由代码补
        assert person["multi_view_ref"] == "紫衣女子多视图"  # 由代码补（句柄靠它解析）
        assert person["blink"] == "5s" and person["skin"] == "brief"
        assert changes[0].label == "人物"

    def test_new_person_gets_an_unbound_asset_slot(self):
        """新人物要有一张多视图素材位，但**绝不假装已绑好**（红线 1）。"""
        cand, _changes, _notes = rv.apply_ops(
            _sample_ir(),
            [RevisionOp(op="add", path="locked.B2_person",
                        value={"name": "紫衣女子"}, reason="加一个人")],
        )
        added = rv.ensure_reference_slots(cand)
        assert "紫衣女子多视图" in added
        slot = next(r for r in cand["references"] if "紫衣女子多视图" in r["slots"])
        assert slot["asset_id"] is None

    def test_add_beat_gets_shot_id_and_rebalances_durations(self):
        ir = _sample_ir()
        cand, _changes, notes = rv.apply_ops(
            ir,
            [RevisionOp(op="add", path="creative.C2_beats",
                        value={"framing": "近景", "camera_angle": "平视", "camera": "固定",
                               "blocking": "李姐点头"}, reason="再加一个镜头")],
        )
        beats = cand["creative"]["C2_beats"]
        assert len(beats) == len(ir["creative"]["C2_beats"]) + 1
        assert beats[-1]["shot_id"] == 4          # 递增，不能是 0
        total = sum(float(b["duration_sec"]) for b in beats)
        assert abs(total - cand["meta"]["total_duration_sec"]) < 0.05, \
            "镜头数变了要把总时长平摊，否则下一轮判断台词节奏就是错的"
        assert any("平摊" in note.detail for note in notes)

    def test_shot_mode_change_syncs_split_strategy(self):
        """改了镜头模式 → 拆条策略必须跟着改，否则 G9 直接阻断。

        实测第一次 live 改写就是这么翻车的：场景/人物/镜头全改对了，
        却因为 `delivery.strategy` 还停在 S1，判定从 ALL_GREEN 掉成 BLOCKED。
        这是**机械联动**，所以由代码做。
        """
        cand, _changes, notes = rv.apply_ops(_sample_ir(), [
            RevisionOp(op="set", path="meta.shot_mode", value="multi_shot", reason="加镜头"),
        ])
        assert cand["delivery"]["strategy"] == "S2"
        assert any("拆条策略" in note.detail for note in notes)

    def test_same_mode_keeps_strategy_untouched(self):
        """模式没变就不动策略（别为了显得聪明而顺手改）。"""
        cand, _changes, _notes = rv.apply_ops(_sample_ir(), [
            RevisionOp(op="set", path="creative.C1_logline", value="换一句主线"),
        ])
        assert cand["delivery"]["strategy"] == _sample_ir()["delivery"]["strategy"]

    def test_new_speaker_gets_an_audio_slot(self):
        """新人物开口说话 → 代码补音色位，否则 G6 直接阻断。"""
        ir = _sample_ir()
        cand, _changes, _notes = rv.apply_ops(ir, [
            RevisionOp(op="add", path="locked.B2_person", value={"name": "紫衣女子"}, reason="加人"),
            RevisionOp(op="set", path="creative.C2_beats[0].dialogue.speaker",
                       value="紫衣女子", reason="换成她说"),
        ])
        assert rv.ensure_audio_entries(cand) == ["紫衣女子"]
        slots = rv.ensure_reference_slots(cand)
        assert "紫衣女子音频" in slots        # 音色素材位
        assert "紫衣女子多视图" in slots      # 新人物自己的多视图素材位

    def test_unknown_speaker_is_not_invented(self):
        """泛称说话人不补音色 —— 那等于替用户认了一个不存在的人物（红线 5）。"""
        cand, _changes, _notes = rv.apply_ops(_sample_ir(), [
            RevisionOp(op="set", path="creative.C2_beats[0].dialogue.speaker",
                       value="画外音", reason="来个旁白"),
        ])
        assert rv.ensure_audio_entries(cand) == []

    def test_remove_last_beat_is_refused(self):
        ir = _sample_ir()
        beats = ir["creative"]["C2_beats"]
        ops = [RevisionOp(op="remove", path=f"creative.C2_beats[{i}]")
               for i in range(len(beats) - 1, -1, -1)]
        cand, changes, notes = rv.apply_ops(ir, ops)
        assert len(cand["creative"]["C2_beats"]) == 1, "至少要留一个镜头"
        assert any(not note.applied and "至少要留一个镜头" in note.detail for note in notes)
        assert len(changes) == len(beats) - 1

    def test_closed_enum_rejects_unknown_value(self):
        _cand, changes, notes = rv.apply_ops(
            _sample_ir(), [RevisionOp(op="set", path="meta.aspect_ratio", value="4:3")])
        assert not changes
        assert "不在可选值里" in notes[0].detail

    def test_open_enum_accepts_the_users_wording(self):
        """景别/运镜**按用户的说法放行** —— 用户可能就是要词表外的东西。"""
        cand, changes, _notes = rv.apply_ops(
            _sample_ir(),
            [RevisionOp(op="set", path="creative.C2_beats[0].camera", value="航拍俯冲")],
        )
        assert cand["creative"]["C2_beats"][0]["camera"] == "航拍俯冲"
        assert changes

    def test_unknown_field_is_refused(self):
        _cand, _changes, notes = rv.apply_ops(
            _sample_ir(), [RevisionOp(op="set", path="locked.B9_unknown", value="x")])
        assert not notes[0].applied

    def test_dialogue_can_be_rewritten_by_user_request(self):
        """红线 3 防的是 Agent **自己编**台词；用户点名要改就改，但提醒会跟上。"""
        cand, changes, _notes = rv.apply_ops(
            _sample_ir(),
            [RevisionOp(op="set", path="creative.C2_beats[0].dialogue.line",
                        value="你这气色是真不一样了", reason="台词改口语一点")],
        )
        assert cand["creative"]["C2_beats"][0]["dialogue"]["line"] == "你这气色是真不一样了"
        assert rv.deviation_warning([c.path for c in changes]), \
            "改了台词必须提醒：这已经不是原片的 1:1 复刻"


# ==========================================================================
# 解析与记账
# ==========================================================================
class TestParsing:
    def test_null_fields_do_not_blow_up(self):
        """结构化输出里字段可能是 null（实测踩过 `{"shots": null}`）。"""
        outcome = rv.parse_outcome({"ops": None, "extensions": None,
                                    "refusals": None, "note": None})
        assert outcome.ops == () and outcome.note == ""

    def test_legacy_edits_key_still_parses(self):
        outcome = rv.parse_outcome({"edits": [{"path": "creative.C1_logline",
                                               "action": "set", "value": "新主线"}]})
        assert outcome.ops[0].path == "creative.C1_logline"

    def test_refusals_are_carried_through(self):
        outcome = rv.parse_outcome({"ops": [], "refusals": [
            {"instruction": "加一个画外音旁白", "reason": "没有对应音色素材"}]})
        assert outcome.refusals[0].instruction == "加一个画外音旁白"


class TestRecordRoundTrip:
    def test_to_dict_from_dict_keeps_everything(self):
        """第七轮的教训：契约加字段必须同步 to_dict，否则记账落盘就没了。"""
        record = RevisionRecord(
            id="abc12345", instruction="换场景", note="改了 1 处", model="stub",
            seconds=1.5, verdict_before="ALL_GREEN", verdict_after="PASS_WITH_WARNINGS",
            changes=(rv.RevisionChange(path="a", label="场景", before="旧", after="新"),),
            notes=(ReviewNote(path="", action="revise", reason="r", applied=True, detail="d"),),
            rejected=(ReviewNote(path="b", action="set", reason="r", applied=False, detail="no"),),
            refusals=(rv.RevisionRefusal(instruction="i", reason="why"),),
        )
        again = RevisionRecord.from_dict(record.to_dict())
        assert again.to_dict() == record.to_dict()


# ==========================================================================
# 编排：用户意志优先，但坏版本不给
# ==========================================================================
class _StubReviser:
    model_id = "stub-reviser"

    def __init__(self, outcome=None, error: Exception | None = None):
        self.outcome = outcome
        self.error = error
        self.seen_instruction = ""
        self.seen_prompt = ""

    def revise(self, *, instruction, ir, prompt="", evidence=None):
        self.seen_instruction = instruction
        self.seen_prompt = prompt
        if self.error:
            raise self.error
        return self.outcome


def _run(ir, reviser, lib, *, check_gates: bool = False):
    """跑一次改写。**默认不跑交付闸门**（用户 2026-10-08：改一下不用重新体检）。"""
    from engine.app.pipeline import revision as pipeline

    return pipeline.run_revision(ir, {"understandings": []}, reviser, lib, "换场景",
                                 check_gates=check_gates)


class TestBackgroundActionsAreProtected:
    """改写删人物时，**不许把画面里其他人的动作一起删掉**（用户 2026-10-07 实测）。

    用户原话：「我直接用改写功能改，叫你去掉那些多余的人物，你倒好，人物是删掉了，
    **连着阿姨后景的学员做动作的提示词也给我去掉了**。」

    后景学员/观众由生成模型自行生成，但**他们做什么必须留在节拍里** ——
    那是提示词里唯一说明"背景有人在练瑜伽"的地方。
    """

    def _ir(self):
        from engine.core import framework as fw

        ir = fw.empty_ir(genre="studio", shot_mode="one_take", aspect_ratio="9:16")
        ir["locked"]["B2_person"] = [
            {"name": "紫色上衣女子", "display_name": "紫色上衣女子",
             "multi_view_ref": "紫色上衣女子多视图"},
            {"name": "粉色背心女子", "display_name": "粉色背心女子",
             "multi_view_ref": "粉色背心女子多视图"},
        ]
        ir["creative"]["C2_beats"] = [{
            "shot_id": 1, "duration_sec": 10.0,
            "blocking": "紫色上衣女子面向镜头站立；其余数人在她身后各自瑜伽垫上分散站立做动作",
        }]
        return ir

    def test_the_dropped_background_clause_is_restored(self):
        from engine.app.pipeline.revision import protect_background_actions

        base = self._ir()
        candidate = self._ir()
        # 模拟模型"顺手"把后景那句也删了
        candidate["creative"]["C2_beats"][0]["blocking"] = "紫色上衣女子面向镜头站立"

        notes = protect_background_actions(base, candidate)
        assert "其余数人在她身后" in candidate["creative"]["C2_beats"][0]["blocking"]
        assert notes and notes[0].applied is False
        assert "其他人的动作" in notes[0].reason

    def test_an_unrelated_edit_is_left_alone(self):
        from engine.app.pipeline.revision import protect_background_actions

        base = self._ir()
        candidate = self._ir()
        candidate["creative"]["C2_beats"][0]["blocking"] = (
            "紫色上衣女子面向镜头站立；其余数人在她身后各自瑜伽垫上分散站立做动作，"
            "动作更慢一些"
        )
        assert protect_background_actions(base, candidate) == []
        assert "动作更慢一些" in candidate["creative"]["C2_beats"][0]["blocking"]


class TestOrchestration:
    def test_applies_and_reports_changes(self, lib):
        from engine.core.contracts import RevisionOutcome

        stub = _StubReviser(RevisionOutcome.of(
            ops=[RevisionOp(op="set", path="locked.B1_scene.anchor",
                            value="场景始终为【场景参考图】的咖啡馆，靠窗", reason="换场景")],
            note="把场景换成了咖啡馆",
        ))
        run = _run(_sample_ir(), stub, lib)
        assert run.applied and run.record.ok
        assert run.ir["locked"]["B1_scene"]["anchor"].endswith("靠窗")
        assert run.record.note == "把场景换成了咖啡馆"
        assert run.record.changes
        assert stub.seen_prompt, "要把交付物原文给模型看，用户说「第三句」才对得上"

    def test_nothing_landable_is_not_reported_as_success(self, lib):
        """一条都落不了地时**不许说"已完成"** —— 这是这个功能最容易骗人的地方。"""
        from engine.core.contracts import RevisionOutcome

        stub = _StubReviser(RevisionOutcome.of(
            ops=[RevisionOp(op="set", path="references[0].asset_id", value="x")],
            refusals=[rv.RevisionRefusal(instruction="加旁白", reason="没有音色素材")],
        ))
        run = _run(_sample_ir(), stub, lib)
        assert not run.applied and not run.record.ok
        assert run.record.refusals and run.record.rejected

    def test_model_failure_is_reported_not_swallowed(self, lib):
        run = _run(_sample_ir(), _StubReviser(error=RuntimeError("超时")), lib)
        assert not run.applied
        assert "超时" in run.record.note
        assert run.record.refusals

    def test_verdict_downgrade_is_kept_but_disclosed(self, lib):
        """（开了闸门时）用户要的改动让判定变差 → **保留 + 告知**，不回滚。"""
        from engine.core.contracts import RevisionOutcome

        ir = _sample_ir()
        stub = _StubReviser(RevisionOutcome.of(ops=[
            RevisionOp(op="set", path="locked.B1_scene.anchor",
                       value="场景始终为【场景参考图】的咖啡馆" + "，细节很多" * 60, reason="换场景"),
        ]))
        run = _run(ir, stub, lib, check_gates=True)
        assert run.applied, "用户明确要求的改动不许被悄悄回滚"
        assert any("交付判定" in note.detail for note in run.record.notes) or \
            run.record.verdict_after == run.record.verdict_before

    def test_default_path_is_a_single_model_call(self, lib):
        """**默认形态：一次模型调用**（用户 2026-10-08 的要求）。

        改完直接拼出新正文填回去 —— 不跑 14 项交付闸门、不做闸门反馈修复轮、
        也没有"判定变差/代价"那一堆记账。用户原话：
        「修改提示词不需要再重跑了，直接用 AI 改写提示词填回去就可以了」。
        """
        from engine.core.contracts import RevisionOutcome

        class _CountingReviser:
            model_id = "stub"
            calls = 0

            def revise(self, *, instruction, ir, prompt="", evidence=None):
                type(self).calls += 1
                # 故意写一句会被 G14 判为泛化表述的话：默认路径下它不该拦住我
                return RevisionOutcome.of(ops=[RevisionOp(
                    op="set", path="locked.B1_scene.anchor",
                    value="场景始终为【场景参考图】的咖啡馆，两镜之间场景与光线保持一致",
                    reason="换场景")])

        _CountingReviser.calls = 0
        run = _run(_sample_ir(), _CountingReviser(), lib)
        assert _CountingReviser.calls == 1, "默认只调一次模型"
        assert run.applied and run.record.ok
        assert run.record.verdict_before == "" and run.record.verdict_after == ""
        assert not any("交付判定" in note.detail for note in run.record.notes)
        assert not any(note.action == "repair-round" for note in run.record.notes)

    def test_default_path_still_refuses_an_unusable_deliverable(self, lib):
        """不跑闸门 ≠ 没有底线：正文里留没解析的句柄照样回滚（红线 6）。"""
        from engine.core.contracts import RevisionOutcome

        ir = _sample_ir()
        stub = _StubReviser(RevisionOutcome.of(ops=[
            RevisionOp(op="set", path="creative.C1_logline",
                       value="看【不存在的素材】讲经络瑜伽", reason="改主线"),
        ]))
        run = _run(ir, stub, lib)
        assert not run.applied and run.ir == ir
        assert "回滚" in run.record.note

    def test_broken_deliverable_is_rolled_back(self, lib):
        """正文里出现解析不了的句柄（红线 6）→ 回滚：这种版本给他没有意义。"""
        from engine.core.contracts import RevisionOutcome

        ir = _sample_ir()
        stub = _StubReviser(RevisionOutcome.of(ops=[
            RevisionOp(op="add", path="creative.C2_beats",
                       value={"framing": "近景", "blocking": "看【不存在的素材】",
                              "dialogue": {"speaker": "王姐", "line": "这句里有【野句柄】"}},
                       reason="加镜头"),
        ]))
        run = _run(ir, stub, lib)
        assert not run.applied
        assert run.ir == ir, "回滚必须把原 IR 原样还回去"
        assert "回滚" in run.record.note

    def test_gate_feedback_triggers_one_repair_round(self, lib):
        """第一轮改出**新的阻断**时，把闸门结论递回去再修一轮（最多一次）。

        实测场景：模型改对了内容，却写了泛化表述「保持一致」→ G14 阻断。
        闸门的 suggestion 里已经写着替换句，所以再问一次就能改对 ——
        这正是项目里一直用的模式：链路自己发现缺口，然后回去补。
        """
        from engine.core.contracts import RevisionOutcome

        class _TwoStageReviser:
            model_id = "stub"
            calls = 0

            def revise(self, *, instruction, ir, prompt="", evidence=None):
                type(self).calls += 1
                if type(self).calls == 1:
                    return RevisionOutcome.of(ops=[RevisionOp(
                        op="set", path="locked.B1_scene.anchor",
                        value="场景始终为【场景参考图】的咖啡馆，两镜之间场景与光线保持一致",
                        reason="换场景")])
                assert "交付闸门" in instruction, "第二轮必须把闸门结论带过去"
                return RevisionOutcome.of(ops=[RevisionOp(
                    op="set", path="locked.B1_scene.anchor",
                    value="场景始终为【场景参考图】的咖啡馆，两镜之间场景与光线100%相同",
                    reason="按闸门建议改成量化写法")])

        _TwoStageReviser.calls = 0
        run = _run(_sample_ir(), _TwoStageReviser(), lib, check_gates=True)
        assert _TwoStageReviser.calls == 2, "只许再修一轮，不能无限循环"
        assert run.applied and run.record.ok
        assert "100%相同" in run.ir["locked"]["B1_scene"]["anchor"]
        assert any("再修一轮" in note.detail for note in run.record.notes)


# ==========================================================================
# HTTP 层：端点契约、落盘、撤销
# ==========================================================================
class _StubApiReviser:
    """API 测试用的替身：不烧真模型。

    每轮改一个**不同的值**（``round_no`` 递增），这样"三轮都真的落地"，
    才能验证"一次提交一轮、逐步撤销"。
    """

    model_id = "stub-api"

    def __init__(self, round_no: int = 1) -> None:
        self.round_no = round_no

    def revise(self, *, instruction, ir, prompt="", evidence=None):
        from engine.core.contracts import RevisionOutcome

        return RevisionOutcome.of(
            ops=[RevisionOp(op="set", path="locked.B1_scene.anchor",
                            value=("场景始终为【场景参考图】的社区咖啡馆，靠窗座位"
                                   f"（第 {self.round_no} 版）"),
                            reason="按用户要求换场景")],
            note="场景换成社区咖啡馆",
        )


class _SecondRoundFailsReviser(_StubApiReviser):
    """第二轮"没改成"（模型一条都落不了地）—— 用来验记账与快照不会错位。"""

    def revise(self, *, instruction, ir, prompt="", evidence=None):
        from engine.core.contracts import RevisionOutcome

        if self.round_no == 2:
            return RevisionOutcome.of(
                refusals=[RevisionRefusal(instruction=instruction, reason="这条我做不到")],
                note="这轮没改成",
            )
        return super().revise(instruction=instruction, ir=ir, prompt=prompt,
                              evidence=evidence)


class TestReviseApi:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        from dataclasses import replace

        from fastapi.testclient import TestClient

        from engine.app import api as api_module
        from engine.app.config import Settings
        from engine.app.store import ProjectStore

        settings = replace(Settings.load(), workspace_dir=tmp_path)
        store = ProjectStore(tmp_path / "projects")
        project = store.create("改写接口测试", "")
        project.ir = _sample_ir()
        project.report = {"understandings": []}
        project.status = "ready"
        store.save(project)

        counter = {"n": 0}

        def _make(_settings):
            counter["n"] += 1
            return _StubApiReviser(counter["n"])

        monkeypatch.setattr(api_module, "build_reviser", _make)
        with TestClient(api_module.create_app(settings)) as client:
            client.project_id = project.project_id          # type: ignore[attr-defined]
            yield client

    def test_revise_then_assemble_then_undo(self, client):
        pid = client.project_id                              # type: ignore[attr-defined]
        before = client.get(f"/api/projects/{pid}/assemble").json()
        assert "咖啡馆" not in before["document"]["text"]

        response = client.post(f"/api/projects/{pid}/revise",
                               json={"instruction": "换成咖啡馆"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["record"]["note"] == "场景换成社区咖啡馆"
        assert payload["changes"], "要把改动明细返回给前端"
        assert payload["can_undo"] is True

        after = client.get(f"/api/projects/{pid}/assemble").json()
        assert "咖啡馆" in after["document"]["text"]
        assert len(after["revisions"]) == 1, "记账必须落盘（刷新页面还能看到）"
        assert after["can_undo"] is True

        undo = client.post(f"/api/projects/{pid}/revise/undo")
        assert undo.status_code == 200
        restored = client.get(f"/api/projects/{pid}/assemble").json()
        assert "咖啡馆" not in restored["document"]["text"]
        assert restored["revisions"] == []
        # 没有可撤销的了 → 409（而不是 200 静默什么都不做）
        assert client.post(f"/api/projects/{pid}/revise/undo").status_code == 409

    def test_each_round_is_one_submission_and_undo_walks_back_one_step(self, client):
        """**一次提交一轮、可撤销**（用户 2026-10-08 确认的形态）。

        语义要写清楚，不然界面上会说不明白：

        - 每轮只提交**一句**意见，同步返回；
        - 每轮都推一份 IR 快照，撤销是**逐步回退**（一次退一轮），
          撤到底才回 409 —— 不是"一键回到最初"；
        - 每轮的账都留着（前端刷新后还能看到"我当时说了什么"）。
        """
        pid = client.project_id                              # type: ignore[attr-defined]
        instructions = ["换成咖啡馆", "再多加一个镜头", "台词口语一点"]

        for index, text in enumerate(instructions, start=1):
            response = client.post(f"/api/projects/{pid}/revise",
                                   json={"instruction": text})
            assert response.status_code == 200
            assert response.json()["can_undo"] is True
            state = client.get(f"/api/projects/{pid}/assemble").json()
            assert len(state["revisions"]) == index, "每轮都要留下一笔记账"

        # 逐步回退：三次撤销，每次都少一轮
        for remaining in (2, 1, 0):
            assert client.post(f"/api/projects/{pid}/revise/undo").status_code == 200
            state = client.get(f"/api/projects/{pid}/assemble").json()
            assert len(state["revisions"]) == remaining
            assert state["can_undo"] is (remaining > 0)
        # 撤到底就没有了
        assert client.post(f"/api/projects/{pid}/revise/undo").status_code == 409

    def test_failed_round_does_not_desync_records_and_snapshots(self, tmp_path, monkeypatch):
        """"没改成"的那一轮会记账但**不推快照** —— 撤销必须仍然对齐。

        这是补测抓出来的真 bug：原来撤销是"各弹各的"（弹一条快照 + 弹一条记账），
        于是第二轮没落地时，撤销会**删掉第二轮的记录、却退回第一轮之前**的版本，
        历史里留下一条"声称改了、其实已回退"的记录。
        修法：快照带上是哪条记录的（``{"id", "ir"}``），成套弹出。
        """
        from dataclasses import replace

        from fastapi.testclient import TestClient

        from engine.app import api as api_module
        from engine.app.config import Settings
        from engine.app.store import ProjectStore

        settings = replace(Settings.load(), workspace_dir=tmp_path)
        store = ProjectStore(tmp_path / "projects")
        project = store.create("对账测试", "")
        project.ir = _sample_ir()
        project.report = {"understandings": []}
        project.status = "ready"
        store.save(project)

        counter = {"n": 0}

        def _make(_settings):
            counter["n"] += 1
            return _SecondRoundFailsReviser(counter["n"])

        monkeypatch.setattr(api_module, "build_reviser", _make)
        with TestClient(api_module.create_app(settings)) as client:
            pid = project.project_id
            assert client.post(f"/api/projects/{pid}/revise",
                               json={"instruction": "换成咖啡馆"}).json()["ok"] is True
            second = client.post(f"/api/projects/{pid}/revise",
                                 json={"instruction": "再来点别的"}).json()
            assert second["ok"] is False, "第二轮应当如实回报「没改成」"
            state = client.get(f"/api/projects/{pid}/assemble").json()
            assert len(state["revisions"]) == 2, "两轮都记账"
            assert state["can_undo"] is True, "只有第一轮推了快照"

            client.post(f"/api/projects/{pid}/revise/undo")
            after = client.get(f"/api/projects/{pid}/assemble").json()
            assert "咖啡馆" not in after["document"]["text"], "要退回第一轮之前的版本"
            assert after["revisions"] == [], "两条记账都该跟着消失（否则历史与正文对不上）"
            assert after["can_undo"] is False

    def test_revise_path_never_calls_another_model(self, client, monkeypatch):
        """**改写只调一个模型**（用户 2026-10-08 的核心要求）。

        原话：「提交，调用 deepseek 模型改提示词，回填到提示词里，
        **就不需要重新再给 mimo 去自检交付了**」。

        所以这里把底层 HTTP 调用直接改成"一调就炸"：如果改写路径里还藏着
        任何一次自检/组装/闸门之外的真实模型调用（比如又送 MiMo 跑自检），
        这个测试立刻失败。正常路径下它一次都不调 —— 只有被测的替身会"改"。
        """
        from engine.app.providers import mimo as mimo_module

        def _explode(*_args, **_kwargs):
            raise AssertionError("改写路径不该调用任何真实模型接口（除改写模型本身）")

        monkeypatch.setattr(mimo_module.MimoClient, "_post", _explode)

        pid = client.project_id                              # type: ignore[attr-defined]
        response = client.post(f"/api/projects/{pid}/revise",
                               json={"instruction": "换成咖啡馆"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        # 替身不经过 MimoClient，所以这次改写**没有任何真实模型调用记录**
        assert payload["record"]["model_calls"] == []
        # 而且判定字段是空的 —— 说明没跑那套交付自检
        assert payload["verdict_before"] == "" and payload["verdict_after"] == ""

    def test_empty_instruction_is_rejected(self, client):
        pid = client.project_id                              # type: ignore[attr-defined]
        assert client.post(f"/api/projects/{pid}/revise",
                           json={"instruction": "   "}).status_code == 400

    def test_project_without_ir_is_conflict(self, client):
        """还没反推就改写 → 409（前端据此判断，不当错误弹）。

        **必须用这个测试自己的临时工作区**（``client.app.state.store``）——
        用 ``Settings.load()`` 会往用户真实的 workspace 里建项目。
        """
        store = client.app.state.store
        blank = store.create("还没反推", "")
        response = client.post(f"/api/projects/{blank.project_id}/revise",
                               json={"instruction": "换场景"})
        assert response.status_code == 409
