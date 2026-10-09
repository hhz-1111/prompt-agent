"""自检-重写回路：core 纯函数部分。

回路的价值全在这三条上，所以每条都单独钉住：

    1. 只改白名单字段（永不可裁的东西一个字都不能动）
    2. 能真的改对（重复/矛盾这类语义问题修得掉）
    3. 改坏了要能整体回滚（宁可少改，不可改坏）
"""

from __future__ import annotations

import json

import copy

import pytest

from engine.core import review as review_core
from engine.core.contracts import CritiqueEdit
from engine.tests.test_core import _sample_ir


class TestEditWhitelist:
    def test_allows_the_semantic_fields(self):
        for path in (
            "locked.B1_scene.anchor",
            "locked.B2_person[0].display_name",
            "locked.B2_person[0].position",
            "creative.C1_logline",
            "creative.C2_beats[0].location",
            "creative.C2_beats[2].blocking",
        ):
            assert review_core.is_allowed(path), path

    def test_refuses_the_never_cut_fields(self):
        """站位/音色/台词原文/角色名/素材绑定 —— 改了就不再是复刻。"""
        for path in (
            "locked.B3_blocking.positions",
            "locked.B3_blocking.positions[0]",
            "locked.B6_audio",
            "creative.C2_beats[0].dialogue.line",
            "locked.B2_person[0].name",
            "references[0].asset_id",
            "forbidden",
            "meta.shot_mode",
        ):
            assert not review_core.is_allowed(path), path

    def test_refuses_measurements_and_the_dimension_notes(self):
        """量测结果与"逐镜头维度"的自由文本字段也不能让自检动。

        - 景别/运镜是 CV 量测，让模型用自己的复述推翻量测＝把量测降级成猜；
        - occlusion/eye_line/off_screen 在契约里是**单值自由文本**，而它们描述的是
          **逐镜头**的事。实测自检把"镜头1 只有 1 人"这种人数矛盾塞进 off_screen，
          与 positions 里的「固定 2 位人物出镜」并排出现 —— 既矛盾又重复。
        """
        for path in (
            "creative.C2_beats[0].framing",
            "creative.C2_beats[0].camera",
            "creative.C2_beats[1].camera_angle",
            "locked.B3_blocking.occlusion",
            "locked.B3_blocking.eye_line",
            "locked.B3_blocking.off_screen",
        ):
            assert not review_core.is_allowed(path), path

    def test_rejection_says_why(self):
        note = review_core.apply_edits(
            _sample_ir(),
            [CritiqueEdit(path="locked.B3_blocking.positions", action="set", value="乱改")],
        )[1][0]
        assert not note.applied
        assert "永不可裁" in note.detail


class TestApplyEdits:
    def test_clear_removes_duplicate_scene(self):
        ir = _sample_ir()
        assert ir["creative"]["C2_beats"][1]["location"] == "菜市场摊位前"
        updated, notes = review_core.apply_edits(
            ir, [CritiqueEdit(path="creative.C2_beats[1].location", action="clear",
                              reason="场景与镜1重复")],
        )
        assert notes[0].applied
        assert updated["creative"]["C2_beats"][1]["location"] == ""
        # **原 IR 不能被改动**（调用方拿它做回滚）
        assert ir["creative"]["C2_beats"][1]["location"] == "菜市场摊位前"

    def test_set_rewrites_only_that_field(self):
        ir = _sample_ir()
        before = copy.deepcopy(ir)
        updated, notes = review_core.apply_edits(
            ir,
            [CritiqueEdit(path="creative.C2_beats[0].blocking", action="set",
                          value="王姐睁大眼睛，抬手指向李姐的画外方向",
                          reason="与画面事实不符：她指的方向在后景")],
        )
        assert notes[0].applied
        assert updated["creative"]["C2_beats"][0]["blocking"].startswith("王姐睁大眼睛")
        # 其它字段一字未动
        assert updated["creative"]["C2_beats"][1] == before["creative"]["C2_beats"][1]
        assert updated["locked"] == before["locked"]

    def test_no_op_and_bad_path_are_recorded_not_silently_dropped(self):
        ir = _sample_ir()
        _updated, notes = review_core.apply_edits(ir, [
            CritiqueEdit(path="creative.C2_beats[0].location", action="set",
                         value="菜市场摊位前", reason="值没变"),
            CritiqueEdit(path="creative.C2_beats[99].location", action="clear", reason="越界"),
            CritiqueEdit(path="creative.C2_beats[0].dialogue.line", action="set",
                         value="我替你写一句台词", reason="试图代写台词"),
        ])
        assert [n.applied for n in notes] == [False, False, False]
        assert "值与现状相同" in notes[0].detail
        assert "下标越界" in notes[1].detail or "路径不存在" in notes[1].detail
        assert "代写" in notes[2].detail or "红线" in notes[2].detail


class TestEvidence:
    def test_report_persists_the_review_notes(self):
        """记账必须能**落盘**。

        实测踩过：`InferenceReport.to_dict()` 是逐字段手写的，加了 `reviews`
        字段却没加进 to_dict —— 回路在生产链路里跑了，进度也出现了「自检与重写」，
        但报告里一条记账都没有（看起来像"回路没发现问题"，其实是丢了）。
        """
        from engine.core.contracts import (
            InferenceReport, ReviewNote, VideoInfo,
        )

        report = InferenceReport(video=VideoInfo(
            path="x.mp4", duration_sec=1.0, width=1, height=1, fps=1.0, has_audio=False,
        ))
        report.reviews = (ReviewNote(
            path="creative.C2_beats[0].expression", action="set",
            reason="表情句挂在了错的人身上", applied=True, detail="a → b",
        ),)
        stored = report.to_dict()
        assert stored["reviews"][0]["path"] == "creative.C2_beats[0].expression"
        assert stored["reviews"][0]["applied"] is True
        assert stored["reviews"][0]["reason"] == "表情句挂在了错的人身上"

    def test_evidence_has_the_facts_the_critic_needs(self):
        ir = _sample_ir()
        from engine.core.contracts import (
            InferenceReport, ShotUnderstanding, VideoInfo,
        )

        report = InferenceReport(
            video=VideoInfo(
                path="x.mp4", duration_sec=12.0, width=1080, height=1920,
                fps=30.0, has_audio=True,
            ),
            understandings=(ShotUnderstanding(
                shot_index=1, framing="中景", camera_angle="平视", camera="固定",
                location="菜市场摊位前", blocking="王姐抬手", expression="惊讶",
                emotion="惊讶", people=["王姐", "李姐"],
            ),),
        )
        evidence = review_core.critique_evidence(ir, report)
        assert evidence["逐镜头画面事实"][0]["画面里是谁"] == ["王姐", "李姐"]
        assert evidence["有没有台词"] is True
        assert evidence["台词"], "台词必须给出来，否则模型无法核对说话人"
        # 再带上一轮的问题，避免它原地打转
        assert review_core.critique_evidence(ir, report, ["x：越界"])["上一轮没解决或越界的问题"]


class TestPromptContract:
    def test_rubric_lists_every_allowed_path(self):
        """提示词里的可改路径必须与代码白名单**同源**，不能各写一份。"""
        reference = review_core.allowed_path_reference()
        for path in review_core.ALLOWED_EDIT_PREFIXES:
            assert path in reference

    def test_critic_system_prompt_carries_the_whitelist(self):
        from engine.app.providers.mimo_analysis import CRITIQUE_SYSTEM

        assert "创作.C2_beats" in CRITIQUE_SYSTEM or "creative.C2_beats" in CRITIQUE_SYSTEM
        assert "宁少勿多" in CRITIQUE_SYSTEM
        assert "不许动" in CRITIQUE_SYSTEM


@pytest.mark.parametrize("action", ["set", "clear"])
def test_actions_are_limited(action):
    assert action in ("set", "clear")


class TestInternalSentinels:
    def test_unknown_mixed_into_a_sentence_is_stripped(self):
        """模型会把 `unknown` **混在句子中间**，它绝不能进交付物。

        实测踩过（生产链路重推时抓到的）：
        `expression = "unknown，面部朝向侧对镜头，看不清具体表情"`
        整串不等于 `unknown`，原来的判空放它过去，交付物里于是出现
        「……走向画面右侧的前台，**unknown**，面部朝向侧对镜头」。
        """
        from engine.app.pipeline.to_ir import _text

        assert _text("unknown") == ""
        assert _text("未知") == ""
        assert _text("unknown，面部朝向侧对镜头，看不清具体表情") == "面部朝向侧对镜头，看不清具体表情"
        assert _text("女子走向前台， unknown ，面带微笑") == "女子走向前台，面带微笑"
        # 正常内容一个字不能动
        assert _text("面部朝向侧对镜头") == "面部朝向侧对镜头"


# --------------------------------------------------------------------------
# 编排层：回路真的跑起来的样子
# --------------------------------------------------------------------------
class _StubCritic:
    """测试替身：按脚本提修改，并把收到的东西存下来供断言。"""

    def __init__(self, edits):
        self.edits = list(edits)
        self.calls: list[dict] = []

    def critique(self, *, prompt, evidence):
        self.calls.append({"prompt": prompt, "evidence": evidence})
        return list(self.edits)


class _BrokenCritic:
    def critique(self, *, prompt, evidence):
        raise RuntimeError("网关 502")


def _report_for(ir):
    from engine.core.contracts import InferenceReport, VideoInfo

    return InferenceReport(
        video=VideoInfo(path="x.mp4", duration_sec=12.0, width=1080, height=1920,
                        fps=30.0, has_audio=True),
    )


class TestProbeWiring:
    """取证回路：模型可以要求"再看一眼"，看完再决定改不改。

    这是"看不懂就去查"的能力 —— 以前它只能留空或硬猜。
    """

    def _lib(self):
        from engine.core import phrases as ph

        return ph.load_library()

    def _report(self):
        return _report_for(_sample_ir(minimal=True))

    def test_probe_then_rewrite(self):
        """第一轮只说"我要看"，拿到画面后第二轮才改。"""
        from engine.app.pipeline.review import run_review
        from engine.core.contracts import CritiqueOutcome, ProbeFinding, ProbeRequest

        seen: list[dict] = []

        class Critic:
            def __init__(self):
                self.calls = 0

            def critique(self, *, prompt, evidence):
                seen.append(evidence)
                self.calls += 1
                if self.calls == 1:
                    # 第一轮：不猜，要求取证
                    return CritiqueOutcome.of([], [ProbeRequest(
                        question="镜头1 那个人在说话还是在演示？",
                        time_start=1.0, time_end=1.6, region="center",
                        reason="画面事实里看不出",
                    )])
                return CritiqueOutcome.of([CritiqueEdit(
                    path="creative.C2_beats[0].blocking", action="set",
                    value="王姐站着说话，手扶着摊位边缘", reason="取证后确认是在说话",
                )], [])

        class Probe:
            def __init__(self):
                self.requests = []

            def look(self, video_path, request):
                self.requests.append((video_path, request))
                return ProbeFinding(request=request, answer="她在说话，手扶摊位", confidence=0.7)

        probe = Probe()
        ir = _sample_ir()
        new_ir, notes, degradations = run_review(
            ir, self._report(), Critic(), self._lib(),
            rounds=1, probe=probe, video_path="foo.mp4", probe_budget=3,
        )
        # 取证真的执行了，且带上了时间点与区域
        assert probe.requests and probe.requests[0][0] == "foo.mp4"
        assert probe.requests[0][1].region == "center"
        # 第二轮自检**看到了取证结果**（分开放，不混进原始画面事实）
        assert "定向取证结果（比上面的一次性印象更可信）" in seen[1]
        assert "她在说话" in str(seen[1]["定向取证结果（比上面的一次性印象更可信）"])
        # 取证之后提出的修改被应用
        assert new_ir["creative"]["C2_beats"][0]["blocking"].startswith("王姐站着说话")
        assert [n.applied for n in notes if n.action == "probe"] == [True]
        assert not degradations

    def test_probe_budget_is_respected(self):
        from engine.app.pipeline.review import run_review
        from engine.core.contracts import CritiqueOutcome, ProbeFinding, ProbeRequest

        class Critic:
            def critique(self, *, prompt, evidence):
                return CritiqueOutcome.of([], [
                    ProbeRequest(question=f"问题{i}", time_start=i, time_end=i + 1)
                    for i in range(5)
                ])

        class Probe:
            def __init__(self):
                self.count = 0

            def look(self, video_path, request):
                self.count += 1
                return ProbeFinding(request=request, answer="看到了点什么")

        probe = Probe()
        run_review(_sample_ir(), self._report(), Critic(), self._lib(),
                   rounds=1, probe=probe, video_path="foo.mp4", probe_budget=2)
        assert probe.count == 2, "预算只有 2 次，不能让它把 5 条都跑了"

    def test_probe_failure_is_recorded_not_fatal(self):
        from engine.app.pipeline.review import run_review
        from engine.core.contracts import CritiqueOutcome, ProbeRequest

        class Critic:
            def critique(self, *, prompt, evidence):
                return CritiqueOutcome.of([], [ProbeRequest(
                    question="看得到吗", time_start=0.0, time_end=0.5,
                )])

        class BrokenProbe:
            def look(self, video_path, request):
                raise RuntimeError("抽帧失败")

        new_ir, notes, degradations = run_review(
            _sample_ir(), self._report(), Critic(), self._lib(),
            rounds=1, probe=BrokenProbe(), video_path="foo.mp4", probe_budget=3,
        )
        assert new_ir is not None
        failed = [n for n in notes if n.action == "probe" and not n.applied]
        assert failed and "抽帧失败" in failed[0].detail
        assert not degradations, "取证失败只是这一次没查到，不是系统故障"

    def test_probes_are_ignored_without_a_probe_provider(self):
        """没给取证能力（或预算 0）时，模型要取证也不能崩。"""
        from engine.app.pipeline.review import run_review
        from engine.core.contracts import CritiqueOutcome, ProbeRequest

        class Critic:
            def critique(self, *, prompt, evidence):
                return CritiqueOutcome.of([], [ProbeRequest(
                    question="想看看", time_start=0.0, time_end=1.0,
                )])

        new_ir, notes, degradations = run_review(
            _sample_ir(), self._report(), Critic(), self._lib(), rounds=1
        )
        assert not [n for n in notes if n.action == "probe"]
        assert not degradations


class TestTimeBudget:
    """自检-重写的时间预算与子进度（用户 2026-10-07 报"卡很久"）。

    推理模型一次调用可能想很久；没有预算与子进度，用户看到的就是"卡死在 0.97"。
    """

    def _lib(self):
        from engine.core import phrases as ph

        return ph.load_library()

    def test_budget_stops_the_loop_and_keeps_the_current_version(self):
        import time

        from engine.app.pipeline.review import run_review
        from engine.core.contracts import CritiqueEdit, CritiqueOutcome

        class SlowCritic:
            def __init__(self):
                self.calls = 0

            def critique(self, *, prompt, evidence):
                self.calls += 1
                time.sleep(0.3)          # 模拟"想很久"
                # 每轮都提一条**可应用**的修改，这样循环不会因为"没得改"提前退出，
                # 真正被考验的是时间预算
                return CritiqueOutcome.of([CritiqueEdit(
                    path=f"creative.C2_beats[{self.calls % 3}].blocking",
                    action="set", value=f"第 {self.calls} 轮改写后的动作", reason="x",
                )])

        ir = _sample_ir()
        critic = SlowCritic()
        new_ir, _notes, degradations = run_review(
            ir, _report_for(ir), critic, self._lib(), rounds=5, budget_sec=0.5,
        )
        assert any("超时" in text for text in degradations), degradations
        assert critic.calls <= 4, "超时之后不该继续调模型"
        assert new_ir is not None

    def test_sub_progress_is_reported(self):
        from engine.app.pipeline.review import run_review

        stages: list[str] = []
        ir = _sample_ir()
        run_review(
            ir, _report_for(ir), _StubCritic([]), self._lib(), rounds=1,
            on_progress=lambda stage, progress: stages.append(stage),
        )
        assert any("自检与重写" in stage for stage in stages), stages

    def test_zero_budget_means_no_limit(self):
        """0 = 不设预算（离线/测试可以关掉）。"""
        from engine.app.pipeline.review import run_review
        from engine.core.contracts import CritiqueOutcome

        class Critic:
            def critique(self, *, prompt, evidence):
                return CritiqueOutcome.of([])

        ir = _sample_ir()
        _ir, _notes, degradations = run_review(
            ir, _report_for(ir), Critic(), self._lib(), rounds=1, budget_sec=0,
        )
        assert not any("超时" in text for text in degradations)


class TestEvidenceSizeCap:
    """③ 裁小自检输入（用户 2026-10-07 批准）。

    先说实话：实测证据 193 字 vs 2500 字，同一个模型 60 秒 vs 129 秒 ——
    **主要耗时在模型思考，不在输入长度**。所以这是个防爆护栏（长视频证据会膨胀）。
    """

    def test_oversized_evidence_is_trimmed(self):
        from engine.core.review import shrink_evidence

        shots = [
            {"镜头": index, "景别": "中景", "动作": "看着镜头说话并自然抬起手指向画面右侧" * 3}
            for index in range(1, 15)
        ]
        evidence = {
            "逐镜头画面事实": shots,
            "人物清单": [{"名字": "甲"}],
            "画面人数（模型看到）": [1, 2, 3],
        }
        before = len(json.dumps(evidence, ensure_ascii=False))
        after = len(json.dumps(shrink_evidence(evidence, 1200), ensure_ascii=False))
        assert before > 1200
        assert after <= 1200, after

    def test_suspicious_shots_are_kept_longest(self):
        from engine.core.review import shrink_evidence

        shots = [
            {"镜头": 1, "动作": "普通描述" * 10},
            {"镜头": 2, "动作": "这一处没看清，画面里出现文字，与上一段冲突" * 3},
        ]
        kept = shrink_evidence({"逐镜头画面事实": shots}, 200)
        text = json.dumps(kept, ensure_ascii=False)
        assert "没看清" in text, "有疑点的镜头必须优先保留"

    def test_zero_cap_means_no_trimming(self):
        from engine.core.review import shrink_evidence

        evidence = {"逐镜头画面事实": [{"镜头": i} for i in range(20)]}
        assert shrink_evidence(evidence, 0) == evidence


class TestProbeDeadline:
    """取证也要受自检的时间预算管 —— 实测它把预算冲爆过（563.7 秒 > 480 秒）。"""

    def test_probes_stop_after_the_deadline(self):
        import time

        from engine.app.pipeline.review import _run_probes
        from engine.core.contracts import ProbeRequest, ReviewNote

        calls: list[str] = []

        class _Probe:
            def look(self, video_path, request):
                calls.append(request.question)
                return type("F", (), {"answer": "看到了一点", "request": request})()

        notes: list[ReviewNote] = []
        requests = [
            ProbeRequest(time_start=float(i), time_end=float(i) + 1,
                         region="full", question=f"问题{i}", reason="核对")
            for i in range(3)
        ]
        findings, used = _run_probes(
            _Probe(), "v.mp4", requests, budget=3, notes=notes,
            deadline=time.monotonic() - 1,      # 已经超时
        )
        assert calls == [], "超时之后不许再发起新的取证"
        assert used == 0
        assert any("时间预算已用完" in note.detail for note in notes)


class TestModelCallAccounting:
    """① 每次模型调用打点：阶段计时回答不了"563 秒花在哪一次调用"。"""

    def test_calls_are_recorded_per_thread(self):
        import threading

        from engine.app.providers.mimo import record_call, take_calls

        take_calls()
        record_call("自检", "mimo-v2.6-pro-ultraspeed", 41.5, ok=True)
        record_call("取证", "mimo-v2.6-flash", 12.0, ok=False, detail="断流")
        mine = take_calls()
        assert [item["purpose"] for item in mine] == ["自检", "取证"]
        assert mine[0]["seconds"] == 41.5 and mine[0]["ok"] is True
        assert mine[1]["ok"] is False
        assert take_calls() == [], "取过一次就该清空"

        other: list[dict] = []

        def worker():
            record_call("视觉理解", "mimo-v2.6-flash", 30.0, ok=True)
            other.extend(take_calls())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert [item["purpose"] for item in other] == ["视觉理解"]
        assert take_calls() == [], "**别的线程的调用不能串到我这本账上**"

    def test_token_usage_is_recorded(self):
        """**成本要能看见**（用户 2026-10-07 问"换模型是不是变贵了"）。

        实测：同一个自检任务，pro-ultraspeed 输出 21152 token（推理 20616），
        flash 输出 12792（推理 12163）—— 快的那次**烧的 token 多 1.65 倍**。
        没有 token 账，这件事只能靠猜。
        """
        from engine.app.providers.mimo import record_call, take_calls

        take_calls()
        record_call(
            "自检", "mimo-v2.6-pro-ultraspeed", 67.0, ok=True,
            usage={
                "prompt_tokens": 1893,
                "completion_tokens": 21152,
                "completion_tokens_details": {"reasoning_tokens": 20616},
            },
        )
        record_call("取证", "mimo-v2.6-flash", 12.0, ok=True)   # 有的响应没有 usage
        mine = take_calls()
        assert mine[0]["prompt_tokens"] == 1893
        assert mine[0]["completion_tokens"] == 21152
        assert mine[0]["reasoning_tokens"] == 20616
        assert mine[1]["completion_tokens"] == 0, "没有 usage 就记 0，不要编"

    def test_timeout_and_retries_are_honoured(self):
        """``timeout_sec`` / ``retries`` 覆盖默认值（默认 180 秒 × 3 次太狠）。"""
        from engine.app.providers.mimo import MimoClient

        class _Settings:
            has_llm = True
            mimo_base_url = "http://127.0.0.1:1"
            mimo_api_key = "k"
            request_timeout_sec = 180.0
            max_retries = 2
            model_text = "m"

        client = MimoClient(_Settings())
        seen: list[float] = []

        class _Opener:
            def open(self, request, timeout=None):
                seen.append(timeout)
                raise TimeoutError("超时了")

        client._opener = _Opener()
        try:
            client.complete_json(system="s", user="u", schema={},
                                 timeout_sec=12.0, retries=0, purpose="自检")
        except Exception:
            pass
        assert seen == [12.0], f"必须只用一次、且用覆盖后的超时：{seen}"


class TestDialogueNeverGetsDropped:
    """收紧超时带来的副作用：**归属超时会把真实台词整段丢掉**（实测踩过）。

    真实模型测试「反推应产出至少一句参考台词」就是这么挂的：
    flash 单次调用能跑到 236 秒，而归属一度被卡在 120 秒 → ``attribute`` 返回空 →
    交付物里一句台词都没有、真实存在的转写被静默扔掉。
    """

    def test_fallback_keeps_the_transcript_with_an_empty_speaker(self):
        from engine.app.pipeline.attribution import fallback_turn

        turn = fallback_turn("这是转写出来的一段真实台词。", 3)
        assert turn.text == "这是转写出来的一段真实台词。"
        assert turn.speaker == "", "判不出说话人就留空，**不许猜**"
        assert turn.shot_index == 3
        assert turn.confidence > 0

    def test_fallback_is_not_used_when_attribution_worked(self):
        """有归属结果时不许走兜底（否则会把切好的多句台词压成一句）。"""
        from engine.app.pipeline.attribution import fallback_turn, has_speaker

        turns = [fallback_turn("整段转写", 1)]
        assert not has_speaker(turns[0]), "兜底出来的这一句本来就该是空说话人"


class TestReviewLoop:
    def _lib(self):
        from engine.core import phrases as ph

        return ph.load_library()

    def test_applies_a_semantic_fix_and_records_it(self):
        from engine.app.pipeline.review import run_review

        # 用完整样本：三个镜头的 location 都是「菜市场摊位前」，
        # 正是自检该清掉的那种重复
        ir = _sample_ir()
        assert ir["creative"]["C2_beats"][0]["location"]
        critic = _StubCritic([
            CritiqueEdit(path="creative.C2_beats[0].location", action="clear",
                         reason="场景由【关键一致性】钉过，节拍里重复了"),
        ])
        new_ir, notes, degradations = run_review(
            ir, _report_for(ir), critic, self._lib(), rounds=1
        )
        assert new_ir["creative"]["C2_beats"][0]["location"] == ""
        assert [n.applied for n in notes] == [True]
        assert not degradations
        # 自检模型必须看到**渲染好的交付物**与画面证据，而不是 IR
        assert "【关键一致性】" in critic.calls[0]["prompt"]
        assert "逐镜头画面事实" in critic.calls[0]["evidence"]

    def test_rolls_back_when_the_verdict_gets_worse(self):
        """**宁可少改，不可改坏**：改完判定变差就整体回滚。"""
        from engine.app.pipeline.review import run_review

        ir = _sample_ir(minimal=True)
        critic = _StubCritic([
            CritiqueEdit(path="creative.C2_beats[0].blocking", action="set",
                         value="【没绑定的素材图】里的动作", reason="试图引入不存在的素材引用"),
        ])
        new_ir, notes, degradations = run_review(
            ir, _report_for(ir), critic, self._lib(), rounds=1
        )
        assert new_ir == ir, "判定变差时必须回滚到原 IR"
        assert all(not n.applied for n in notes)
        assert any("回滚" in text for text in degradations), degradations
        assert "变差" in notes[0].detail

    def test_broken_critic_does_not_break_inference(self):
        from engine.app.pipeline.review import run_review

        ir = _sample_ir(minimal=True)
        new_ir, notes, degradations = run_review(
            ir, _report_for(ir), _BrokenCritic(), self._lib(), rounds=1
        )
        assert new_ir == ir
        assert not notes
        assert any("502" in text for text in degradations)

    def test_zero_rounds_is_a_pure_pipeline(self):
        from engine.app.pipeline.review import run_review

        ir = _sample_ir(minimal=True)
        critic = _StubCritic([CritiqueEdit(path="creative.C2_beats[0].location",
                                          action="clear", reason="x")])
        new_ir, notes, degradations = run_review(
            ir, _report_for(ir), critic, self._lib(), rounds=0
        )
        assert new_ir == ir and not notes and not degradations
        assert not critic.calls, "关掉回路时不该调用模型"

    def test_out_of_bounds_edits_are_recorded_and_loop_survives(self):
        from engine.app.pipeline.review import run_review

        ir = _sample_ir(minimal=True)
        critic = _StubCritic([
            CritiqueEdit(path="locked.B3_blocking.positions", action="set",
                         value="把站位改成居中", reason="试探越界"),
            CritiqueEdit(path="creative.C2_beats[0].dialogue.line", action="set",
                         value="我替你写台词", reason="试探代写"),
        ])
        new_ir, notes, degradations = run_review(
            ir, _report_for(ir), critic, self._lib(), rounds=1
        )
        assert new_ir == ir
        assert len(notes) == 2 and all(not n.applied for n in notes)
        assert not degradations, "越界不是系统故障，不该记降级"

    def test_clean_critique_is_reported_as_no_issue(self):
        from engine.app.pipeline.review import run_review

        ir = _sample_ir(minimal=True)
        new_ir, notes, degradations = run_review(
            ir, _report_for(ir), _StubCritic([]), self._lib(), rounds=1
        )
        assert new_ir == ir
        assert notes and notes[0].action == "none"
        assert "没提出任何修改" in notes[0].detail

