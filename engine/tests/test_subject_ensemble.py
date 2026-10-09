"""主体（老师）+ 群体（学员）—— 用户 2026-10-07 第 1 条。

用户原话（要点）：

> 「多人带练一起练习这种视频，你要认清楚，**主体是老师**，主体肯定位于画面最中间、
>  最显眼、最特别，而且和其他 AI 生成的学员不同！**其他学员直接由 Seedance 自己生成**，
>  只需要给她们定位成学员们，然后穿着、年龄定位好，动作跟着老师做就行了。
>  **这样就不需要那么多的提示词。**」

所以：老师照旧享受完整的人物一致性约束（多视图、五官锁定），
学员**一句话**（群体 + 年龄穿着 + 跟随谁），不占人物编号、不占参考图。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.app.pipeline import to_ir
from engine.core import assembler, framework as fw, phrases as ph
from engine.core.contracts import InferenceReport, ShotSignals, VideoInfo


@pytest.fixture(scope="module")
def lib():
    return ph.load_library()


def _report(*shots) -> InferenceReport:
    """构造一次反推：``shots`` 是 ``(subject, group, people, blocking)``。"""
    from engine.core.contracts import ShotBoundary

    understandings = []
    signals = []
    for index, (subject, group, people, blocking) in enumerate(shots, start=1):
        understandings.append(SimpleNamespace(
            shot_index=index, subject=subject, group=group, people=tuple(people),
            blocking=blocking, description=blocking, framing="全景",
            camera_angle="平视", camera="固定", location="瑜伽教室",
            expression="专注", emotion="专注",
        ))
        signals.append(ShotSignals(
            boundary=ShotBoundary(index=index, start_ms=(index - 1) * 1000, end_ms=index * 1000)
        ))
    return InferenceReport(
        video=VideoInfo(path="v.mp4", duration_sec=10.0, width=1080, height=1920,
                        fps=30.0, has_audio=True),
        shots=tuple(signals),
        understandings=tuple(understandings),
    )


# 用户那条瑜伽带练的真实画面描述
_YOGA = (
    "白衣女子位于画面前景中央偏下，其余学员分列其左后与右后两排，"
    "前后错落站位，全员朝向同一侧做相同的侧伸展动作，动作整齐划一"
)
_YOGA_SHOTS = (
    ("白色上衣女子", "40~50 岁女性，穿各色运动上衣与运动长裤",
     ["白色上衣女子", "淡紫色上衣女子", "深蓝色上衣女子", "深蓝色背心女子"], _YOGA),
    ("白色上衣女子", "40~50 岁女性，穿各色运动上衣与运动长裤",
     ["白色上衣女子", "淡紫色上衣女子", "灰色长袖上衣女子"], _YOGA),
)


class TestSubjectAndEnsemble:
    def test_subject_becomes_the_first_person(self):
        ir = fw.empty_ir(genre="teaching", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "action"
        to_ir._fill_persons(ir, _report(*_YOGA_SHOTS))
        persons = ir["locked"]["B2_person"]
        assert [p["name"] for p in persons] == ["白色上衣女子"], persons

    def test_students_become_one_group_line(self):
        ir = fw.empty_ir(genre="teaching", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "action"
        to_ir._fill_persons(ir, _report(*_YOGA_SHOTS))
        ensemble = ir["locked"]["B2b_ensemble"]
        assert ensemble["label"] == "学员们"
        assert "40~50 岁女性" in ensemble["look"]
        assert ensemble["follows"] == "人物1"

    def test_renders_one_line_not_four_people(self, lib):
        ir = fw.empty_ir(genre="teaching", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "action"
        to_ir._fill_persons(ir, _report(*_YOGA_SHOTS))
        text = assembler.assemble(ir, lib).full_text()
        assert "学员们〔40~50 岁女性，穿各色运动上衣与运动长裤〕" in text
        assert "动作全程跟随人物1" in text
        # **只有一个"人物1"**：学员不占编号、不占参考图
        assert "人物2" not in text, text[:400]
        assert text.count("形象参考") == 1

    def test_at_most_two_persons_get_references(self):
        """**硬规则：需要人物参考的人不超过 2 个**（用户 2026-10-07 反复强调）。

        > 「需要使用人物参考提示词的人物一般都不会超过 2 个，很多其他的人物都是群像！
        >  难道出现十个人物，我都要全部固定人物形象吗？我只要固定主体人物的形象就行了！」
        """
        ir = fw.empty_ir(genre="dialogue", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "speech"
        to_ir._fill_persons(ir, _report(
            ("紫色背心女子", "", ["紫色背心女子", "灰色运动外套女子", "浅蓝女童"],
             "三人站在摊位前说话"),
        ))
        persons = [p["name"] for p in ir["locked"]["B2_person"]]
        assert len(persons) <= to_ir.MAX_REFERENCE_PERSONS, persons
        ensemble = ir["locked"].get("B2b_ensemble") or {}
        assert ensemble.get("label"), "多出来的人必须变成群像，不能凭空消失"
        assert ensemble.get("look"), "群像至少要给年龄或穿着（用户要的就是这个）"

    def test_two_person_dialogue_keeps_both(self):
        """两个人时两个都留 —— 双人对话/双人采访是常见形态。"""
        ir = fw.empty_ir(genre="dialogue", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "speech"
        to_ir._fill_persons(ir, _report(
            ("甲女子", "", ["甲女子", "乙女子"], "两人面对面说话"),
        ))
        assert len(ir["locked"]["B2_person"]) == 2

    def test_no_subject_means_no_forced_grouping(self):
        ir = fw.empty_ir(genre="teaching", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "action"
        to_ir._fill_persons(ir, _report(("", "", ["甲女子", "乙女子"], "两人各自做动作")))
        # 两个人：都在人物清单里（≤2），不需要群像
        assert len(ir["locked"]["B2_person"]) == 2
        assert not (ir["locked"].get("B2b_ensemble") or {}).get("label")

    def test_arrangement_does_not_lock_the_group_size(self, lib):
        """有群体时**不写「固定 N 位人物出镜，不做增减」** —— 学员是生成出来的。"""
        ir = fw.empty_ir(genre="teaching", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "action"
        report = _report(*_YOGA_SHOTS)
        to_ir._fill_persons(ir, report)
        to_ir._fill_blocking(ir, report)
        positions = "、".join(ir["locked"]["B3_blocking"]["positions"])
        assert "固定 4 位人物出镜" not in positions
        assert "其余为群体角色" in positions


class TestSubjectDedupe:
    """主体那条逻辑**必须按身份键认领**，不能按字符串比较。

    用户 2026-10-07 实测：「最新上传的视频把人物穿的衣服也认成了一个人物，
    变成了人物1」—— 交付物里是：

        人物1〔白色T恤女子 · 画面中央〕：人物1形象参考[图1]…
        人物2〔白色T恤女子〕：**人物1**形象参考[图2]…      ← 同一个人，引用还指错

    根因：模型给的主体是「白色T恤盘发女子」，而跨镜头去重后的名字是
    「白色T恤女子」（`compact_person_name` 把发型压掉了）。字符串不相等，
    于是同一个人被插了两遍。
    """

    def test_subject_written_differently_is_not_duplicated(self):
        ir = fw.empty_ir(genre="studio", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "speech"
        to_ir._fill_persons(ir, _report((
            "白色T恤盘发女子", "", ["白色T恤盘发女子"], "白色T恤盘发女子面对镜头说话",
        )))
        persons = ir["locked"]["B2_person"]
        assert len(persons) == 1, [p["name"] for p in persons]
        assert persons[0]["name"] == "白色T恤女子"

    def test_identity_key_is_what_matches(self):
        assert to_ir._identity_key("白色T恤盘发女子") == to_ir._identity_key("白色T恤女子")

    def test_display_names_never_collide(self):
        """显示名撞车 → 退回完整名/人物N。

        同名会让别名映射把后来那条的引用指到第一个人身上
        （实测渲出「人物2…人物1形象参考[图2]」）。
        """
        # 两个不同的人，但压出来的显示名一样（颜色+服装+性别）
        names = ["白色T恤盘发女子", "白色T恤长直发女子"]
        displays = to_ir.disambiguate_display_names(names)
        assert len(set(displays)) == 2, displays
        assert displays[0] == "白色T恤盘发女子"      # 撞名就退回完整名
        # 不撞名时保持简洁
        assert to_ir.disambiguate_display_names(["紫色连衣裙长卷发女子"]) == ["紫色连衣裙女子"]
    def test_ensemble_is_a_contract_field(self):
        assert fw.spec_for("locked.B2b_ensemble") is not None
        assert fw.spec_for("locked.B2b_ensemble.label") is not None
        assert fw.spec_for("locked.B2b_ensemble.look") is not None
        assert fw.spec_for("locked.B2b_ensemble.follows") is not None
        assert fw.spec_for("locked.B2b_ensemble.mode") is not None


class TestCrowdIsAlwaysAGroup:
    """**观众也是群像**（用户 2026-10-07 第二次指出）。

    真实那条视频（演播厅舞台互动，2 个镜头）：

    - 镜1：subject=「蓝色短袖上衣男子」（**台上讲话的男老师**）、
      group=「观众席人群，以中老年为主，穿各色休闲上衣，部分人举手机拍摄」；
    - 镜2：subject=「深蓝色碎花上衣女子」（观众席里举手机的那位）、
      people=4 位女士。

    修前的结果是 **4 个人物**（全是镜2 的女士，**老师被截断丢了**），
    观众也没合成群体 —— 用户看到的就是"老师被认成了深蓝色上衣女子"。
    根因三条：① 截断发生在认主角之前；② 群体化被错误地限定在动作类；
    ③ 各镜头 subject 取众数，平票时随机（那时是用 set 去重）。
    """

    STAGE = (
        ("蓝色短袖上衣男子", "观众席人群，以中老年为主，穿各色休闲上衣，部分人举手机拍摄",
         ["蓝色短袖上衣男子"], "男子立于舞台中央面向观众；观众背对镜头就坐"),
        ("深蓝色碎花上衣女子", "中老年观众，穿各色花色上衣，多人举起手机拍摄",
         ["深蓝色碎花上衣女子", "红色碎花上衣女子", "酒红色上衣女子", "扎马尾女子"],
         "女子坐于座位上高举右臂持手机；周围观众坐姿举手机拍摄"),
    )

    def _ir(self, content_kind="speech"):
        ir = fw.empty_ir(genre="studio", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = content_kind
        return ir

    def test_the_teacher_survives_and_the_audience_becomes_one_group(self):
        ir = self._ir()
        to_ir._fill_persons(ir, _report(*self.STAGE))
        persons = [p["name"] for p in ir["locked"]["B2_person"]]
        assert persons == ["蓝色短袖上衣男子"], persons
        ensemble = ir["locked"]["B2b_ensemble"]
        assert ensemble["label"] == "观众们"
        assert "中老年" in ensemble["look"]
        assert ensemble["mode"] == "watch", "观众是看的，不是跟着做动作的"
        assert ensemble["follows"] == ""

    def test_audience_line_does_not_invent_their_actions(self, lib):
        """群像句**不许替他们编行为**（观众是看的、后景学员在做瑜伽 —— 一刀切就会写错）。

        他们的动作写在【镜头节拍】的逐镜描述里（用户 2026-10-07 指出）。
        """
        ir = self._ir()
        to_ir._fill_persons(ir, _report(*self.STAGE))
        text = assembler.assemble(ir, lib).full_text()
        assert "观众们〔" in text
        audience_line = text.split("观众们", 1)[1].split("。", 1)[0]
        assert "跟随" not in audience_line, audience_line
        assert "观看" not in audience_line, "不能替观众断言他们在做什么"
        assert "按画面里的位置与动作自然存在" in audience_line, audience_line

    def test_it_works_for_speech_videos_too(self):
        """**不能只在动作类上折叠群体** —— 舞台互动的观众同样是群像。"""
        ir = self._ir(content_kind="speech")
        to_ir._fill_persons(ir, _report(*self.STAGE))
        assert ir["locked"]["B2b_ensemble"]["label"] == "观众们"

    def test_subject_of_the_first_shot_wins_a_tie(self):
        """各镜头 subject 平票时取**最早镜头**的那个 —— 开场镜头给主角。"""
        assert to_ir._video_subject(_report(*self.STAGE)) == "蓝色短袖上衣男子"

    def test_speaker_beats_shot_order(self):
        """有台词时**说话人**优先（他是主角，最硬的证据）。"""
        report = _report(
            ("甲女子", "", ["甲女子", "乙女子"], "两人站着说话"),
            ("乙女子", "", ["甲女子", "乙女子"], "乙女子面向镜头说话"),
        )
        report.transcript = (
            type("T", (), {"speaker": "乙女子", "text": "大家好"})(),
        )
        assert to_ir._video_subject(report) == "乙女子"

    def test_the_subject_is_never_the_wrong_person(self):
        """**主体不能被认错人** —— 名字写法不同、身份键还会撞（实测踩过）。

        这条视频里模型给的主角是「紫色长裙女子」，而清单里还有四位「…女子」；
        早期实现按"公共子串 ≥2"匹配，把主角认成了「丁女子」（公共串只有"女子"两个字）。
        """
        ir = self._ir()
        to_ir._fill_persons(ir, _report((
            "紫色长裙女子", "",
            ["甲女子", "乙女子", "丙女子", "丁女子", "紫色长裙女子"],
            "五个人站在一起，紫色长裙女子在中间说话",
        )))
        persons = [p["name"] for p in ir["locked"]["B2_person"]]
        assert len(persons) <= to_ir.MAX_REFERENCE_PERSONS, persons
        assert "紫" in persons[0], persons      # 主角是"紫…"，不是随便一个"…女子"
        assert not persons[0].startswith(("甲", "乙", "丙", "丁")), persons

    def test_a_small_cast_is_not_collapsed(self):
        """**不许乱合并**：两个人的剧情片里"几个人就是几个人"。"""
        ir = self._ir(content_kind="speech")
        to_ir._fill_persons(ir, _report(
            ("甲女子", "", ["甲女子", "乙女子"], "两人面对面说话"),
        ))
        assert not (ir["locked"].get("B2b_ensemble") or {}).get("label")

    def test_students_still_follow(self):
        """学员是**跟着做**的（和观众分开）—— 带练那条视频不能被改坏。"""
        ir = fw.empty_ir(genre="teaching", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "action"
        to_ir._fill_persons(ir, _report((
            "白色上衣女子", "40~50 岁女性，穿各色运动上衣与长裤",
            ["白色上衣女子", "淡紫色上衣女子", "深蓝色上衣女子", "灰色上衣女子"],
            "白衣女子位于前景中央，其余学员分列两侧做相同的侧伸展动作",
        )))
        ensemble = ir["locked"]["B2b_ensemble"]
        assert ensemble["label"] == "学员们"
        assert ensemble["mode"] == "follow"
        assert ensemble["follows"] == "人物1"


class TestModelDecidesTheStructure:
    """**人物结构由模型判断，不由代码猜**（用户 2026-10-08：「我做的不是 agent 吗？」）。

    三次"一堆人物参考"的共同根因都是：我在代码里用关键词/硬规则猜结构，
    模型换一种说法就失效（「其余数人」不命中"学员/观众"词表）。
    既然已经用了多模态理解，结构判断就该由模型给出，代码只做校验。
    """

    def _report(self, structure, people, subject="", blocking=""):
        from types import SimpleNamespace

        from engine.core.types import Genre

        report = _report((subject, "", people, blocking or "画面里有人"))
        # 与 SemanticSummary 同形：genre 要留给代码兜底（眨眼节拍按 genre 取）
        report.semantic = SimpleNamespace(genre=Genre.STUDIO, people_structure=structure)
        return report

    def test_the_models_answer_wins(self):
        """模型说了谁是主角、谁是群像，就按它说的来。"""
        ir = fw.empty_ir(genre="studio", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "speech"
        report = self._report(
            {
                "lead": "紫色短袖上衣女子",
                "second": "",
                "distinct_others": [],
                "crowd_label": "背景人物",
                "crowd_look": "背景中多名20~40岁女性，穿各色瑜伽背心，正在做弓步伸展",
                "crowd_behavior": "own_action",
            },
            ["浅紫色短袖上衣女子", "粉色背心女子", "黑色上衣女子", "紫色背心女子"],
            subject="浅紫色短袖上衣女子",
            blocking="浅紫色短袖上衣女子面向镜头站立；其余数人在她身后做动作",
        )
        to_ir._fill_persons(ir, report)
        persons = [p["name"] for p in ir["locked"]["B2_person"]]
        assert persons == ["紫色短袖上衣女子"], persons
        ensemble = ir["locked"]["B2b_ensemble"]
        assert ensemble["label"] == "背景人物"
        assert "瑜伽背心" in ensemble["look"]
        assert ensemble["mode"] == "watch", "模型没说 follow，就不许写成跟随"
        assert ir["meta"]["people_structure_source"] == "model"

    def test_it_locks_two_when_the_model_says_so(self):
        """模型说第二位也要锁定（双人对话）—— 那就锁两个。"""
        ir = fw.empty_ir(genre="dialogue", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "speech"
        report = self._report(
            {
                "lead": "紫色上衣女子", "second": "灰色外套女子",
                "distinct_others": [], "crowd_label": "", "crowd_look": "",
                "crowd_behavior": "none",
            },
            ["紫色上衣女子", "灰色外套女子", "路人甲"],
            subject="紫色上衣女子",
        )
        to_ir._fill_persons(ir, report)
        persons = [p["name"] for p in ir["locked"]["B2_person"]]
        assert "灰色外套女子" in persons, persons
        assert len(persons) <= to_ir.MAX_REFERENCE_PERSONS

    def test_a_wrong_name_from_the_model_is_ignored(self):
        """模型答的名字在画面标签里对不上 → 当作它看错了，整块不用、退回兜底。"""
        ir = fw.empty_ir(genre="studio", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "speech"
        report = self._report(
            {
                "lead": "画面里根本没有的人", "second": "", "distinct_others": [],
                "crowd_label": "群像", "crowd_look": "x", "crowd_behavior": "none",
            },
            ["紫色上衣女子", "粉色背心女子", "黑色上衣女子"],
            subject="紫色上衣女子",
        )
        to_ir._fill_persons(ir, report)
        assert ir["meta"]["people_structure_source"] == "code"
        assert [p["name"] for p in ir["locked"]["B2_person"]] == ["紫色上衣女子"]

    def test_speakers_survive_even_if_the_model_forgot_them(self):
        """模型没把说话人列进主角/第二位 —— 说话人也必须在人物清单里。"""
        ir = fw.empty_ir(genre="studio", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "speech"
        report = self._report(
            {
                "lead": "紫色上衣女子", "second": "", "distinct_others": [],
                "crowd_label": "观众们", "crowd_look": "中老年", "crowd_behavior": "watch",
            },
            ["紫色上衣女子", "灰色外套女子"],
            subject="紫色上衣女子",
        )
        report.transcript = (type("T", (), {"speaker": "灰色外套女子", "text": "你好"})(),)
        to_ir._fill_persons(ir, report)
        persons = [p["name"] for p in ir["locked"]["B2_person"]]
        assert "灰色外套女子" in persons, persons

    def test_code_fallback_is_declared_honestly(self):
        """模型没给结构时**如实说明这是推断的**，并告诉用户一句话能改。"""
        ir = fw.empty_ir(genre="studio", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "speech"
        report = _report((
            "紫色上衣女子", "", ["紫色上衣女子", "粉色背心女子", "黑色上衣女子"],
            "三个人在教室里",
        ))
        to_ir._fill_persons(ir, report)
        assert ir["meta"]["people_structure_source"] == "code"
        note = to_ir.people_structure_note(ir, report)
        assert note and "推断" in note
        assert "改写功能" in note, "要告诉用户怎么改，而不是让他自己猜"


class TestNoInternalMeasurementInPrompt:
    def test_no_box_coordinates_anywhere(self, lib):
        """用户 2026-10-07 第 3 条：提示词里不要「检测框中心约 x≈0.145」。"""
        ir = fw.empty_ir(genre="teaching", shot_mode="one_take", aspect_ratio="9:16")
        ir["meta"]["content_kind"] = "action"
        report = _report(*_YOGA_SHOTS)
        # 逐镜头的检测框量测（左中右）
        report.shots = tuple(
            ShotSignals(
                boundary=signal.boundary,
                blocking=(SimpleNamespace(x_ratio=0.145, side="left"),),
                composition="left",
            )
            for signal in report.shots
        )
        to_ir._fill_persons(ir, report)
        to_ir._fill_blocking(ir, report)
        text = assembler.assemble(ir, lib).full_text()
        assert "x≈" not in text
        assert "检测框" not in text
        assert "0.145" not in text
        # 位置本身照旧写着
        assert "画面左侧" in text or "左侧" in text
