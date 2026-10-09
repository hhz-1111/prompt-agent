"""内容类型（要不要写【动作与物理】）+ 1:1 动作复刻句。

用户 2026-10-07 两条反馈，都用真实项目复现过：

1. 舞台**纯口播**的视频里出现了【动作与物理】—— 人物只是站着说话、左手做展示手势，
   「保持原有动作轨迹与幅度」在这种片子里是无对象的空约束；
2. **动作类**（老师带学员练）片子里缺一句 1:1 复刻动作的话。

判据刻意**不看 genre**：实测瑜伽带练被判成 `dialogue`、舞台口播被判成 `teaching`
—— genre 分的是题材，和"有没有动作要复刻"不是一回事。
"""

from __future__ import annotations

import pytest

from engine.core import assembler, framework as fw, phrases as ph
from engine.core.assembler import HandleResolver, looks_like_action, needs_action_physics
from engine.core.types import ContentKind
from engine.tests.test_core import _sample_ir


@pytest.fixture(scope="module")
def lib():
    return ph.load_library()


# 用户那两条视频的真实动作描述（实测原文，不是编的例子）
_TALKING_HEAD_ACTION = (
    "蓝色短袖上衣人物位于圆形舞台中央，身体正对镜头，"
    "右手持话筒于胸前，左手抬起做展示手势，站在大屏幕前。"
)
_YOGA_ACTION = "全员朝向同一侧做相同的侧伸展动作，动作整齐划一"


def _ir(blocking: str, **meta) -> dict:
    ir = _sample_ir()
    ir["creative"]["C2_beats"][0]["blocking"] = blocking
    for beat in ir["creative"]["C2_beats"][1:]:
        beat["blocking"] = blocking
    ir["meta"].update(meta)
    return ir


class TestContentKind:
    def test_model_verdict_wins(self):
        assert needs_action_physics({"meta": {"content_kind": "action"}})
        assert not needs_action_physics({"meta": {"content_kind": "speech"}})

    def test_falls_back_to_the_body_movement_words(self):
        """模型没判出来时按画面里的**动作词**兜底（宁可多写，不给动作片漏写）。"""
        assert needs_action_physics(_ir(_YOGA_ACTION))
        assert not needs_action_physics(_ir(_TALKING_HEAD_ACTION))

    def test_talking_head_gestures_are_not_body_movement(self):
        """「持话筒 / 抬手 / 展示手势 / 比划」**不算**身体动作 ——

        这正是用户那条口播片的画面描述。按小动作判会把口播片误判成动作片。
        """
        assert not looks_like_action(_ir(_TALKING_HEAD_ACTION))
        assert not looks_like_action(_ir("抬起双手做展示手势，随后比划了一下"))
        assert looks_like_action(_ir("抬腿做侧伸展，弯腰下压"))

    def test_scene_type_also_counts(self):
        assert looks_like_action({"locked": {"B1_scene": {"type": "瑜伽教室带练"}}})


class TestRendering:
    def test_speech_video_has_no_action_section(self, lib):
        doc = assembler.assemble(_ir(_TALKING_HEAD_ACTION, content_kind="speech"), lib)
        assert fw.SEC_ACTION not in [s.title for s in doc.sections]
        assert "1:1复刻" not in doc.full_text()

    def test_action_video_gets_the_one_to_one_sentence(self, lib):
        doc = assembler.assemble(_ir(_YOGA_ACTION, content_kind="action"), lib)
        assert fw.SEC_ACTION in [s.title for s in doc.sections]
        text = doc.full_text()
        assert "1:1复刻其中的肢体轨迹、身体角度、抬手弯腰幅度" in text
        assert "不出现任何其他动作" in text
        assert "动作自然连贯" in text
        # 用户 2026-09-29 要求的"不变形"那句也还在
        assert "肢体比例正常、动作连贯不变形" in text

    def test_names_the_people(self, lib):
        doc = assembler.assemble(_ir(_YOGA_ACTION, content_kind="action"), lib)
        text = doc.full_text()
        assert "人物1" in text and "的动作轨迹参考" in text

    def test_uses_the_video_handle_when_bound(self, lib):
        """有参考视频素材就用它的句柄。"""
        ir = _ir(_YOGA_ACTION, content_kind="action")
        ir["references"].append({
            "slots": ["原片"], "type": "video", "placeholder": "【原片】",
            "asset_id": "a1", "seedance_role": "reference_video",
        })
        doc = assembler.assemble(ir, lib)
        assert "的动作轨迹参考[视频" in doc.full_text()

    def test_action_slot_renders_the_readable_handle(self, lib):
        """**动作类自动登记"参考动作视频"素材位，正文写 ``[参考动作视频1]``**。

        用户 2026-10-07：「我要的是提示词，你把『参考上传的原片』改成
        **[参考动作视频1]**，我直接复制提示词上去，这个参考动作视频我会自己
        在生成平台上面上传。」
        """
        from engine.app.pipeline.to_ir import _ensure_action_video_slot

        ir = _ir(_YOGA_ACTION, content_kind="action")
        _ensure_action_video_slot(ir)
        text = assembler.assemble(ir, lib).full_text()
        assert "的动作轨迹参考[参考动作视频1]" in text
        assert "上传的原片" not in text

    def test_unregistered_legacy_data_falls_back_to_a_legal_handle(self, lib):
        """老数据（没有那条素材位）退化成 ``[视频1]`` —— 它符合句柄格式，G1 不会阻断。"""
        doc = assembler.assemble(_ir(_YOGA_ACTION, content_kind="action"), lib)
        assert "的动作轨迹参考[视频1]" in doc.full_text()
        assert "上传的原片" not in doc.full_text()

    def test_unknown_kind_uses_the_fallback(self, lib):
        """老数据没有 content_kind 时，兜底判据必须仍然生效。"""
        assert fw.SEC_ACTION in [
            s.title for s in assembler.assemble(_ir(_YOGA_ACTION), lib).sections
        ]
        assert fw.SEC_ACTION not in [
            s.title for s in assembler.assemble(_ir(_TALKING_HEAD_ACTION), lib).sections
        ]


class TestContractAndAsset:
    def test_content_kind_is_a_contract_field(self):
        spec = fw.spec_for("meta.content_kind")
        assert spec is not None
        assert set(spec.choices) == {"action", "speech"}

    def test_template_carries_the_users_wording(self, lib):
        """句子在**资产**里（单一来源），不是散在代码里。"""
        template = lib.templates["T-ACTION"]
        rendered = template.render(who="人物1", 参考视频句柄="[参考动作视频1]")
        assert rendered.startswith("人物1的动作轨迹参考[参考动作视频1]")
        for fragment in ("1:1复刻", "肢体轨迹", "身体角度", "抬手弯腰幅度",
                         "不出现任何其他动作", "动作自然连贯"):
            assert fragment in rendered
        # 两个硬表述是资产里的 mandatory_fragments，改措辞不能把它们改掉
        assert set(template.mandatory_fragments) == {"1:1复刻", "不出现任何其他动作"}
        # 开头必须是资产的**视频用途关键词** —— G12 靠它认定素材已在正文声明用途
        assert rendered.startswith("人物1的动作轨迹参考"), rendered

    def test_enum_values(self):
        assert ContentKind.ACTION.value == "action"
        assert ContentKind.SPEECH.value == "speech"


class TestHints:
    def test_legacy_action_video_gets_a_migration_hint(self):
        """老数据（没有『参考动作视频』素材位）提示重新反推一次即可补上。

        **这不再是"要去绑定"的待办** —— 用户 2026-10-07 定下：交付物直接写
        ``[参考动作视频1]``，他自己在生成平台上传。
        """
        from engine.app.api import _blocking_hints

        hints = _blocking_hints(_ir(_YOGA_ACTION, content_kind="action"))
        assert any("参考动作视频1" in hint for hint in hints)

    def test_new_data_gets_no_hint(self):
        from engine.app.api import _blocking_hints
        from engine.app.pipeline.to_ir import _ensure_action_video_slot

        ir = _ir(_YOGA_ACTION, content_kind="action")
        _ensure_action_video_slot(ir)
        assert not any("参考动作视频" in hint for hint in _blocking_hints(ir))

    def test_speech_video_gets_no_such_hint(self):
        from engine.app.api import _blocking_hints

        hints = _blocking_hints(_ir(_TALKING_HEAD_ACTION, content_kind="speech"))
        assert not any("参考动作视频" in hint for hint in hints)

    def test_video_handle_prefers_the_action_slot(self):
        from engine.app.pipeline.to_ir import _ensure_action_video_slot

        ir = _ir(_YOGA_ACTION, content_kind="action")
        ir["references"].append({
            "slots": ["别的视频"], "type": "video", "placeholder": "【别的视频】",
            "asset_id": "a1", "seedance_role": "reference_video",
        })
        _ensure_action_video_slot(ir)
        assert assembler._video_handle(ir, HandleResolver(ir)) == "[参考动作视频1]"


class TestToIrResolution:
    def test_model_value_is_written_through(self):
        from types import SimpleNamespace

        from engine.app.pipeline.to_ir import _fill_content_kind

        ir: dict = {"meta": {}}
        _fill_content_kind(ir, SimpleNamespace(semantic=SimpleNamespace(content_kind="action")))
        assert ir["meta"]["content_kind"] == "action"

    def test_fallback_when_the_model_stays_silent(self):
        from types import SimpleNamespace

        from engine.app.pipeline.to_ir import _fill_content_kind

        ir = _ir(_YOGA_ACTION)
        _fill_content_kind(ir, SimpleNamespace(semantic=SimpleNamespace(content_kind="")))
        assert ir["meta"]["content_kind"] == "action"

        ir = _ir(_TALKING_HEAD_ACTION)
        _fill_content_kind(ir, SimpleNamespace(semantic=SimpleNamespace(content_kind="")))
        assert ir["meta"]["content_kind"] == "speech"
