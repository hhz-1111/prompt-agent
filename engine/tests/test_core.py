"""领域内核：框架契约、位置标注、短语工程、组装、闸门。

测试按"契约 → 位置 → 短语 → 组装 → 闸门"的依赖顺序组织，
与代码分层一一对应。
"""

from __future__ import annotations

import json

import pytest

from engine.core import assembler, framework as fw, gates, phrases as ph, positions
from engine.core.assembler import _is_cjk
from engine.core.types import (
    Color,
    Grade,
    OutputForm,
    PositionCategory,
    Provenance,
    ConfidenceBasis,
    Span,
    UNKNOWN,
)


# ==========================================================================
# 契约
# ==========================================================================
class TestFrameworkContract:
    def test_schema_paths_unique(self):
        """契约里不能有重复路径 —— 重复就意味着两条规则打架。"""
        paths = [spec.path for spec in fw.FIELDS.values()]
        assert len(paths) == len(set(paths))

    def test_every_child_path_extends_parent(self):
        for spec in fw.FIELDS.values():
            if "." not in spec.path:
                continue
            parent = spec.path.rsplit(".", 1)[0].split("[")[0]
            assert parent in fw.FIELDS, f"{spec.path} 的父路径 {parent} 不在契约中"

    def test_four_sections_are_present(self):
        """四段式结构是冻结契约，缺一段就是结构升级。"""
        titles = {spec.apply_section for spec in fw.FIELDS.values()}
        assert {"参考素材", "关键一致性", "剧情", "镜头节拍", "对白与口型", "禁止项"} <= titles

    def test_empty_ir_has_every_root_key(self):
        ir = fw.empty_ir()
        assert set(ir) == {"framework_version", "meta", "references", "locked",
                           "creative", "forbidden", "delivery"}
        assert ir["framework_version"] == fw.FRAMEWORK_VERSION

    def test_locked_fields_reject_writes(self):
        """红线 2：站位、视线、钩子类型不可改。"""
        for path in (
            "locked.B3_blocking.positions",
            "locked.B3_blocking.eye_line",
            "locked.B3_blocking.occlusion",
            "creative.C2_beats[].beat_role",
            "locked.B1_scene.anchor",
        ):
            decision = fw.can_write(path, value="随便改")
            assert not decision.allowed, f"{path} 不应允许写入"
            assert "锁定位" in decision.reason

    def test_param_fields_enforce_whitelist(self):
        """档位位只能从白名单切换，不能自由填。"""
        ok = fw.can_write("creative.C2_beats[].framing", value="中景")
        assert ok.allowed
        bad = fw.can_write("creative.C2_beats[].framing", value="超广角航拍")
        assert not bad.allowed
        assert bad.requires_whitelist

    def test_forbidden_can_grow_but_not_shrink(self):
        """禁止项可增不可减。"""
        grow = fw.can_write("forbidden", value=[1, 2, 3], current=[1, 2])
        assert grow.allowed
        shrink = fw.can_write("forbidden", value=[1], current=[1, 2, 3])
        assert not shrink.allowed
        assert "可增不可减" in shrink.reason

    def test_free_creative_fields_are_writable(self):
        for path in (
            "creative.C1_logline",
            "creative.C2_beats[].blocking",
            "creative.C2_beats[].dialogue.line",
            "creative.C2_beats[].dialogue.speaker",
        ):
            assert fw.can_write(path, value="用户写的").allowed, path

    def test_get_set_roundtrip(self):
        ir = fw.empty_ir()
        ir["locked"]["B2_person"] = [{"name": "王姐", "multi_view_ref": "王姐多视图"}]
        assert fw.get_in(ir, "locked.B2_person[0].name") == "王姐"
        fw.set_in(ir, "locked.B2_person[0].name", "李姐")
        assert fw.get_in(ir, "locked.B2_person[0].name") == "李姐"
        assert fw.get_in(ir, "locked.B2_person[9].name", "缺省") == "缺省"

    def test_normalize_path_strips_index(self):
        assert fw.normalize_path("locked.B2_person[0].name") == "locked.B2_person[].name"
        assert fw.normalize_path("references[12].asset_id") == "references[].asset_id"
        assert fw.spec_for("locked.B2_person[7].name") is not None

    def test_unknown_vs_empty_are_distinguishable(self):
        """红线 7「不猜」：没反推出来 与 用户明确留空 必须能区分。"""
        ir = fw.empty_ir()
        assert ir["locked"]["B1_scene"]["type"] == UNKNOWN
        assert ir["creative"]["C1_logline"] == ""


# ==========================================================================
# 位置与三色
# ==========================================================================
class TestPositions:
    def test_only_three_colors_ever(self):
        """PRD F2.2：不出现第四种状态。"""
        ir = _sample_ir()
        table = positions.build_positions(ir)
        assert {p.color for p in table} <= {Color.BLUE, Color.ORANGE, Color.GRAY}

    def test_reference_slots_are_blue(self):
        ir = _sample_ir()
        table = positions.build_positions(ir)
        blues = table.of_color(Color.BLUE)
        assert blues
        assert all(p.path.startswith("references") for p in blues)
        assert any(p.path.endswith("].asset_id") for p in blues)

    def test_creative_content_is_orange(self):
        ir = _sample_ir()
        table = positions.build_positions(ir)
        oranges = {p.path for p in table.of_color(Color.ORANGE)}
        assert "creative.C1_logline" in oranges
        assert "creative.C2_beats[0].dialogue.line" in oranges

    def test_locked_positions_are_gray_and_not_editable(self):
        ir = _sample_ir()
        table = positions.build_positions(ir)
        pos = table.by_path("locked.B3_blocking.positions")
        assert pos is not None
        assert pos.color is Color.GRAY
        assert not pos.editable
        assert pos.category is PositionCategory.LOCK

    def test_apply_target_is_reported(self):
        """框架 §10.3：必须能回答"改了这个槽位，输出第几段会变"。"""
        ir = _sample_ir()
        table = positions.build_positions(ir)
        assert table.by_path("creative.C2_beats[0].dialogue.line").apply_target == fw.SEC_DIALOGUE
        assert table.by_path("locked.B1_scene.anchor").apply_target == fw.SEC_LOCKED

    def test_provenance_drives_confirm_list(self):
        ir = _sample_ir()
        prov = {
            "creative.C1_logline": Provenance(
                confidence=0.42, basis=ConfidenceBasis.INFERRED,
                span=Span(0, 2000), note="归纳推断",
            ),
        }
        table = positions.build_positions(ir, prov)
        pending = {p.path for p in table.pending_confirm}
        assert "creative.C1_logline" in pending
        assert table.by_path("creative.C1_logline").provenance.disposition == "blank"

    def test_blue_manifest_lists_all_references(self):
        ir = _sample_ir()
        manifest = positions.blue_slot_manifest(ir)
        assert len(manifest) == len(ir["references"])
        assert [item["type"] for item in manifest] == [
            "multi_view", "multi_view", "scene", "audio", "audio",
        ]
        assert all(item["bound"] for item in manifest)
        assert [item["seedance_role"] for item in manifest] == [
            "reference_image", "reference_image", "reference_image",
            "reference_audio", "reference_audio",
        ]

    def test_blue_manifest_flags_unbound_slot(self):
        """红线 1：不编造参考素材。没绑定的必须显式标出来。"""
        ir = _sample_ir()
        ir["references"][0]["asset_id"] = None
        manifest = positions.blue_slot_manifest(ir)
        assert manifest[0]["bound"] is False
        assert manifest[1]["bound"] is True

    def test_scalar_list_keeps_container_path(self):
        """标量数组是一个逻辑值，必须整体产出在契约声明的路径上。

        拆成 ``positions[0]`` 会让契约里那条 LOCK 规则永远匹配不上，
        G10 就会漏检站位改动 —— 而 PRD 把站位列为致命级维度。
        """
        ir = _sample_ir()
        flat = dict(fw.flatten(ir))
        assert "locked.B3_blocking.positions" in flat
        assert flat["locked.B3_blocking.positions"] == [
            "李姐始终在画面右侧", "王姐始终在画面左侧",
        ]
        assert "locked.B2_person[0].name" in flat

    def test_apply_targets_reverse_lookup(self):
        ir = _sample_ir()
        mapping = positions.apply_targets(ir)
        assert fw.SEC_LOGLINE in mapping
        assert "creative.C1_logline" in mapping[fw.SEC_LOGLINE]


# ==========================================================================
# 短语工程
# ==========================================================================
class TestPhraseEngine:
    @staticmethod
    @pytest.fixture(scope="class")
    def lib():
        return ph.load_library()

    def test_library_loads_with_expected_assets(self, lib):
        assert lib.version == "mod-phrases-v1"
        assert len(lib.templates) >= 10
        assert len(lib.forbidden_groups) == 9
        assert len(lib.downgrade_rules) == 6

    def test_mandatory_dimensions_match_corpus_evidence(self, lib):
        """必写维度必须由实证覆盖率支撑，不能凭感觉。"""
        assert "blink" in lib.mandatory_dimensions
        assert "skin" in lib.mandatory_dimensions
        assert "framing" in lib.mandatory_dimensions
        for key in lib.mandatory_dimensions:
            coverage = lib.dimension(key).get("evidence_coverage")
            assert coverage is not None and coverage >= 0.6, key

    def test_fixed_templates_render_verbatim(self, lib):
        tpl = lib.templates["T-SKIN"]
        assert tpl.is_fixed
        assert "毛孔" in tpl.variant("full")
        assert "无过度磨皮" in tpl.variant("brief")

    @pytest.mark.parametrize("key,fragment", [
        # 逐字钉住 TEMPLATES §2.4 的三档原句。
        # 曾经 build_phrases.py 是手打这两句的，漏了「隔」还多出空格，
        # 于是引擎输出的眨眼句与历史成功提示词**逐字对不上** ——
        # 而固定模块的全部价值就在于逐字。
        ("3-4s", "人物每隔3-4秒自然快速眨眼一次（每次约0.1秒），融入说话节奏，眨眼频率自然。"),
        ("5s", "人物每隔5秒自然快速眨眼一次（每次约0.1秒），融入对话节奏，眨眼频率自然。"),
        ("3s", "人物每隔3秒自然快速眨眼一次（每次约0.1秒），融入动作节奏，眨眼频率自然。"),
    ])
    def test_blink_variants_are_verbatim_from_spec(self, lib, key, fragment):
        assert lib.templates["T-BLINK"].variant(key) == fragment

    def test_template_render_fails_loudly_on_missing_slot(self, lib):
        with pytest.raises(ValueError, match="缺少槽位"):
            lib.templates["T-PERSON"].render(角色名="王姐")

    def test_template_render_fills_slots(self, lib):
        text = lib.templates["T-PERSON"].render(角色名="王姐", 多视图句柄="[图1]")
        assert "王姐形象参考[图1]" in text

    @pytest.mark.parametrize("text,expected", [
        ("人物每 5 秒眨眼一次，每次约 0.1 秒", Grade.A_QUANTIFIED),
        ("鼻翼和脸颊有真实的毛孔细节", Grade.B_GEOMETRIC),
        ("100%透明字幕", Grade.A_QUANTIFIED),
        ("禁止出现bgm、logo、水印、时间戳", Grade.D_ENUMERATED),
        ("禁止出现的杂元素有：bgm，logo，水印，时间戳", Grade.D_ENUMERATED),
        ("服装保持一致", Grade.E_GENERIC),
        ("尽量保持人物一致", Grade.F_WEAK),
    ])
    def test_constraint_grading(self, lib, text, expected):
        grade, _ = lib.grade(text)
        assert grade is expected

    def test_weak_grade_wins_over_strong(self, lib):
        """一句泛化表述会污染整条约束 —— 不能因为别处有量化就被平均掉。"""
        text = "100%相同，但尽量参考一下"
        grade, hit = lib.grade(text)
        assert grade is Grade.F_WEAK
        assert hit

    def test_downgrade_detected(self, lib):
        issues = lib.detect_downgrades(["动作1:1复刻原片，参考一下就行"])
        assert not issues, "原短语仍在，属于并存描述，不应误报"

        issues = lib.detect_downgrades(["动作参考一下原片"])
        assert issues
        assert issues[0].code == "F3.5"
        assert "1:1复刻" in issues[0].title

    def test_downgrade_ignores_plain_speech(self, lib):
        """台词里出现"参考"不该误报。

        创作区是用户的表达，G14 只体检约束段落 —— 否则误报会让用户
        不再信任闸门，那比漏报更糟。
        """
        doc = assembler.assemble(
            _ir_with_dialogue("镜头2（李姐，语气温和）：“你先参考一下我的动作”"), lib
        )
        issues = ph.scan_texts(lib, doc.text_lines())
        assert not issues

    def test_scan_reports_both_weak_and_downgrade(self, lib):
        issues = ph.scan_texts(lib, ["尽量自然一点", "动作参考一下原片"])
        codes = {i.code for i in issues}
        assert "F3.4" in codes
        assert "F3.5" in codes

    def test_forbidden_default_groups(self, lib):
        ids = lib.default_forbidden_ids()
        assert any(i.startswith("D0.") for i in ids)
        assert any(i.startswith("D7.") for i in ids)
        assert not any(i.startswith("D5.") for i in ids), "D5 属于低频组，不该默认全选"

    def test_capacity_lookup_is_config_driven(self, lib):
        assert lib.capacity_for("seedance-2.0")["image"] == [1, 9]
        assert lib.capacity_for("seedance-2.5")["image"] == [1, 30]
        with pytest.raises(KeyError):
            lib.capacity_for("seedance-9.9")

    def test_mandatory_coverage_flags_missing_dimension(self, lib):
        ir = fw.empty_ir()
        coverage = ph.check_mandatory_dimensions(ir, lib)
        assert coverage
        assert all(not item.covered for item in coverage)
        assert all(item.suggestion for item in coverage)


# ==========================================================================
# 组装
# ==========================================================================
class TestAssembler:
    @staticmethod
    @pytest.fixture(scope="class")
    def lib():
        return ph.load_library()

    def test_assembles_all_sections(self, lib):
        doc = assembler.assemble(_sample_ir(), lib)
        titles = [s.title for s in doc.sections]
        # 样例是**菜市场对话**（人物只是抬手说话，没有身体动作）→ 没有【动作与物理】。
        # 该段的出现条件见 test_content_kind.py：只有动作/体式/教学动作类才有。
        assert titles == [
            fw.SEC_LOCKED, fw.SEC_LOGLINE,
            fw.SEC_BEATS, fw.SEC_DIALOGUE, fw.SEC_AUDIO,
            fw.SEC_FORBIDDEN,
        ]

    def test_person_names_are_replaced_by_labels(self, lib):
        """用户 2026-09-29（第三轮）：人物N 的特征已经在〔〕里标定过，下文一律用人物N。

        否则「灰色运动外套女子」这种长描述会在正文里被抄 5 遍
        （人物句、镜头节拍、台词、音色、动作各一次）。
        """
        text = assembler.assemble(_sample_ir(), lib).full_text()
        # 〔…〕 里保留"人物N 是谁"的定义，**不能**一起换掉
        assert "人物1〔王姐〕" in text
        assert "人物1形象参考[图1]" in text, "人物句本身要用标签"
        # 其他段落也用标签，不再重复长名字
        assert "王姐声音音色参考[音频1]" not in text
        assert "人物1声音音色参考[音频1]" in text
        assert "角色名" not in text  # 占位兜底断言：确保上面看的确实是渲染结果
        assert "人物2" in text

    def test_person_label_map_is_longest_first(self):
        """短写法是长写法的子串时，先换长的那一次，否则会把长名切碎。"""
        ir = {"locked": {"B2_person": [
            {"name": "红色开衫女子"}, {"name": "开衫女子"},
        ]}}
        aliases = assembler.person_label_map(ir)
        assert aliases[0][0] == "红色开衫女子"
        assert assembler.unify_person_labels(
            ["红色开衫女子走进来"], aliases
        ) == ["人物1走进来"]

    def test_person_label_map_ignores_single_char_names(self):
        """单字写法做替换会把别的词一起吃掉，必须拒收。"""
        ir = {"locked": {"B2_person": [{"name": "女"}]}}
        assert assembler.person_label_map(ir) == ()

    def test_action_physics_is_positive_constraint(self, lib):
        """动作与物理是**正面**约束（要保住什么），与 D0.6 的禁止项互补。

        **只对动作类写**（用户 2026-10-07）：样例本身是对话类，所以要显式标成
        action 才该出现 —— 这同时也是"判据生效"的一个反证。
        """
        assert assembler.assemble(_sample_ir(), lib).section(fw.SEC_ACTION) is None

        ir = _sample_ir()
        ir["meta"]["content_kind"] = "action"
        doc = assembler.assemble(ir, lib)
        section = doc.section(fw.SEC_ACTION)
        assert section is not None
        text = "\n".join(section.lines)
        assert "动作轨迹与幅度" in text
        assert "不变形" in text
        # 动作类还必须带 1:1 复刻句（用户原话）
        assert "1:1复刻其中的肢体轨迹、身体角度、抬手弯腰幅度" in text
        assert "不出现任何其他动作" in text

    def test_reference_handles_are_numbered_per_role(self, lib):
        """图片与音频各自独立编号 —— 不能混成一个序列。"""
        ir = _sample_ir()
        handles = assembler.HandleResolver(ir)
        assert handles.all_tokens == [
            "[图1]", "[图2]", "[图3]", "[音频1]", "[音频2]",
        ]
        # 句柄必须在正文里被用到（用户 2026-09-29 删掉了【参考素材】清单段，
        # 所以句柄唯一的落点就是正文里的用途声明句）。
        body = assembler.assemble(ir, lib).body_text
        assert [t for t in handles.all_tokens if t in body] == handles.all_tokens

    def test_no_reference_material_section(self, lib):
        """用户 2026-09-29 明确要求：交付物里不要再出现【参考素材】这一段。"""
        doc = assembler.assemble(_sample_ir(), lib)
        assert fw.SEC_REFS not in [s.title for s in doc.sections]
        assert "【参考素材】" not in doc.full_text()
        assert "【参考素材】" not in doc.body_text

    def test_section_titles_are_not_part_of_body(self, lib):
        """段落标题写作【关键一致性】，不能因此被 G1 判成占位符残留。"""
        doc = assembler.assemble(_sample_ir(), lib)
        assert "【关键一致性】" not in doc.body_text
        assert "【关键一致性】" in doc.full_text()
        assert not gates.g1_placeholder_residue(doc, _sample_ir()).blocking

    def test_placeholders_fully_converted(self, lib):
        """红线 6：交付物不留【】。"""
        doc = assembler.assemble(_sample_ir(), lib)
        text = doc.full_text()
        assert "【场景参考图】" not in text
        assert "[图3]" in text
        assert "[音频1]" in text

    def test_param_header_follows_delivery_mode(self, lib):
        """PRD §6.4.5：手动模式保留参数声明，直调模式移除。"""
        ir = _sample_ir()
        manual = assembler.assemble(ir, lib)
        assert manual.param_header, "Phase 1 手动模式必须保留参数声明"

        ir2 = _sample_ir()
        ir2["delivery"]["mode"] = "phase2_api"
        api = assembler.assemble(ir2, lib)
        assert not api.param_header, "Phase 2 直调模式必须移除参数声明"

    def test_char_count_excludes_header_and_english(self, lib):
        """按 PRD §5 示例的 486/500 口径：只数中文字符，英文画质预设不计入。"""
        doc = assembler.assemble(_sample_ir(), lib)
        assert doc.char_count == sum(1 for ch in doc.body_text if _is_cjk(ch))
        assert doc.total_char_count > doc.char_count, "画质预设是英文，应计入总字符但不计入中文预算"

    def test_three_output_forms_are_consistent(self, lib):
        doc = assembler.assemble(_sample_ir(), lib)
        plain = doc.render(OutputForm.PLAIN)
        markdown = doc.render(OutputForm.MARKDOWN)
        as_json = json.loads(doc.render(OutputForm.JSON))
        for section in doc.sections:
            assert section.text in plain
            assert section.text in markdown
        assert as_json["sections"][0]["lines"] == doc.sections[0].lines
        assert as_json["char_count"] == doc.char_count

    def test_dialogue_is_always_quoted(self, lib):
        ir = _sample_ir()
        ir["creative"]["C2_beats"][0]["dialogue"]["line"] = "你这气色也太好了吧"
        doc = assembler.assemble(ir, lib)
        assert '“你这气色也太好了吧”'.replace("“", '"').replace("”", '"') in doc.full_text()

    def test_dialogue_without_line_is_skipped(self, lib):
        ir = _sample_ir()
        for beat in ir["creative"]["C2_beats"]:
            beat["dialogue"]["line"] = ""
        doc = assembler.assemble(ir, lib)
        assert doc.section(fw.SEC_DIALOGUE) is None

    def test_duplicate_person_label_is_collapsed(self, lib):
        """``人物2（人物2）`` 要收成 ``人物2``。

        实测踩过（生产链路重推）：自检回路自己写「人物2（深灰色上衣女子）微笑」，
        同一轮人称统一把括号里的名字也换成 ``人物2`` → 交付物出现
        「人物2（人物2）微笑」。号不同**不合并**（那是有信息量的）。
        """
        lines = assembler.unify_person_labels(
            ["人物2（深灰色上衣女子）微笑", "人物1（人物2）站在一起"],
            [("深灰色上衣女子", "人物2")],
        )
        assert lines[0] == "人物2微笑", lines[0]
        assert lines[1] == "人物1（人物2）站在一起", lines[1]

    def test_camera_word_that_is_not_a_movement_goes_elsewhere(self, lib):
        """模型的 ``camera`` 不一定是运镜：「过肩」是机位/构图，不是镜头怎么动。

        实测踩过：节拍行渲染成「中景，平视机位拍摄，**过肩**，…」，
        读起来像"镜头在过肩移动"。现在按词分槽位。
        """
        ir = _sample_ir()
        ir["creative"]["C2_beats"][0]["camera"] = "过肩"
        ir["creative"]["C2_beats"][1]["camera"] = "跟拍"
        lines = assembler.assemble(ir, lib).section(fw.SEC_BEATS).lines
        assert "过肩构图" in lines[0], lines[0]
        assert "跟拍" in lines[1], lines[1]

    def test_scene_is_written_once_per_place(self, lib):
        """场景已在【关键一致性】钉过，节拍行不该逐镜头重复。

        实测踩过：一条四镜头片子里「同一室内演播厅舞台，背景弧形大屏与灯带不变」
        被抄了 3 遍；去掉重复后中文从 979 → 919 字。
        """
        doc = assembler.assemble(_sample_ir(), lib)
        lines = doc.section(fw.SEC_BEATS).lines
        assert lines[0].count("菜市场摊位前") == 1, lines[0]
        for line in lines[1:]:
            assert "菜市场摊位前" not in line, f"同一个场景又写了一遍：{line}"

    def test_new_scene_is_still_written(self, lib):
        """**真的换地方**时必须写出来 —— 去重不能把新信息也去掉。"""
        ir = _sample_ir()
        ir["creative"]["C2_beats"][2]["location"] = "茶楼雅间靠窗座位"
        lines = assembler.assemble(ir, lib).section(fw.SEC_BEATS).lines
        assert "菜市场摊位前" in lines[0]
        assert "茶楼雅间靠窗座位" in lines[2], lines[2]

    def test_same_scene_detection_rules(self):
        """判"同一个地方"的两条依据：模型明说 + 共享连续地点名。"""
        same = assembler._looks_like_same_scene
        assert same("同一室内演播厅舞台，背景弧形大屏与灯带不变", "室内演播厅，背景为带弧形大屏的舞台")
        assert same("工作室前台接待台前，台面有平板电脑", "普拉提工作室前台大厅，浅色木地板")
        # 换了地方：共享不到 4 字连续地点名
        assert not same("茶楼雅间靠窗座位", "菜市场摊位前，浅色地砖")

    def test_no_seconds_in_beat_prefix(self, lib):
        """用户 2026-09-29（第三轮）：交付物里不要再出现「0-9.102秒画面：」这种前缀。

        一镜到底也不再写时间戳 —— 时长在生成平台是独立控件，
        而【对白与口型】本来就是按「镜头N（人物）」写的，前缀得对得上。
        """
        for mode in ("one_take", "multi_shot", "same_angle_segments"):
            ir = _sample_ir()
            ir["meta"]["shot_mode"] = mode
            doc = assembler.assemble(ir, lib)
            assert "秒画面" not in doc.full_text(), f"{mode} 仍写出了秒数"
            lines = doc.section(fw.SEC_BEATS).lines
            assert lines[0].startswith("镜头1："), lines[0]

    def test_no_tone_word_for_a_silent_video(self, lib):
        """没有台词的片子不写「语气X」。

        语气是"话是怎么说的"。参考视频里没人说话时写「语气专注」，
        是在描述不存在的声音（实测踩过一条纯静音的动作示范片）。
        情绪本身不丢 —— 它就在同一行的 expression 里。
        """
        ir = _sample_ir()
        for beat in ir["creative"]["C2_beats"]:
            beat["dialogue"]["line"] = ""
        lines = assembler.assemble(ir, lib).section(fw.SEC_BEATS).lines
        assert lines, "节拍行不该为空"
        assert not any("语气" in line for line in lines), lines

    def test_english_is_stripped_not_the_whole_line(self, lib):
        """带英文词的中文句子要**保留**，只摘掉英文片段。

        实测踩过：判据原来是"整行含 ≥3 个连续拉丁字母就整行丢"，
        于是「场景始终为[图3]的演播厅环境，背景有大屏PILATES标识，保持同一位置
        和光线氛围」这句**整句消失** —— 而场景锚定是 LOCK 永不可裁字段。
        """
        keep = assembler._drop_english(
            ["场景始终为[图3]的演播厅环境，背景有大屏PILATES标识，保持同一位置和光线氛围"]
        )
        assert keep == [
            "场景始终为[图3]的演播厅环境，背景有大屏标识，保持同一位置和光线氛围"
        ]
        # 纯英文行仍然整行丢掉（当初要的就是这个效果）
        assert assembler._drop_english(["画质：auto white balance slightly cool"]) == []
        assert assembler._drop_english(["logo"]) == []
        # 没有英文的行原样保留
        assert assembler._drop_english(["100%透明字幕"]) == ["100%透明字幕"]

    def test_tone_word_stays_when_someone_speaks(self, lib):
        lines = assembler.assemble(_sample_ir(), lib).section(fw.SEC_BEATS).lines
        assert any("语气" in line for line in lines), lines

    def test_multi_shot_uses_shot_prefix(self, lib):
        ir = _sample_ir()
        ir["meta"]["shot_mode"] = "multi_shot"
        doc = assembler.assemble(ir, lib)
        lines = doc.section(fw.SEC_BEATS).lines
        assert lines[0].startswith("镜头1：")
        assert lines[1].startswith("镜头2：")

    def test_unbound_reference_is_reported(self, lib):
        ir = _sample_ir()
        ir["references"][0]["asset_id"] = None
        doc = assembler.assemble(ir, lib)
        assert doc.warnings
        assert any("未绑定" in w for w in doc.warnings)

    def test_speech_estimation(self):
        assert assembler.estimate_speech_seconds("") == 0.0
        assert assembler.estimate_speech_seconds("十个字十个字十个字") == pytest.approx(1.8)

    def test_requires_quoted_normalizes_forms(self):
        assert assembler.ensure_quoted("你好") == '"你好"'
        assert assembler.ensure_quoted('"你好"') == '"你好"'
        assert assembler.ensure_quoted("“你好”") == '"你好"'

    def test_split_recommends_by_shot_mode(self, lib):
        ir = _sample_ir()
        plans, _ = assembler.plan_split(ir, lib)
        assert all(p.strategy.value == "S1" for p in plans)
        ir["meta"]["shot_mode"] = "multi_shot"
        plans, _ = assembler.plan_split(ir, lib)
        assert all(p.strategy.value == "S2" for p in plans)
        assert len(plans) == 3

    def test_split_blocks_when_one_take_over_limit(self, lib):
        ir = _sample_ir()
        for beat in ir["creative"]["C2_beats"]:
            beat["duration_sec"] = 12.0
        _, issues = assembler.plan_split(ir, lib)
        codes = {i.code for i in issues}
        assert "SPLIT-S1-OVERFLOW" in codes
        assert any(i.severity == "block" for i in issues)
        assert any("S4" in i.suggestion for i in issues)

    def test_compression_is_never_silent(self, lib):
        """压缩必须有记录、有理由，且绝不触碰永不可裁的内容。

        用**动作类**样例：它的段落最全（含【动作与物理】），压缩路径才走得完整。
        """
        ir = _sample_ir(full_forbidden_library=True)
        ir["meta"]["content_kind"] = "action"
        entries = assembler.compress(ir, lib).entries
        assert entries
        assert all(entry.reason for entry in entries)
        assert all(entry.saved_chars >= 0 for entry in entries)
        steps = " ".join(entry.what for entry in entries)
        for protected in ("站位", "音色", "台词"):
            assert protected not in steps, f"{protected} 不该出现在可压缩清单里"

    def test_compression_actually_reduces_length(self, lib):
        """压缩不是建议，是真执行 —— 交付物必须真的变短。"""
        ir = _sample_ir(full_forbidden_library=True)
        ir["meta"]["content_kind"] = "action"
        plain = assembler.assemble(ir, lib, compress_to_budget=False)
        squeezed = assembler.assemble(ir, lib, compress_to_budget=True)
        assert squeezed.char_count < plain.char_count
        assert squeezed.compression, "压缩必须留下记录"

    def test_person_longtail_is_the_first_big_win(self, lib):
        """人物长尾是 PRD 明列的第一个大杠杆，压缩后应只留硬约束句。"""
        ir = _sample_ir(full_forbidden_library=True)
        doc = assembler.assemble(ir, lib)
        locked = doc.section(fw.SEC_LOCKED).body
        if locked.count("形象参考") and doc.compression:
            used = {entry.what for entry in doc.compression}
            if assembler.PERSON_LONGTAIL_TEXT in used:
                assert "用于" not in locked
                assert "锁定人物五官脸型" in locked

    def test_protected_content_survives_compression(self, lib):
        """永不可裁：站位、音色、台词原文、D0、D7。"""
        ir = _sample_ir(full_forbidden_library=True)
        doc = assembler.assemble(ir, lib)
        text = doc.body_text
        assert "位置全程禁止变动" in text, "站位不可裁"
        assert "百分百还原" in text, "音色不可裁"
        assert "100%透明字幕" in text, "D0 不可裁"
        assert "禁止修改对白台词" in text, "D7 不可裁"
        assert "哪有啊，我就是每天在家练普拉提" in text, "台词原文不可裁"


# ==========================================================================
# 闸门
# ==========================================================================
class TestGates:
    @staticmethod
    @pytest.fixture(scope="class")
    def lib():
        return ph.load_library()

    def test_all_fourteen_gates_present(self, lib):
        report = gates.run_gates(_sample_ir(), assembler.assemble(_sample_ir(), lib), lib)
        assert [r.code for r in report.results] == [f"G{i}" for i in range(1, 15)]

    def test_clean_ir_passes(self, lib):
        """合规输入必须全绿 —— 闸门在这里证明自己不会误报。

        用最小可交付形态：中文 339 字、单人单场景。
        完整的双人长句版本中文约 750 字，会触发 G7（见 test_g7_blocks_over_char_budget）
        —— 那是 PRD 风险 4 的真实冲突，不是闸门的错。
        """
        ir = _sample_ir(minimal=True)
        report = gates.run_gates(ir, assembler.assemble(ir, lib), lib)
        assert report.verdict == "ALL_GREEN", report.summary_lines()
        assert report.deliverable

    def test_g1_blocks_on_unbound_placeholder(self, lib):
        ir = _sample_ir()
        ir["locked"]["B1_scene"]["anchor"] = "场景始终为【没绑定的场景图】环境"
        doc = assembler.assemble(ir, lib)
        result = gates.g1_placeholder_residue(doc, ir)
        assert result.blocking
        assert "没绑定的场景图" in result.detail

    def test_g1_blocks_on_unresolved_handle(self, lib):
        """素材引用没解析成 [图N] 时必须阻断。

        实测盲区：人物改名后 ``multi_view_ref`` 没同步，
        交付物里出现 ``形象参考[王姐多视图]``。它既不是 【】，
        看起来又"像"一个引用，但已经是废的。
        """
        ir = _sample_ir()
        ir["locked"]["B2_person"][0]["multi_view_ref"] = "不存在的素材名"
        doc = assembler.assemble(ir, lib)
        result = gates.g1_placeholder_residue(doc, ir)
        assert result.blocking
        assert "未解析成句柄" in result.detail
        assert result.evidence, "必须给出具体的未解析引用"

    def test_g1_passes_when_all_handles_resolve(self, lib):
        ir = _sample_ir()
        doc = assembler.assemble(ir, lib)
        assert not gates.g1_placeholder_residue(doc, ir).blocking

    def test_g2_blocks_on_quota_overflow(self, lib):
        ir = _sample_ir()
        ir["references"] = [
            {"slots": [f"素材{i}"], "type": "multi_view", "placeholder": "",
             "asset_id": f"a{i}", "seedance_role": "reference_image"}
            for i in range(12)
        ]
        result = gates.g2_quota(ir, lib)
        assert result.blocking
        assert "12" in result.detail

    def test_g2_allows_more_images_on_25(self, lib):
        ir = _sample_ir()
        ir["meta"]["target_model"] = "seedance-2.5"
        ir["references"] = [
            {"slots": [f"素材{i}"], "type": "multi_view", "placeholder": "",
             "asset_id": f"a{i}", "seedance_role": "reference_image"}
            for i in range(12)
        ]
        assert not gates.g2_quota(ir, lib).blocking

    def test_g3_blocks_on_mixed_exclusive_modes(self, lib):
        ir = _sample_ir()
        ir["delivery"]["first_frame_mode"] = "first_last_frame"
        doc = assembler.assemble(ir, lib)
        result = gates.g3_exclusive_modes(doc, ir)
        assert result.blocking
        assert "三选一" in result.suggestion

    def test_g5_autofixes_quotes(self, lib):
        ir = _sample_ir()
        ir["creative"]["C2_beats"][0]["dialogue"]["line"] = "没有引号的台词"
        result = gates.g5_dialogue_quotes(assembler.assemble(ir, lib), ir)
        assert result.autofixed
        assert not result.blocking

    def test_g6_blocks_when_speaker_has_no_voice(self, lib):
        ir = _sample_ir()
        ir["locked"]["B6_audio"] = []
        result = gates.g6_lipsync_binding(ir)
        assert result.blocking
        assert "王姐" in result.detail and "李姐" in result.detail

    def test_g7_warns_over_char_budget(self, lib):
        """字数预算是**建议值**而非硬上限，处置是压缩+标注。

        24 条真实爆款视频压到极限仍在 650~1200 字 ——
        因为永不可裁的部分（D0+D7、站位、音色、台词）与节拍行本身就很长。
        所以这是告警，不是阻断。
        """
        ir = _sample_ir(pad_beats_to=12)
        doc = assembler.assemble(ir, lib)
        result = gates.g7_char_budget(doc, ir, lib)
        assert result.status == gates.WARN
        assert "永不可裁" in result.suggestion
        assert doc.compression, "超预算时必须留下压缩记录，不能静默裁剪"
        assert "已按优先级压掉" in result.detail or doc.char_count <= 500

    def test_g7_passes_within_budget(self, lib):
        ir = _sample_ir(minimal=True)
        doc = assembler.assemble(ir, lib)
        assert gates.g7_char_budget(doc, ir, lib).status == gates.OK

    def test_g8_blocks_on_density(self, lib):
        """G8 是最容易被忽略但最致命的一项。"""
        ir = _sample_ir()
        ir["meta"]["total_duration_sec"] = 4.0
        for beat in ir["creative"]["C2_beats"]:
            beat["duration_sec"] = 2.0
        result = gates.g8_density(ir)
        assert result.blocking
        assert "0.27" in result.detail
        # 建议必须给两个方向：减节拍，或延长目标时长
        assert "拉" in result.suggestion and "合并" in result.suggestion

    def test_g8_target_duration_override_takes_effect(self, lib):
        """目标时长是用户可以调的 —— 延长它应当立刻让密度达标。"""
        ir = _sample_ir()
        for beat in ir["creative"]["C2_beats"]:
            beat["duration_sec"] = 2.0          # 节拍总和 6 秒
        ir["meta"]["total_duration_sec"] = 12.0  # 目标拉到 12 秒
        result = gates.g8_density(ir)
        assert not result.blocking, result.detail

    def test_g8_passes_on_low_density(self, lib):
        ir = _sample_ir()
        for beat in ir["creative"]["C2_beats"]:
            beat["duration_sec"] = 6.0
        assert not gates.g8_density(ir).blocking

    def test_g9_blocks_on_strategy_mismatch(self, lib):
        ir = _sample_ir()
        ir["meta"]["shot_mode"] = "multi_shot"
        ir["delivery"]["strategy"] = "S1"
        doc = assembler.assemble(ir, lib)
        doc.strategy = type(doc.strategy).S1_ONE_TAKE
        result = gates.g9_split_strategy(doc, ir)
        assert result.blocking

    def test_g10_detects_locked_mutation(self, lib):
        baseline = _sample_ir()
        modified = _sample_ir()
        modified["locked"]["B3_blocking"]["positions"] = ["王姐在右", "李姐在左"]
        result = gates.g10_locked_diff(modified, baseline)
        assert result.blocking
        assert "positions" in result.evidence[0]

    def test_g10_ignores_free_field_changes(self, lib):
        baseline = _sample_ir()
        modified = _sample_ir()
        modified["creative"]["C1_logline"] = "完全换掉的剧情"
        modified["creative"]["C2_beats"][0]["dialogue"]["line"] = "换掉的台词"
        assert not gates.g10_locked_diff(modified, baseline).blocking

    def test_g11_blocks_when_hook_lost(self, lib):
        baseline = _sample_ir()
        modified = _sample_ir()
        for beat in modified["creative"]["C2_beats"]:
            beat["beat_role"] = "demo"
        result = gates.g11_hook_preserved(modified, baseline)
        assert result.blocking
        assert "钩子" in result.detail

    def test_g14_ignores_dialogue_content(self, lib):
        """台词是"提及"不是"主张"：里面出现降级短语不该判 BLOCK。

        实测踩过：组装器包台词用的是 **ASCII 双引号**（`"…"`），
        而 `strip_quoted` 只认「」“”『』，于是台词被当成正文主张扫，
        G14 直接 block：「强约束被降级：1:1复刻 → 照着做就行」——
        用户一句正常台词就让整条片子不可交付。
        """
        ir = _sample_ir()
        ir["creative"]["C2_beats"][0]["dialogue"]["line"] = "你照着做就行"
        doc = assembler.assemble(ir, lib)
        assert '"你照着做就行"' in doc.full_text(), "台词应该被 ASCII 引号包着"
        assert not gates.g14_constraint_strength(doc, lib).blocking

    def test_g1_sees_long_placeholder_residue(self, lib):
        """素材名很长时，【】残留不能被漏掉（解析与检测必须同一套正则）。

        实测踩过：解析用 ``{1,24}``、残留检测用 ``{1,40}``，
        28 字素材名两头都不匹配 → 正文带着裸【…】交付，G1 还报「残留=0」。
        """
        ir = _sample_ir()
        long_name = "素" * 28
        ir["references"][0]["slots"] = [long_name]
        ir["locked"]["B2_person"][0]["multi_view_ref"] = long_name
        doc = assembler.assemble(ir, lib)
        assert f"【{long_name}】" not in doc.full_text(), "长素材名也要能解析成句柄"
        assert gates.g1_placeholder_residue(doc, ir).status == gates.OK
        # 真的残留时必须报出来
        doc.sections[0].lines.append(f"人物形象参考【{long_name}】")
        assert gates.g1_placeholder_residue(doc, ir).blocking

    def test_g12_reports_undeclared_reference_honestly(self, lib):
        """素材没在正文里声明用途时要报出来，但**不能说"已自动补"**。

        实测踩过：原来恒返回 `autofixed=True / 已自动补默认声明 N 项`，
        可 doc 与 ir 一个字节都没动 —— 报告在说谎，用户以为已经补好了。
        """
        ir = _sample_ir()
        ir["references"].append({
            "slots": ["未声明素材"], "type": "multi_view", "placeholder": "【未声明素材】",
            "asset_id": "x", "seedance_role": "reference_image",
        })
        doc = assembler.assemble(ir, lib)
        result = gates.g12_ref_roles(doc, ir)
        assert result.status == gates.WARN
        assert "未声明素材" in result.detail
        assert not result.autofixed, "它并没有改任何东西，不能声称已自动修复"
        assert "交付物未被自动改写" in result.detail
        assert "人物形象参考" in result.suggestion

    def test_g13_warns_on_missing_dimensions(self, lib):
        """空 IR 上真正缺的是景别/机位/站位这些**没有固定句**的维度。

        **眨眼节拍不算缺**（实测踩过）：它的句子在正文里是**无条件固定项**，
        没有人物条目时照样写着，而 IR 内容挂在 ``locked.B2_person[]`` 下 ——
        只看 IR 就会报「缺：眨眼节拍 / 整条未反推到」，两句话都不成立。
        """
        ir = fw.empty_ir()
        doc = assembler.assemble(ir, lib)
        result = gates.g13_mandatory_dimensions(doc, ir, lib)
        assert result.status == gates.WARN
        assert "站位" in result.detail and "景别" in result.detail
        assert "眨眼节拍" not in result.detail
        assert "眨眼" in doc.body_text, "正文里确实有这句，不该判缺失"

    def test_g13_audits_assembled_prompt_not_ir(self, lib):
        """IR 里 blink 存的是键名 "5s"，逐字表述是组装时才展开的。

        所以必写维度必须审最终提示词 —— 只审 IR 会把合规提示词误判成缺维度。
        """
        ir = _sample_ir()
        coverage = ph.check_mandatory_dimensions(
            ir, lib, prompt_text=assembler.assemble(ir, lib).body_text
        )
        covered = {item.key for item in coverage if item.covered}
        assert {"blink", "skin", "person_consistency", "blocking", "framing"} <= covered

    def test_g14_blocks_on_downgrade(self, lib):
        ir = _sample_ir()
        ir["creative"]["C2_beats"][0]["blocking"] = "动作参考一下原片"
        doc = assembler.assemble(ir, lib)
        result = gates.g14_constraint_strength(doc, lib)
        assert result.blocking
        assert "1:1复刻" in result.detail

    def test_blocked_report_is_not_deliverable(self, lib):
        """红线 10：闸门未全绿时不许声称可交付。"""
        ir = _sample_ir(pad_beats_to=12)
        # 用确定性的阻断源：把强约束降级（G14），这是最常见也最致命的事故
        ir["creative"]["C2_beats"][0]["blocking"] = "动作参考一下原片"
        doc = assembler.assemble(ir, lib)
        report = gates.run_gates(ir, doc, lib)
        assert not report.deliverable
        assert report.verdict == "BLOCKED"
        assert {r.code for r in report.blockers} == {"G14"}
        assert report.summary_lines()[-1] == "→ 阻断交付"

    def test_warning_only_report_stays_deliverable(self, lib):
        """只有告警（字数超预算、必写维度缺）时仍可交付，但要如实标注。

        刻意用**补齐节拍**的方式构造确定超预算的输入，而不是靠某个固定字数
        刚好越线 —— 这样预算调高调低都不会把这条测试变成"偶然通过"
        （预算已经从 500 调到 800 一次了）。
        """
        ir = _sample_ir(pad_beats_to=12)
        doc = assembler.assemble(ir, lib)
        assert doc.over_budget > 0, "构造的输入本应超预算"
        report = gates.run_gates(ir, doc, lib)
        assert not report.blockers
        assert report.deliverable
        assert report.verdict in ("PASS_WITH_WARNINGS", "ALL_GREEN")
        assert report.warnings

    def test_summary_lines_render(self, lib):
        ir = _sample_ir(minimal=True)
        report = gates.run_gates(ir, assembler.assemble(ir, lib), lib)
        lines = report.summary_lines()
        assert len(lines) == 15, "14 项结论 + 1 行总判定"
        assert lines[0].startswith("G1")
        assert lines[-1].startswith("→")

    def test_summary_reports_budget_overrun(self, lib):
        """超预算时如实报出，不粉饰。"""
        ir = _sample_ir(pad_beats_to=12)
        doc = assembler.assemble(ir, lib)
        report = gates.run_gates(ir, doc, lib)
        g7 = report.by_code("G7")
        assert g7 is not None
        assert g7.status == gates.WARN
        assert "超出" in g7.detail and "压掉" in g7.detail
        assert report.summary_lines()[-1].startswith("→")

    def test_within_budget_reports_ok(self, lib):
        """预算内的输入 G7 必须是 ok —— 否则"全绿"永远不可达。"""
        ir = _sample_ir(minimal=True)
        doc = assembler.assemble(ir, lib)
        assert doc.over_budget == 0
        assert gates.run_gates(ir, doc, lib).by_code("G7").status == gates.OK

    def test_blink_and_skin_are_unconditional(self, lib):
        """眨眼节拍与皮肤质感是**无条件固定项**（用户 2026-09-29 指定）。

        它们是最容易被模型做错的默认项，所以不挂在人物条目下面 ——
        反推不出人物时也必须出现。
        """
        ir = _sample_ir(minimal=True)
        ir["locked"]["B2_person"] = []          # 故意让反推不出人物
        text = assembler.assemble(ir, lib).full_text()
        assert "眨眼" in text, "没有人物条目时眨眼节拍也必须写"
        assert "毛孔" in text and "磨皮" in text, "皮肤质感句也必须写"

    def test_blink_and_skin_not_duplicated_per_person(self, lib):
        """两个人物的片子，这两句各出现一次，不是每人一遍。

        按**行数**统计，不按子串出现次数 ——
        「眨眼」在一句里自然出现两次（「快速眨眼一次」+「眨眼频率自然」）。
        """
        ir = _sample_ir()
        assert len(ir["locked"]["B2_person"]) >= 2
        lines = assembler.assemble(ir, lib).full_text().splitlines()
        assert sum(1 for line in lines if "眨眼" in line) == 1
        assert sum(1 for line in lines if "毛孔" in line) == 1


class TestPersonRenameCascade:
    """改人物名必须带动所有派生引用。

    实测踩过：改名后交付物同时出现「王姐形象参考[图1]」和
    「穿紫色连衣裙的女子声音音色参考[音频1]」，且闸门 G6 误报
    "王姐缺音色绑定" —— 音色其实有，只是还挂在旧名字下。
    """

    def test_rename_updates_audio_binding(self):
        """按**名字**匹配音色，不按索引 —— B6_audio 顺序来自台词说话人，
        与 B2_person 下标不保证一致。"""
        ir = _sample_ir()
        # 故意把音色放在与人物下标不同的位置，验证不依赖索引
        ir["locked"]["B6_audio"] = [
            {"speaker": "李姐", "audio_ref": "李姐音频", "volume_mode": "normal"},
            {"speaker": "王姐", "audio_ref": "王姐音频", "volume_mode": "normal"},
        ]
        touched = fw.cascade_person_rename(ir, 0, "王姐", "小美")
        by_speaker = {t["speaker"]: t["audio_ref"] for t in ir["locked"]["B6_audio"]}
        assert by_speaker["小美"] == "小美音频"
        assert by_speaker["李姐"] == "李姐音频"
        assert any("B6_audio[1]" in path for path in touched)

    def test_rename_updates_reference_slots(self):
        ir = _sample_ir()
        touched = fw.cascade_person_rename(ir, 0, "王姐", "小美")
        slots = [s for ref in ir["references"] for s in ref.get("slots", [])]
        assert "小美多视图" in slots
        assert not any("王姐" in s for s in slots)
        assert any("references[" in path for path in touched)

    def test_rename_updates_dialogue_speaker(self):
        ir = _sample_ir()
        fw.cascade_person_rename(ir, 0, "王姐", "小美")
        speakers = {
            (b.get("dialogue") or {}).get("speaker")
            for b in ir["creative"]["C2_beats"]
        }
        assert "小美" in speakers
        assert "王姐" not in speakers

    def test_rename_keeps_others_untouched(self):
        ir = _sample_ir()
        fw.cascade_person_rename(ir, 0, "王姐", "小美")
        assert ir["locked"]["B2_person"][1]["name"] == "李姐"
        assert any("李姐" in s for ref in ir["references"] for s in ref.get("slots", []))

    def test_noop_on_same_or_empty_name(self):
        ir = _sample_ir()
        assert fw.cascade_person_rename(ir, 0, "王姐", "王姐") == []
        assert fw.cascade_person_rename(ir, 0, "", "小美") == []


class TestBlockingRenderNoDuplication:
    def test_fixed_tail_is_added_when_missing(self):
        doc = assembler.assemble(_sample_ir(), ph.load_library())
        body = doc.section(fw.SEC_LOCKED).body
        assert "位置全程禁止变动" in body

    def test_fixed_tail_is_not_duplicated(self):
        """数据里已带那句时不能再叠一遍 —— 实测渲出过叠字。"""
        ir = _sample_ir()
        ir["locked"]["B3_blocking"]["positions"] = [
            "李姐始终在画面右侧", "王姐始终在画面左侧", "位置全程禁止变动",
        ]
        doc = assembler.assemble(ir, ph.load_library())
        body = doc.section(fw.SEC_LOCKED).body
        assert body.count("位置全程禁止变动") == 1


# ==========================================================================
# 测试夹具
# ==========================================================================
def _ir_with_dialogue(line: str) -> dict:
    """把一句台词塞进样本 IR，用于验证创作区不被约束体检误伤。"""
    ir = _sample_ir()
    ir["creative"]["C2_beats"][1]["dialogue"]["line"] = line
    return ir


def _sample_ir(
    *,
    full_forbidden_library: bool = False,
    minimal: bool = False,
    pad_beats_to: int = 0,
) -> dict:
    """一条结构完整的"菜市场双人对话"，对齐 PRD §5 的示例。

    - 默认：合规产线的常见形态（禁止项取用户指定的默认组 D0 + D1 + D7）。
    - ``full_forbidden_library=True``：把整库禁止项都塞进去。
    - ``minimal=True``：最小可交付形态，用于验证"闸门对合规输入不误报"。
    - ``pad_beats_to=N``：把节拍补到 N 个（每个带台词与动作）。
      **用来构造"确定超预算"的输入** —— 别再靠"某个固定字数刚好越线"，
      因为预算值会调（500 → 800），靠字数越线的测试会随预算变化偶然通过或失败。
    """
    lib = ph.load_library()
    ir = fw.empty_ir(genre="dialogue", shot_mode="one_take")
    ir["meta"]["total_duration_sec"] = 12.0
    ir["references"] = [
        {"slots": ["王姐多视图"], "type": "multi_view", "placeholder": "【王姐多视图】",
         "asset_id": "asset-a", "seedance_role": "reference_image", "prep_step": 1},
        {"slots": ["李姐多视图"], "type": "multi_view", "placeholder": "【李姐多视图】",
         "asset_id": "asset-b", "seedance_role": "reference_image", "prep_step": 2},
        {"slots": ["菜市场场景图"], "type": "scene", "placeholder": "【场景参考图】",
         "asset_id": "asset-c", "seedance_role": "reference_image", "prep_step": 3},
    ]
    ir["locked"]["B1_scene"] = {
        "type": "菜市场",
        "anchor": "场景始终为【场景参考图】的菜市场环境，保持同一位置和光线氛围，镜头无跳切，背景不虚化",
        "lighting": "室内光自然通透",
        "cross_shot_consistency": True,
    }
    ir["locked"]["B2_person"] = [
        {"name": "王姐", "multi_view_ref": "王姐多视图", "lock_strength": "standard",
         "wardrobe": "着装100%还原【王姐多视图】", "skin": "brief", "blink": "5s"},
        {"name": "李姐", "multi_view_ref": "李姐多视图", "lock_strength": "standard",
         "wardrobe": "着装100%还原【李姐多视图】", "skin": "brief", "blink": "5s"},
    ]
    ir["locked"]["B3_blocking"] = {
        "positions": ["李姐始终在画面右侧", "王姐始终在画面左侧"],
        "occlusion": "", "eye_line": "", "off_screen": "",
    }
    ir["locked"]["B4_camera"] = {
        "shot_mode": "one_take", "rig": "固定机位全程不变",
        "framing_set": ["中景", "中景"], "ratio": "9:16", "transition_logic": "",
    }
    ir["locked"]["B5_quality"] = {"preset": "P1", "skin_text": "brief", "grain": ""}
    ir["locked"]["B6_audio"] = [
        {"speaker": "王姐", "audio_ref": "王姐音频", "volume_mode": "normal"},
        {"speaker": "李姐", "audio_ref": "李姐音频", "volume_mode": "normal"},
    ]
    ir["references"].extend([
        {"slots": ["王姐音频"], "type": "audio", "placeholder": "【王姐音频】",
         "asset_id": "asset-d", "seedance_role": "reference_audio", "prep_step": 4},
        {"slots": ["李姐音频"], "type": "audio", "placeholder": "【李姐音频】",
         "asset_id": "asset-e", "seedance_role": "reference_audio", "prep_step": 5},
    ])
    ir["creative"]["C1_logline"] = (
        "菜市场摊位前，王姐看到李姐气色大变，惊讶追问，李姐道出在家练普拉提的实情。"
    )
    ir["creative"]["C2_beats"] = [
        {
            "shot_id": 1, "framing": "中景", "camera": "平视", "location": "菜市场摊位前",
            "beat_role": "hook", "duration_sec": 4.0,
            "blocking": "王姐略微睁大眼睛，自然抬起手指向李姐",
            "expression": "", "emotion": "惊讶",
            "dialogue": {"speaker": "王姐", "line": "你这气色也太好了吧",
                         "tone": "惊讶", "lip_sync": "", "pause_at": []},
        },
        {
            "shot_id": 2, "framing": "中景", "camera": "过肩", "location": "菜市场摊位前",
            "beat_role": "demo", "duration_sec": 4.0,
            "blocking": "李姐轻轻摆手", "expression": "面带微笑", "emotion": "温和",
            "dialogue": {"speaker": "李姐", "line": "哪有啊，我就是每天在家练普拉提",
                         "tone": "温和", "lip_sync": "", "pause_at": []},
        },
        {
            "shot_id": 3, "framing": "中近景", "camera": "平视", "location": "菜市场摊位前",
            "beat_role": "proof", "duration_sec": 4.0,
            "blocking": "李姐双手从腰部向两侧滑移", "expression": "", "emotion": "自信",
            "dialogue": {"speaker": "李姐", "line": "你摸摸我这腰线",
                         "tone": "轻快", "lip_sync": "", "pause_at": []},
        },
    ]
    if full_forbidden_library:
        ir["forbidden"] = lib.forbidden_items()
    else:
        # 合规产线的默认集合 = 用户指定的默认组（D0 + D1 + D7）
        ir["forbidden"] = lib.forbidden_items()

    if pad_beats_to > 0:
        # 补齐节拍到指定数量，用来构造**确定超预算**的输入。
        # 每节拍带台词 + 动作 + 表情，是真实长片的形态。
        template = ir["creative"]["C2_beats"][-1]
        while len(ir["creative"]["C2_beats"]) < pad_beats_to:
            index = len(ir["creative"]["C2_beats"]) + 1
            clone = json.loads(json.dumps(template, ensure_ascii=False))
            clone["shot_id"] = index
            clone["blocking"] = f"人物{index}号在摊位前抬手比划，身体转向画面左侧" * 2
            clone["expression"] = "神情专注，目光平视前方"
            clone["emotion"] = "专注"
            clone["dialogue"]["line"] = f"这是第{index}句台词，用来把正文撑到超预算的长度"
            ir["creative"]["C2_beats"].append(clone)

    if minimal:
        # 最小可交付形态：一个角色（有人物多视图 + 音色）、一个场景、无画质预设。
        # 目的不是"让闸门好看"，而是给"闸门不误报"一个真实可达的输入。
        ir["references"] = [
            ir["references"][0],  # 王姐多视图
            ir["references"][2],  # 菜市场场景图
            ir["references"][3],  # 王姐音频
        ]
        ir["locked"]["B2_person"] = [{
            "name": "王姐", "multi_view_ref": "王姐多视图", "lock_strength": "standard",
            "wardrobe": "", "skin": "brief", "blink": "5s",
        }]
        ir["locked"]["B3_blocking"]["eye_line"] = "王姐视线望向画面左侧"
        ir["locked"]["B4_camera"]["rig"] = "机位全程不变，镜头无跳切"
        ir["locked"]["B5_quality"] = {"preset": "", "skin_text": "brief", "grain": ""}
        ir["locked"]["B6_audio"] = [
            {"speaker": "王姐", "audio_ref": "王姐音频", "volume_mode": "normal"},
        ]
        ir["creative"]["C2_beats"] = [
            {
                "shot_id": 1, "framing": "中景", "camera": "平视", "location": "",
                "beat_role": "hook", "duration_sec": 8.0,
                "blocking": "动作1:1复刻原片，不出现任何其他动作",
                "expression": "", "emotion": "惊讶",
                "dialogue": {"speaker": "王姐", "line": "你这气色也太好了吧",
                             "tone": "惊讶", "lip_sync": "", "pause_at": []},
            },
        ]
        ir["forbidden"] = [item for item in ir["forbidden"] if item["id"] == "D0.1"]
    return ir
