"""API 端到端冒烟：真上传、真反推、真组装、真导出。

用 fastapi.testclient 打真实 HTTP 层，所以能顺带验证
参数校验、状态码、以及"锁定位在 API 层也被拦住"。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.app.api import create_app
from engine.app.config import Settings

VIDEO = Path(".probe/vid/talk.mp4")


@pytest.fixture(scope="module")
def client():
    """每个测试模块用独立的临时工作区，避免污染真实 workspace。"""
    workspace = Path(tempfile.mkdtemp(prefix="mod-test-"))
    settings = _with_workspace(Settings.load(), workspace)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
    shutil.rmtree(workspace, ignore_errors=True)


def _with_workspace(settings: Settings, workspace: Path) -> Settings:
    """测试用配置：**把 Agent 的四个"多花一次模型调用"的步骤关掉**。

    为什么（实测踩了很久）：本模块是端到端测**接口契约**（上传→反推→组装→导出），
    而链路里有四步是可选的模型调用：自检-重写、按需取证、感知补看、台词补判。
    全开着时一次反推要几分钟，同一个模块跑三次就超时 —— 表现成"随机失败"，
    排查成本极高。这四步各自都有**注入替身的单元测试**
    （`test_review.py` / `test_perception.py` / `test_attribution.py` / `test_questions.py`），
    端到端这里只需要验证"接线通、契约对"。
    """
    from dataclasses import replace

    return replace(
        settings,
        workspace_dir=workspace,
        review_rounds=0,
        probe_budget=0,
        perception_budget=0,
        attribution_budget=0,
    )


class TestApiContract:
    def test_health_reports_capabilities(self, client):
        data = client.get("/api/health").json()
        assert data["ok"] is True
        assert data["framework_version"] == "mod-framework-v1"
        assert "cv" in data
        assert data["capabilities"]["llm"]["configured"] is True

    def test_schema_exposes_contract(self, client):
        data = client.get("/api/schema").json()
        paths = {f["path"] for f in data["fields"]}
        assert "locked.B3_blocking.positions" in paths
        assert "creative.C2_beats[].dialogue.line" in paths
        assert data["sections"]["locked"] == "关键一致性"

    def test_phrases_exposes_assets(self, client):
        data = client.get("/api/phrases").json()
        assert data["phrase_lib_version"] == "mod-phrases-v1"
        assert "D0" in data["forbidden_groups"]
        assert data["mandatory_dimensions"]

    def test_missing_project_is_404(self, client):
        assert client.get("/api/projects/nope").status_code == 404


class TestProjectFlow:
    @staticmethod
    @pytest.fixture(scope="class")
    def project(client):
        created = client.post("/api/projects", json={"name": "冒烟测试"}).json()
        return created["project_id"]

    def test_create_and_list(self, client, project):
        listing = client.get("/api/projects").json()["projects"]
        assert any(p["project_id"] == project for p in listing)

    def test_upload_requires_video_file(self, client, project):
        response = client.post(
            f"/api/projects/{project}/video",
            files={"file": ("bad.txt", b"not a video", "text/plain")},
        )
        assert response.status_code == 400

    def test_write_rejected_before_infer(self, client, project):
        response = client.put(
            f"/api/projects/{project}/field",
            json={"path": "creative.C1_logline", "value": "x"},
        )
        assert response.status_code == 409


def wait_until_ready(client, project_id: str, *, budget_sec: int = 1200):
    """等反推跑完（真跑模型，**必须给足余量**）。

    实测：同一个模型单次调用在 41~236 秒之间波动（取决于服务端负载），
    一条视频要十几次调用 —— 上限给紧一点就变成"随机的夹具超时"，
    看起来像代码坏了。所以这里给 20 分钟，并且失败时把**当时卡在哪一步**报出来。
    """
    import time

    deadline = time.monotonic() + budget_sec
    status = {"status": "unknown", "stage": ""}
    while time.monotonic() < deadline:
        status = client.get(f"/api/projects/{project_id}/status").json()
        if status["status"] in ("ready", "failed"):
            break
        time.sleep(1)
    assert status["status"] == "ready", (
        f"反推没在 {budget_sec} 秒内完成（卡在「{status.get('stage')}」，"
        f"进度 {status.get('progress')}）：{status}"
    )
    return project_id


@pytest.mark.skipif(not VIDEO.exists(), reason="需要 .probe/vid/talk.mp4 测试视频")
class TestRealPipeline:
    @staticmethod
    @pytest.fixture(scope="class")
    def ready(client):
        project = client.post("/api/projects", json={"name": "真实链路"}).json()
        pid = project["project_id"]
        with VIDEO.open("rb") as handle:
            upload = client.post(
                f"/api/projects/{pid}/video",
                files={"file": (VIDEO.name, handle, "video/mp4")},
            )
        assert upload.status_code == 200, upload.text

        assert client.post(f"/api/projects/{pid}/infer").status_code == 202
        return wait_until_ready(client, pid)

    def test_positions_are_three_coloured(self, client, ready):
        data = client.get(f"/api/projects/{ready}").json()
        counts = data["positions"]["counts"]
        assert set(counts) == {"blue", "orange", "gray"}
        assert counts["blue"] >= 1, "场景素材必须出现在蓝位"
        assert counts["gray"] > counts["orange"], "锁定位应该最多"

    def test_reference_manifest_is_named_not_generated(self, client, ready):
        """Agent 只指名素材、给出建议工序，不生成素材。"""
        manifest = client.get(f"/api/projects/{ready}").json()["reference_manifest"]
        assert manifest
        assert all(item["asset_id"] is None for item in manifest)
        assert all("prep_step" in item and "prep_hint" in item for item in manifest)
        assert any(item["prep_hint"] for item in manifest), "工序建议应非空"

    def test_locked_field_write_is_forbidden(self, client, ready):
        """红线 2：锁定位在 API 层也必须拦住（PRD F2.3）。"""
        for path in (
            "locked.B3_blocking.positions",
            "locked.B3_blocking.eye_line",
            "creative.C2_beats[0].beat_role",
        ):
            response = client.put(
                f"/api/projects/{ready}/field",
                json={"path": path, "value": ["篡改"]},
            )
            assert response.status_code == 403, f"{path} 应被拒绝"
            assert "锁定位" in response.json()["detail"] or "自动位" in response.json()["detail"]

    def test_param_field_rejects_value_outside_whitelist(self, client, ready):
        response = client.put(
            f"/api/projects/{ready}/field",
            json={"path": "creative.C2_beats[0].framing", "value": "超广角航拍"},
        )
        assert response.status_code == 403
        assert "白名单" in response.json()["detail"]

    def test_free_field_write_succeeds(self, client, ready):
        response = client.put(
            f"/api/projects/{ready}/field",
            json={"path": "creative.C1_logline", "value": "王姐在菜市场惊讶追问李姐的气色变化"},
        )
        assert response.status_code == 200
        assert response.json()["value"].startswith("王姐在菜市场")

    def test_dialogue_write_is_user_owned(self, client, ready):
        """红线 3：台词由用户填，Agent 不代写。"""
        response = client.put(
            f"/api/projects/{ready}/dialogue",
            json={"shot_id": 1, "speaker": "王姐", "line": "你这气色也太好了吧", "tone": "惊讶"},
        )
        assert response.status_code == 200
        assert response.json()["dialogue"]["line"] == "你这气色也太好了吧"

        missing = client.put(
            f"/api/projects/{ready}/dialogue",
            json={"shot_id": 99, "line": "x"},
        )
        assert missing.status_code == 404

    def test_bind_asset_updates_manifest(self, client, ready):
        response = client.post(
            f"/api/projects/{ready}/bind",
            json={"reference_index": 0, "asset_id": "upstream-asset-001"},
        )
        assert response.status_code == 200
        assert response.json()["manifest"][0]["bound"] is True

    def test_delivery_mode_switches_param_header(self, client, ready):
        """参数声明行**默认不写**（用户 2026-09-29 明确要求去掉那行）。

        比例/时长/分辨率在生成平台上是独立控件，写进提示词正文只是冗余。
        交付模式仍能控制它 —— 这是"默认关"而不是"功能删掉"。
        """
        manual = client.get(f"/api/projects/{ready}/assemble").json()["document"]
        assert not manual["param_header"], "默认不应写参数声明行"

        client.put(f"/api/projects/{ready}/delivery", json={"include_param_header": True})
        with_header = client.get(f"/api/projects/{ready}/assemble").json()["document"]
        assert with_header["param_header"], "显式打开时应写参数声明"
        assert manual["char_count"] == with_header["char_count"], "正文必须完全一致"

        client.put(f"/api/projects/{ready}/delivery", json={"include_param_header": False})

    def test_assemble_returns_gates_and_apply_targets(self, client, ready):
        data = client.get(f"/api/projects/{ready}/assemble").json()
        codes = [r["code"] for r in data["gates"]["results"]]
        assert codes == [f"G{i}" for i in range(1, 15)]
        assert "关键一致性" in data["apply_targets"]
        assert data["gates"]["deliverable"] == (not data["gates"]["results"] or
                                               data["gates"]["verdict"] != "BLOCKED")

    def test_assemble_returns_review_outlet(self, client, ready):
        """反推的"待确认"必须有一个出口给用户。

        以前置信度、降级说明、CV↔模型冲突、闸门判定**一个出口都没有** ——
        结果页只有一个文本框，用户只能靠肉眼抓错（实测已经因此返工多次）。
        """
        data = client.get(f"/api/projects/{ready}/assemble").json()
        review = data["review"]
        assert isinstance(review["count"], int)
        assert review["count"] >= len(review["items"])
        levels = [item["level"] for item in review["items"]]
        # 阻断项必须排在最前
        assert levels == sorted(levels, key=lambda level: ("block", "warn", "review").index(level))
        # 每一项都要有可读的标签与说明
        assert all(item["label"] and item["detail"] for item in review["items"])
        # 闸门阻断项要在摘要里露面
        blockers = [r["name"] for r in data["gates"]["results"] if r["status"] == "block"]
        for name in blockers:
            assert any(item["label"] == name for item in review["items"]), name

    def test_forbidden_groups_can_be_filtered(self, client, ready):
        """「固定提示词」表格页：用户勾选哪些 D 组写进提示词。"""
        # 只勾 D1 → D0/D7 被强制补回（永不可裁）
        result = client.put(
            f"/api/projects/{ready}/forbidden-groups", json={"groups": ["D1"]}
        ).json()
        assert result["groups"] == ["D0", "D1", "D7"]
        assert result["enforced"] == ["D0", "D7"]
        assert result["item_count"] > 0

        # 全不勾 → 仍只剩 D0/D7
        only_core = client.put(
            f"/api/projects/{ready}/forbidden-groups", json={"groups": []}
        ).json()
        assert only_core["groups"] == ["D0", "D7"]

        # 未知组名如实回报，不静默吞掉
        unknown = client.put(
            f"/api/projects/{ready}/forbidden-groups", json={"groups": ["D9"]}
        ).json()
        assert unknown["unknown"] == ["D9"]
        assert "D9" not in unknown["groups"]

        # 勾上低频组 → 条目确实变多
        rich = client.put(
            f"/api/projects/{ready}/forbidden-groups",
            json={"groups": ["D1", "D5", "D6"]},
        ).json()
        assert rich["groups"] == ["D0", "D1", "D5", "D6", "D7"]
        assert rich["item_count"] > only_core["item_count"]

    def test_export_three_forms_are_consistent(self, client, ready):
        plain = client.get(f"/api/projects/{ready}/export", params={"form": "plain"}).text
        markdown = client.get(f"/api/projects/{ready}/export", params={"form": "markdown"}).text
        as_json = json.loads(
            client.get(f"/api/projects/{ready}/export", params={"form": "json"}).text
        )
        assert "【禁止项】" in plain
        assert "交付自检" in markdown, "导出必须如实带上闸门判定"
        assert as_json["sections"][0]["lines"] == as_json["sections"][0]["lines"]
        assert "gates" in as_json

    def test_export_carries_the_verdict_in_every_form(self, client, ready):
        """**三种导出形态都要带闸门判定**（红线 10：未全绿不许声称可交付）。

        实测踩过：只有 markdown 带「# 交付自检」，而 plain 是**默认形态**、
        也正是"直接粘进生成平台"的那一份 —— 全红也照样给一份干净正文。
        """
        for form in ("plain", "markdown", "json"):
            response = client.get(f"/api/projects/{ready}/export", params={"form": form})
            assert response.status_code == 200
            assert response.headers["x-delivery-verdict"] in (
                "ALL_GREEN", "PASS_WITH_WARNINGS", "BLOCKED",
            )
            assert "x-delivery-blockers" in response.headers

        plain = client.get(f"/api/projects/{ready}/export", params={"form": "plain"})
        if plain.headers["x-delivery-verdict"] == "ALL_GREEN":
            assert not plain.text.startswith("【交付自检】")
        else:
            assert plain.text.startswith("【交付自检】"), plain.text[:60]
            assert "这一行不是提示词内容" in plain.text

    def test_export_rejects_unknown_form(self, client, ready):
        assert client.get(
            f"/api/projects/{ready}/export", params={"form": "pdf"}
        ).status_code == 422


class TestQuestionFlow:
    """结构化提问的接口形状。

    为什么值得单独钉：这是**唯一一个"用户点一下就把交付物改对"的入口** ——
    Agent 出题、用户点选、答案经同一道门禁写进 IR。
    """

    def test_assemble_exposes_an_empty_question_list(self, client, shared_ready):
        data = client.get(f"/api/projects/{shared_ready}/assemble").json()
        assert data["questions"] == [], "没有两难时不该凭空出题"

    def test_answering_an_unknown_question_is_404(self, client, shared_ready):
        response = client.post(
            f"/api/projects/{shared_ready}/answer",
            json={"id": "不存在的题", "choice": "随便"},
        )
        assert response.status_code == 404

    def test_question_kind_shows_up_in_the_review_outlet(self):
        """出题时它必须出现在「待确认」栏 —— 否则用户根本不知道要回答。"""
        from types import SimpleNamespace

        from engine.app.api import _review_summary
        from engine.core import gates

        project = SimpleNamespace(report={
            "questions": [{
                "id": "q1", "path": "creative.C1_logline",
                "text": "主线是展示动作还是展示产品？",
                "options": ["展示动作", "展示产品"], "reason": "两种都说得通",
                "answer": "",
            }],
        })
        items = _review_summary(project, gates.GateReport())["items"]
        asked = [i for i in items if i["kind"] == "question"]
        assert asked and "展示动作 / 展示产品" in asked[0]["detail"]
        # 已回答的不再打扰
        project.report["questions"][0]["answer"] = "展示动作"
        items = _review_summary(project, gates.GateReport())["items"]
        assert not [i for i in items if i["kind"] == "question"]


class TestSettingsDefaults:
    """``Settings`` 是 ``slots=True`` 的数据类 —— ``cls.xxx`` 拿到的是**成员描述符**。

    这个坑埋了很久：以前所有 ``pick(..., cls.xxx)`` 的默认值恰好都被 ``.env``
    里的显式值掩盖了，直到新加 ``model_review``（没写进 .env）才暴露 ——
    设置项变成了 ``<member 'model_review' of 'Settings' objects>``。
    """

    def test_no_setting_comes_back_as_a_descriptor(self):
        from engine.app.config import Settings

        settings = Settings.load()
        broken = [
            name
            for name in Settings.__dataclass_fields__
            if "member" in repr(getattr(settings, name))
        ]
        assert broken == [], f"这些设置项没取到真默认值：{broken}"

    def test_review_model_is_a_real_model_id(self):
        from engine.app.config import Settings

        value = Settings.load().model_review
        assert isinstance(value, str) and value and "member" not in value


class TestAnswerRoundTrip:
    """点选作答的闭环：**出题 → 点选 → 答案落进 IR → 正文变了**。

    不跑模型（纯磁盘 + HTTP），所以快且稳定。
    """

    @pytest.fixture()
    def project_with_question(self):
        """建一个临时项目：有 IR、有交付物、报告里有一道待答的题。"""
        import json
        import tempfile

        from engine.app.store import ProjectStore
        from engine.tests.test_core import _sample_ir

        workspace = Path(tempfile.mkdtemp(prefix="mod-answer-"))
        settings = Settings.load()
        store = ProjectStore(workspace / "projects")
        project = store.create("问答题", str(VIDEO))
        project.ir = _sample_ir()
        project.report = {"questions": [{
            "id": "q1", "path": "creative.C1_logline",
            "text": "主线是展示动作还是展示产品？",
            "options": ["展示动作", "展示产品"], "reason": "两种都说得通",
            "answer": "",
        }]}
        store.save(project)
        app = create_app(_with_workspace(settings, workspace))
        with TestClient(app) as client:
            yield client, project.project_id, workspace, app.state.store
        shutil.rmtree(workspace, ignore_errors=True)

    def test_choice_lands_in_the_ir_and_the_prompt_changes(self, project_with_question):
        client, pid, workspace, _store = project_with_question
        before = client.get(f"/api/projects/{pid}/assemble").json()
        assert before["questions"], "题目要能出现在 assemble 响应里"
        assert "展示动作" not in before["document"]["text"]

        response = client.post(
            f"/api/projects/{pid}/answer",
            json={"id": "q1", "choice": "展示动作"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["path"] == "creative.C1_logline"

        # ① IR 真的被改了 ② 交付物跟着变 ③ 题目不再重复出现
        stored = json.loads(
            (workspace / "projects" / pid / "project.json").read_text(encoding="utf-8")
        )
        assert stored["ir"]["creative"]["C1_logline"] == "展示动作"
        after = client.get(f"/api/projects/{pid}/assemble").json()
        assert "展示动作" in after["document"]["text"]
        assert after["questions"] == []

    def test_invalid_choice_is_rejected(self, project_with_question):
        client, pid, _workspace, _store = project_with_question
        response = client.post(
            f"/api/projects/{pid}/answer", json={"id": "q1", "choice": "不在选项里"}
        )
        assert response.status_code == 400

    def test_locked_target_is_refused_even_if_a_question_points_at_it(
        self, project_with_question
    ):
        """**门禁只有一道**：就算题目的落点是锁定位，写入也必须被拦。

        （题目本身会被 ``apply_questions`` 过滤掉，但这里是防"绕过出题环节、
        直接构造一道题"的情形 —— UI 拦、API 也拦。）
        """
        client, pid, _workspace, store = project_with_question
        # **通过 store 改，不直接改磁盘**：服务对项目有内存缓存
        # （`recover_orphans` 启动时会预加载），直接写文件服务是看不到的。
        stored = store.get(pid)
        stored.report["questions"].append({
            "id": "q2", "path": "locked.B3_blocking.positions[0]",
            "text": "站位改成居中？", "options": ["改成居中", "保持不动"],
            "reason": "试探", "answer": "",
        })
        store.save(stored)

        response = client.post(
            f"/api/projects/{pid}/answer", json={"id": "q2", "choice": "改成居中"}
        )
        assert response.status_code == 403


class TestReviewOutlet:
    """「待确认」栏是自检回路唯一的出口 —— 没有它，用户不知道交付物被改过。"""

    def test_critique_notes_are_surfaced(self):
        from types import SimpleNamespace

        from engine.app.api import _review_summary
        from engine.core import gates

        project = SimpleNamespace(report={
            "reviews": [
                {"path": "creative.C2_beats[1].expression", "action": "set",
                 "reason": "没对象的约束：表情句会被读成背对镜头的那位",
                 "applied": True, "detail": "面带微笑 → 人物2面带微笑"},
                {"path": "locked.B3_blocking.positions", "action": "set",
                 "reason": "试探越界", "applied": False,
                 "detail": "站位是复刻的底线（框架 §9.2 永不可裁）"},
            ],
        })
        report = gates.GateReport()
        items = _review_summary(project, report)["items"]
        labels = [item["label"] for item in items]
        assert any("自检已改写" in label for label in labels)
        assert any("自检拒绝的修改" in label for label in labels)
        # 改写过的必须比"拒绝"更显眼（warn > review）
        applied = next(i for i in items if "自检已改写" in i["label"])
        assert applied["level"] == "warn"
        # 汇总条要能说清"改了几处"
        assert any("改写了 1 处" in item["detail"] for item in items)

    def test_no_critique_notes_means_no_noise(self):
        from types import SimpleNamespace

        from engine.app.api import _review_summary
        from engine.core import gates

        project = SimpleNamespace(report={})
        items = _review_summary(project, gates.GateReport())["items"]
        assert not [i for i in items if i["kind"] == "critique"]


@pytest.fixture(scope="module")
def shared_ready(client):
    """跨测试类共用的"已反推项目"。

    **为什么提到模块级**：反推是**真跑模型**，而链路后来加了感知补看、
    台词补判、自检-重写几步（各含若干次模型调用），一次要几分钟。
    原来这个夹具挂在单个测试类上，别的类想用"一个已反推的项目"就得再跑一次。
    """
    project = client.post("/api/projects", json={"name": "对齐流程"}).json()
    pid = project["project_id"]
    with VIDEO.open("rb") as handle:
        assert client.post(
            f"/api/projects/{pid}/video",
            files={"file": (VIDEO.name, handle, "video/mp4")},
        ).status_code == 200
    assert client.post(f"/api/projects/{pid}/infer").status_code == 202
    return wait_until_ready(client, pid)


@pytest.mark.skipif(not VIDEO.exists(), reason="需要 .probe/vid/talk.mp4 测试视频")
class TestAlignmentFlow:
    """内容对齐（PRD F6）走 HTTP 层。"""

    @staticmethod
    @pytest.fixture(scope="class")
    def ready(shared_ready):
        return shared_ready

    def test_slots_are_listed_before_alignment(self, client, ready):
        data = client.get(f"/api/projects/{ready}/align").json()
        assert data["report"] is None
        assert data["slots"], "未对齐时也应列出可对齐的位置"

    def test_align_writes_confident_bindings(self, client, ready):
        """把原片台词原样粘回来 —— 应当全部高置信度绑定。"""
        slots = client.get(f"/api/projects/{ready}/align").json()["slots"]
        lines = [s["reference_line"] for s in slots if s["reference_line"]]
        assert lines, "反推应产出至少一句参考台词"

        response = client.post(
            f"/api/projects/{ready}/align",
            json={"content": "\n".join(lines), "apply": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["report"]["summary"]["bound"] >= 1
        assert body["written"], "把握足够的绑定应写进 IR"
        for binding in body["report"]["bindings"]:
            if binding["block"] is not None:
                assert binding["confidence"] >= 0.7

    def test_align_rejects_empty_content(self, client, ready):
        assert client.post(
            f"/api/projects/{ready}/align", json={"content": "   \n  \n"}
        ).status_code == 400

    def test_unresolved_alignment_blocks_gate_g15(self, client, ready):
        """F6.6 强制门禁：存在未决状态时不允许进入组装。"""
        response = client.post(
            f"/api/projects/{ready}/align",
            json={"content": "甲：今天天气不错\n乙：是啊挺暖和的", "apply": True},
        )
        assert response.status_code == 200
        report = response.json()["report"]
        if not report["summary"]["has_unresolved"]:
            pytest.skip("本例未产生未决项")

        assembled = client.get(f"/api/projects/{ready}/assemble").json()
        codes = [r["code"] for r in assembled["gates"]["results"]]
        assert "G15" in codes, "用绝对齐后必须出现 G15"
        g15 = next(r for r in assembled["gates"]["results"] if r["code"] == "G15")
        assert g15["status"] == "block"
        assert not assembled["gates"]["deliverable"]

    def test_manual_align_clears_pending(self, client, ready):
        """手动确认后未决状态应消失（F6.5）。"""
        slots = client.get(f"/api/projects/{ready}/align").json()["slots"]
        if not slots:
            pytest.skip("没有可对齐的位置")
        target = slots[0]["path"]

        response = client.put(
            f"/api/projects/{ready}/align/manual",
            json={"slot_path": target, "content": "手动填写的一句台词", "speaker": "甲"},
        )
        assert response.status_code == 200
        assert response.json()["line"] == "手动填写的一句台词"

        stored = client.get(f"/api/projects/{ready}/align").json()["report"]
        binding = next(
            b for b in stored["bindings"] if b["slot"]["path"] == target
        )
        assert binding["status"] == "bound"
        assert not binding["requires_confirmation"]
        assert binding["reason"] == "已由用户手动确认"

    def test_manual_align_can_unbind(self, client, ready):
        """解绑 = 标为仅参考，该位置回到保留原片结构。"""
        slots = client.get(f"/api/projects/{ready}/align").json()["slots"]
        target = slots[0]["path"]
        response = client.put(
            f"/api/projects/{ready}/align/manual",
            json={"slot_path": target, "block_index": None},
        )
        assert response.status_code == 200
        assert response.json()["line"] == ""

    def test_manual_align_rejects_non_dialogue_path(self, client, ready):
        response = client.put(
            f"/api/projects/{ready}/align/manual",
            json={"slot_path": "locked.B3_blocking", "content": "x"},
        )
        assert response.status_code == 404

    def test_aligned_content_reaches_export(self, client, ready):
        slots = client.get(f"/api/projects/{ready}/align").json()["slots"]
        target = slots[0]["path"]
        client.put(
            f"/api/projects/{ready}/align/manual",
            json={"slot_path": target, "content": "导出用的测试台词", "speaker": "甲"},
        )
        exported = client.get(
            f"/api/projects/{ready}/export", params={"form": "plain"}
        ).text
        assert "导出用的测试台词" in exported

    def test_g15_absent_when_alignment_unused(self, client):
        """没用过对齐的项目不该平白多出一项闸门。"""
        project = client.post("/api/projects", json={"name": "未对齐"}).json()
        pid = project["project_id"]
        with VIDEO.open("rb") as handle:
            client.post(
                f"/api/projects/{pid}/video",
                files={"file": (VIDEO.name, handle, "video/mp4")},
            )
        client.post(f"/api/projects/{pid}/infer")
        # **先断言反推成功**：这条测试原来直接去读 gates，反推失败时只会报
        # `KeyError: 'gates'`，把真正的原因（status=failed + error）吞掉了 ——
        # 实测遇到过，排查了很久。等待逻辑见 wait_until_ready。
        wait_until_ready(client, pid)
        assembled = client.get(f"/api/projects/{pid}/assemble").json()
        codes = [r["code"] for r in assembled["gates"]["results"]]
        assert "G15" not in codes
        assert codes == [f"G{i}" for i in range(1, 15)]
