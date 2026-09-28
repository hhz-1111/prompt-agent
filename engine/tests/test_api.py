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
    from dataclasses import replace

    return replace(settings, workspace_dir=workspace)


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

        import time

        for _ in range(120):
            status = client.get(f"/api/projects/{pid}/status").json()
            if status["status"] in ("ready", "failed"):
                break
            time.sleep(1)
        assert status["status"] == "ready", status
        return pid

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
        """PRD §6.4.5：手动模式保留参数声明，直调模式移除。"""
        manual = client.get(f"/api/projects/{ready}/assemble").json()["document"]
        assert manual["param_header"], "Phase 1 应保留参数声明"

        client.put(f"/api/projects/{ready}/delivery", json={"mode": "phase2_api"})
        api_mode = client.get(f"/api/projects/{ready}/assemble").json()["document"]
        assert not api_mode["param_header"], "Phase 2 应移除参数声明"
        assert manual["char_count"] == api_mode["char_count"], "正文应完全一致"

        client.put(f"/api/projects/{ready}/delivery", json={"mode": "phase1_manual"})

    def test_assemble_returns_gates_and_apply_targets(self, client, ready):
        data = client.get(f"/api/projects/{ready}/assemble").json()
        codes = [r["code"] for r in data["gates"]["results"]]
        assert codes == [f"G{i}" for i in range(1, 15)]
        assert "关键一致性" in data["apply_targets"]
        assert data["gates"]["deliverable"] == (not data["gates"]["results"] or
                                               data["gates"]["verdict"] != "BLOCKED")

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

    def test_export_rejects_unknown_form(self, client, ready):
        assert client.get(
            f"/api/projects/{ready}/export", params={"form": "pdf"}
        ).status_code == 422
