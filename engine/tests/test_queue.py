"""反推并发：固定工作池 + FIFO + 位置可见 + 重启不卡死。

用户 2026-10-07：

> 「无队列/限流：每次反推起一个线程，同时传多条视频会一起打满。
>   未来肯定是给团队用的，可能会出现很多人一起上传视频去反推。」

一次反推要调十几次模型，所以这里必须钉住三件事：**封顶、公平、不丢**。
"""

from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from engine.app.inference_queue import InferenceQueue, QueueFull


class _Recorder:
    """记录每个任务何时开始/结束，用来验证并发与顺序。"""

    def __init__(self, *, block: bool = False):
        self.started: list[str] = []
        self.finished: list[str] = []
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()
        self.gate = threading.Event()
        self.block = block

    def __call__(self, project_id: str) -> None:
        with self._lock:
            self.started.append(project_id)
            self._active += 1
            self.peak = max(self.peak, self._active)
        if self.block:
            self.gate.wait(timeout=5)
        with self._lock:
            self.finished.append(project_id)
            self._active -= 1


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestPool:
    def test_concurrency_is_capped(self):
        """**同时在跑的不能超过 workers 个** —— 这是这条改动的全部意义。"""
        recorder = _Recorder(block=True)
        pool = InferenceQueue(recorder, workers=2, max_queued=10)
        pool.start()
        try:
            for index in range(6):
                pool.submit(f"p{index}")
            assert _wait_until(lambda: len(recorder.started) == 2)
            time.sleep(0.1)
            assert len(recorder.started) == 2, "只能有 2 个在跑"
            assert pool.running_count == 2
            assert len(pool.pending_ids) == 4
            recorder.gate.set()
            assert _wait_until(lambda: len(recorder.finished) == 6)
            assert recorder.peak == 2
        finally:
            pool.shutdown()

    def test_fifo_order(self):
        recorder = _Recorder()
        pool = InferenceQueue(recorder, workers=1, max_queued=10)
        pool.start()
        try:
            for index in range(5):
                pool.submit(f"p{index}")
            assert _wait_until(lambda: len(recorder.finished) == 5)
            assert recorder.started == [f"p{i}" for i in range(5)], "必须先来先跑"
        finally:
            pool.shutdown()

    def test_position_is_visible(self):
        """每个人都能看到"我前面还有几条"。"""
        recorder = _Recorder(block=True)
        pool = InferenceQueue(recorder, workers=1, max_queued=10)
        pool.start()
        try:
            pool.submit("first")
            assert _wait_until(lambda: pool.position("first")["state"] == "running")
            pool.submit("second")
            pool.submit("third")
            assert pool.position("second") == {"state": "queued", "position": 1, "ahead": 0}
            assert pool.position("third") == {"state": "queued", "position": 2, "ahead": 1}
            assert pool.position("never") is None
            state = pool.state().to_dict()
            assert state["pending"] == ["second", "third"]
            assert state["running"][0]["project_id"] == "first"
            recorder.gate.set()
        finally:
            pool.shutdown()

    def test_submit_is_idempotent(self):
        recorder = _Recorder(block=True)
        pool = InferenceQueue(recorder, workers=1, max_queued=5)
        pool.start()
        try:
            pool.submit("p1")
            assert _wait_until(lambda: pool.position("p1")["state"] == "running")
            pool.submit("p1")          # 重复提交
            assert len(pool.pending_ids) == 0
            recorder.gate.set()
            assert _wait_until(lambda: len(recorder.finished) == 1)
            assert recorder.started == ["p1"], "同一个项目只能跑一次"
        finally:
            pool.shutdown()

    def test_queue_full_is_refused(self):
        recorder = _Recorder(block=True)
        pool = InferenceQueue(recorder, workers=1, max_queued=2)
        pool.start()
        try:
            pool.submit("a")
            # 等 a 真的**开始跑**再提交后面两条 —— 否则它可能还在待跑队列里，
            # 计数就变成 3 了（这是测试的竞态，不是队列的 bug）
            assert _wait_until(lambda: pool.position("a")["state"] == "running")
            pool.submit("b")
            pool.submit("c")
            with pytest.raises(QueueFull):
                pool.submit("d")
            recorder.gate.set()
        finally:
            pool.shutdown()

    def test_one_failure_does_not_stop_the_pool(self):
        """一个任务炸掉，池子必须继续跑下一个（团队使用时会一直有人提交）。"""
        errors: list[tuple[str, BaseException]] = []

        def runner(project_id: str) -> None:
            if project_id == "bad":
                raise RuntimeError("炸了")
            done.append(project_id)

        done: list[str] = []
        pool = InferenceQueue(
            runner, workers=1, max_queued=5, on_error=lambda pid, exc: errors.append((pid, exc))
        )
        pool.start()
        try:
            pool.submit("bad")
            pool.submit("good")
            assert _wait_until(lambda: done == ["good"])
            assert errors and errors[0][0] == "bad"
            assert pool.running_count == 0, "失败也必须释放槽位"
        finally:
            pool.shutdown()


class TestOrphanRecovery:
    def test_projects_stuck_in_analyzing_are_marked_failed(self):
        """进程一重启，内存里的队列就没了 —— 卡在"分析中"的项目必须如实标失败。

        不处理的话它会永远转圈，团队使用时不断有人来问"怎么不动了"。
        """

        class _Project:
            def __init__(self, pid: str, status: str) -> None:
                self.project_id = pid
                self.status = status
                self.stage = "视觉理解"
                self.error = ""

        projects = [_Project("a", "analyzing"), _Project("b", "ready"),
                    _Project("c", "analyzing")]
        saved: list[str] = []
        recovered = InferenceQueue.recover_orphans(
            projects, save=lambda project: saved.append(project.project_id)
        )
        assert recovered == ["a", "c"]
        assert saved == ["a", "c"]
        assert projects[0].status == "failed"
        assert "重启" in projects[0].error
        assert projects[1].status == "ready", "正常完成的项目不能被动"


class TestApiWiring:
    """走 HTTP 层验证：队列满回 503、状态里能看到"前面还有几条"。"""

    @pytest.fixture()
    def app_client(self):
        from fastapi.testclient import TestClient

        from engine.app.api import create_app
        from engine.app.config import Settings

        workspace = Path(tempfile.mkdtemp(prefix="mod-queue-"))
        settings = replace(
            Settings.load(), workspace_dir=workspace,
            max_concurrent_infer=1, max_queued_infer=1,
        )
        app = create_app(settings)
        # 把真正的流水线换成"卡住不返回"的替身：验证排队行为，不跑模型
        gate = threading.Event()
        started: list[str] = []

        def slow_runner(project_id: str) -> None:
            started.append(project_id)
            gate.wait(timeout=5)

        app.state.queue._runner = slow_runner
        with TestClient(app) as client:
            yield client, app, started, gate
        gate.set()

    def _project(self, client, app, name: str) -> str:
        project = client.post("/api/projects", json={"name": name}).json()
        pid = project["project_id"]
        stored = app.state.store.get(pid)
        stored.video_path = "fake.mp4"      # runner 是替身，不会真读文件
        app.state.store.save(stored)
        return pid

    def test_full_queue_returns_503_and_rolls_back(self, app_client):
        client, app, started, gate = app_client
        first = self._project(client, app, "排队1")
        second = self._project(client, app, "排队2")
        third = self._project(client, app, "排队3")

        assert client.post(f"/api/projects/{first}/infer").status_code == 202
        assert client.post(f"/api/projects/{second}/infer").status_code == 202
        # workers=1 + max_queued=1 → 第三条收不下
        response = client.post(f"/api/projects/{third}/infer")
        assert response.status_code == 503
        assert "队列已满" in response.json()["detail"]
        # **不能留下一个永远"排队中"的项目**
        assert client.get(f"/api/projects/{third}/status").json()["status"] == "idle"

    def test_status_shows_queue_position(self, app_client):
        client, app, started, gate = app_client
        first = self._project(client, app, "第一位")
        second = self._project(client, app, "第二位")
        client.post(f"/api/projects/{first}/infer")
        client.post(f"/api/projects/{second}/infer")

        data = client.get(f"/api/projects/{second}/status").json()
        assert data["queue"]["state"] == "queued"
        assert "前面还有 0 条" in data["stage"], data["stage"]

        health = client.get("/api/health").json()
        assert health["queue"]["workers"] == 1
        assert health["queue"]["pending_count"] == 1


class TestFastModeAndTimings:
    """快速反推（跳过自检）+ 逐阶段计时。

    用户 2026-10-07 报"自检与重写卡了很久"：实测一次自检调用要 **129 秒**，
    所以既要能看见时间花在哪，也要能按次跳过它。
    """

    def _run_with(self, inferencer, *, review_enabled: bool):
        import tempfile

        import engine.app.api as api_module
        from engine.app.api import _run_inference
        from engine.app.config import Settings
        from engine.app.store import ProjectStore

        workspace = Path(tempfile.mkdtemp(prefix="mod-fast-"))
        store = ProjectStore(workspace / "projects")
        project = store.create("快速反推", "fake.mp4")
        project.review_enabled = review_enabled
        store.save(project)

        seen: dict[str, object] = {}

        def build(settings):
            seen["review_rounds"] = settings.review_rounds
            return inferencer

        original = api_module.build_inferencer
        api_module.build_inferencer = build
        try:
            _run_inference(store, Settings.load(), project.project_id)
        finally:
            api_module.build_inferencer = original
        return store.get(project.project_id), seen

    @staticmethod
    def _report():
        from engine.core.contracts import InferenceReport, VideoInfo

        return InferenceReport(
            video=VideoInfo(
                path="f", duration_sec=1.0, width=1, height=1, fps=1.0, has_audio=False
            )
        )

    def test_fast_mode_skips_review_and_says_so(self):
        report = self._report()

        class _Inferencer:
            def infer_to_ir(self, video_path, *, workspace=None, on_progress=None):
                if on_progress:
                    on_progress("视觉理解", 0.3)
                    on_progress("结构归纳", 0.6)
                return {"locked": {}}, report

        saved, seen = self._run_with(_Inferencer(), review_enabled=False)
        assert seen["review_rounds"] == 0, "关掉自检必须真的把轮数设成 0"
        assert saved.status == "ready"
        assert any("快速反推" in text for text in saved.report["degradations"])
        # **逐阶段计时**：至少记到"视觉理解"那一段
        assert saved.report["timings"].get("视觉理解") is not None, saved.report["timings"]

    def test_default_keeps_the_review_on(self):
        class _Inferencer:
            def infer_to_ir(self, video_path, *, workspace=None, on_progress=None):
                return {"locked": {}}, self_report

        self_report = self._report()
        _saved, seen = self._run_with(_Inferencer(), review_enabled=True)
        assert seen["review_rounds"] == 1


class TestZombieAnalyzingState:
    """状态写着"分析中"、队列里却没有它 —— 点"重新反推"不能毫无反应。

    用户 2026-10-08：「右上角**重新反推的按钮是假的**」。
    原因有两半，都在这一轮修掉了：前端在已出结果时不显示进度面板（看着像没反应）、
    后端对"僵尸状态"照着旧状态直接返回（点了也真没反应）。这里是后端那一半。
    """

    def test_stale_analyzing_state_is_resubmitted(self):
        import shutil
        import tempfile
        import threading
        from dataclasses import replace

        from fastapi.testclient import TestClient

        from engine.app.api import create_app
        from engine.app.config import Settings

        workspace = Path(tempfile.mkdtemp(prefix="mod-zombie-"))
        app = create_app(replace(
            Settings.load(), workspace_dir=workspace,
            max_concurrent_infer=1, max_queued_infer=2,
        ))
        started: list[str] = []
        gate = threading.Event()

        def slow_runner(project_id: str) -> None:
            started.append(project_id)
            gate.wait(timeout=5)

        app.state.queue._runner = slow_runner
        try:
            with TestClient(app) as client:
                project = client.post("/api/projects", json={"name": "僵尸"}).json()
                pid = project["project_id"]
                stored = app.state.store.get(pid)
                stored.video_path = "fake.mp4"
                stored.status = "analyzing"      # ← 上次进程被杀留下的状态
                stored.stage = "自检与重写"
                app.state.store.save(stored)

                response = client.post(f"/api/projects/{pid}/infer")
                assert response.status_code == 202, response.text
                # **真的重新入队了**（而不是"已经在分析中"就返回）
                assert app.state.queue.position(pid) is not None, (
                    "僵尸状态必须重新入队，否则按钮点了永远没反应"
                )
                gate.set()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


class TestDeletedWhileRunning:
    """反推进行中被删除：**不能复活，也不能继续烧模型调用**。

    实测踩过：`_run_inference` 收尾写的是 `store.get(id) or project` ——
    项目中途被删除时 `get` 返回 None，于是回退到内存里的旧对象并 `save()`，
    **把已删掉的项目又写了回来**（还标成"完成"）。
    这是并发场景下必然会被踩到的：用户等不及就删了重建。
    """

    def test_deleted_project_is_not_resurrected(self):
        import tempfile
        from dataclasses import replace as _replace

        from engine.app.api import _run_inference
        from engine.app.config import Settings
        from engine.app.store import ProjectStore

        workspace = Path(tempfile.mkdtemp(prefix="mod-delete-"))
        store = ProjectStore(workspace / "projects")
        project = store.create("会被删掉", "fake.mp4")
        pid = project.project_id

        # 流水线跑到一半时，用户把项目删了
        class _Inferencer:
            def infer_to_ir(self, video_path, *, workspace=None, on_progress=None):
                if on_progress:
                    on_progress("视觉理解", 0.6)
                store.delete(pid)          # ← 删除发生在反推过程中
                return {"locked": {}}, type("R", (), {"to_dict": lambda self: {}})()

        import engine.app.api as api_module

        original = api_module.build_inferencer
        api_module.build_inferencer = lambda settings: _Inferencer()
        try:
            _run_inference(store, Settings.load(), pid)
        finally:
            api_module.build_inferencer = original

        assert store.get(pid) is None, "删除的项目不许被写回来"
        import shutil

        shutil.rmtree(workspace, ignore_errors=True)

