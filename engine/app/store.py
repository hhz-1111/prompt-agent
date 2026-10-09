"""会话存储：项目状态的持久化。

一个"项目"就是一次复刻任务：视频 → 反推 IR → 用户改 → 组装交付。

存储形态刻意选**文件 + 内存缓存**而不是数据库：
这是团队内部工具、单机运行、项目量级在千级以内。
文件存储的好处是可直接查看、可手工备份、可进版本库做回归样本；
引入 SQLite 只会多一层映射而换不来实际收益。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from engine.core.types import FRAMEWORK_VERSION


@dataclass(slots=True)
class Project:
    """一个复刻项目的全部状态。"""

    project_id: str
    name: str
    video_path: str
    created_at: float
    updated_at: float
    framework_version: str = FRAMEWORK_VERSION
    ir: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    """created / analyzing / ready / failed"""
    stage: str = ""
    progress: float = 0.0
    error: str = ""
    review_enabled: bool = True
    """这次反推是否跑"自检-重写"（用户 2026-10-07：它一次调用就要 129 秒）。

    按**项目**记住选择：关掉=快速反推（少几次模型调用），
    代价是交付前没人回头看一遍，如实记在交付物里。
    """

    revisions: list[dict[str, Any]] = field(default_factory=list)
    """用户意见改写的记账（PRD F8.4）。**每一轮一条**，落盘、可展示、可追溯。

    为什么要落盘而不是只放在内存：改完之后用户要能回头看
    "我当时说了什么、它动了哪几处、哪条它没做到" —— 第七轮踩过一次
    （``InferenceReport.to_dict`` 漏字段导致记账静默消失），
    所以这里的序列化**成对写**并有回归测试守着。
    """

    ir_history: list[dict[str, Any]] = field(default_factory=list)
    """改写前的 IR 快照，**只为一键撤销**用（最近若干轮，由配置封顶）。

    存整份 IR 而不是"反向 patch"：IR 只有几 KB，直接存最省心，
    撤销就是"把上一份放回去"，没有反向应用的出错空间。
    """

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "video_path": self.video_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "framework_version": self.framework_version,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "has_ir": bool(self.ir),
            "review_enabled": self.review_enabled,
        }


class ProjectStore:
    """项目的读写门面。线程内使用，够本工具的量级。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Project] = {}

    # ---- 路径 ----
    def _dir(self, project_id: str) -> Path:
        return self.root / project_id

    def _file(self, project_id: str) -> Path:
        return self._dir(project_id) / "project.json"

    def media_dir(self, project_id: str) -> Path:
        target = self._dir(project_id) / "media"
        target.mkdir(parents=True, exist_ok=True)
        return target

    # ---- 写 ----
    def create(self, name: str, video_path: str) -> Project:
        now = time.time()
        project = Project(
            project_id=uuid.uuid4().hex[:12],
            name=name or "未命名项目",
            video_path=video_path,
            created_at=now,
            updated_at=now,
        )
        self.save(project)
        return project

    def save(self, project: Project) -> None:
        project.touch()
        self._cache[project.project_id] = project
        target = self._file(project.project_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self._serialize(project), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_status(
        self,
        project_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        error: str | None = None,
    ) -> Project | None:
        project = self.get(project_id)
        if project is None:
            return None
        if status is not None:
            project.status = status
        if stage is not None:
            project.stage = stage
        if progress is not None:
            project.progress = progress
        if error is not None:
            project.error = error
        self.save(project)
        return project

    # ---- 读 ----
    def get(self, project_id: str) -> Project | None:
        if project_id in self._cache:
            return self._cache[project_id]
        target = self._file(project_id)
        if not target.exists():
            return None
        raw = json.loads(target.read_text(encoding="utf-8"))
        project = self._deserialize(raw)
        self._cache[project_id] = project
        return project

    def list(self) -> list[Project]:
        projects: list[Project] = []
        for directory in sorted(self.root.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            project = self.get(directory.name)
            if project is not None:
                projects.append(project)
        projects.sort(key=lambda p: p.created_at, reverse=True)
        return projects

    def delete(self, project_id: str) -> bool:
        self._cache.pop(project_id, None)
        directory = self._dir(project_id)
        if not directory.exists():
            return False
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        directory.rmdir()
        return True

    # ---- 序列化 ----
    @staticmethod
    def _serialize(project: Project) -> dict[str, Any]:
        return {
            "project_id": project.project_id,
            "name": project.name,
            "video_path": project.video_path,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "framework_version": project.framework_version,
            "status": project.status,
            "stage": project.stage,
            "progress": project.progress,
            "error": project.error,
            "ir": project.ir,
            "report": project.report,
            "revisions": project.revisions,
            "ir_history": project.ir_history,
        }

    @staticmethod
    def _deserialize(raw: Mapping[str, Any]) -> Project:
        return Project(
            project_id=str(raw["project_id"]),
            name=str(raw.get("name", "")),
            video_path=str(raw.get("video_path", "")),
            created_at=float(raw.get("created_at", 0)),
            updated_at=float(raw.get("updated_at", 0)),
            framework_version=str(raw.get("framework_version", FRAMEWORK_VERSION)),
            ir=dict(raw.get("ir") or {}),
            report=dict(raw.get("report") or {}),
            status=str(raw.get("status", "created")),
            stage=str(raw.get("stage", "")),
            progress=float(raw.get("progress", 0)),
            error=str(raw.get("error", "")),
            review_enabled=bool(raw.get("review_enabled", True)),
            revisions=[dict(item) for item in (raw.get("revisions") or [])
                       if isinstance(item, Mapping)],
            ir_history=[dict(item) for item in (raw.get("ir_history") or [])
                        if isinstance(item, Mapping)],
        )
