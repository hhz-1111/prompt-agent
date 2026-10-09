"""反推任务队列：**固定工作池 + 先来先跑 + 位置可见 + 重启不卡死**。

## 为什么必须要有（用户 2026-10-07）

> 「无队列/限流：每次反推起一个线程，同时传多条视频会一起打满。
>   **未来肯定是给团队用的**，可能会出现很多人一起上传视频去反推。」

一次反推要调**十几次模型**（逐镜头视觉理解 → 语义归纳 → 台词归属 → 感知补看
→ 自检-重写 → 按需取证 → 多源裁决），所以：

1. **必须封顶**：同时在跑的反推数要有上限，否则几个人的任务一起跑就直接撞
   模型接口的限流，谁都跑不完；
2. **必须公平**：先来先跑（FIFO），且每个人都能看到"我前面还有几条"；
3. **必须不丢**：任务异常要释放槽位并把项目标成失败；
   **服务重启后卡在"分析中"的项目要被识别出来** —— 否则它会永远转圈，
   团队使用时会不停地有人来问"为什么一直不动"。

这个模块**不依赖 FastAPI、不依赖具体流水线**：runner 是注入进来的，
所以可以用一个假的 runner 把并发、顺序、异常路径全部测掉。
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


class QueueFull(Exception):
    """排队的人太多了 —— 调用方应当回 503，而不是无限收任务。"""


@dataclass
class QueueState:
    """一次快照，给 ``/status``、``/health`` 和运维看。"""

    running: list[dict] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    workers: int = 0
    max_queued: int = 0

    def to_dict(self) -> dict:
        return {
            "workers": self.workers,
            "running": list(self.running),
            "pending": list(self.pending),
            "pending_count": len(self.pending),
            "max_queued": self.max_queued,
        }


class InferenceQueue:
    """反推任务池。

    ``runner(project_id)`` 由调用方注入（生产里是跑流水线的那段代码）——
    它**必须自己把异常处理干净**；万一抛出来，队列也会接住并继续下一个任务
    （一个任务炸掉不能让整个池子停摆）。
    """

    def __init__(
        self,
        runner: Callable[[str], None],
        *,
        workers: int = 2,
        max_queued: int = 20,
        on_error: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        self._runner = runner
        self._on_error = on_error
        self.workers = max(1, int(workers))
        self.max_queued = max(1, int(max_queued))
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._pending: dict[str, float] = {}      # project_id → 入队时刻
        self._running: dict[str, float] = {}      # project_id → 开始时刻
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._closed = False

    # ---- 生命周期 ----
    def start(self) -> None:
        """启动工作线程（幂等）。``create_app`` 时调一次即可。"""
        with self._lock:
            if self._threads or self._closed:
                return
            for index in range(self.workers):
                thread = threading.Thread(
                    target=self._work, name=f"infer-worker-{index + 1}", daemon=True
                )
                thread.start()
                self._threads.append(thread)

    def shutdown(self) -> None:
        """停止收任务并唤醒工作线程（用于测试收尾）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for _ in self._threads:
                self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=1.0)

    # ---- 提交与查询 ----
    def submit(self, project_id: str) -> dict:
        """入队。同一个项目重复提交是**幂等**的（返回它当前的状态）。"""
        with self._lock:
            existing = self._position_locked(project_id)
            if existing is not None:
                return existing
            if len(self._pending) >= self.max_queued:
                raise QueueFull(
                    f"反推队列已满（{self.max_queued} 条在等）。请稍后再提交，"
                    f"或等服务把前面的任务跑完。"
                )
            self._pending[project_id] = time.time()
            self._queue.put(project_id)
            return self._position_locked(project_id) or {}

    def position(self, project_id: str) -> dict | None:
        with self._lock:
            return self._position_locked(project_id)

    def state(self) -> QueueState:
        now = time.time()
        with self._lock:
            return QueueState(
                running=[
                    {"project_id": pid, "elapsed_sec": round(now - started, 1)}
                    for pid, started in self._running.items()
                ],
                pending=list(self._pending),
                workers=self.workers,
                max_queued=self.max_queued,
            )

    def _position_locked(self, project_id: str) -> dict | None:
        if project_id in self._running:
            return {"state": "running", "position": 0, "ahead": 0}
        if project_id in self._pending:
            ahead = list(self._pending).index(project_id)
            return {"state": "queued", "position": ahead + 1, "ahead": ahead}
        return None

    # ---- 工作线程 ----
    def _work(self) -> None:
        while True:
            project_id = self._queue.get()
            try:
                if project_id is None:          # 关机信号
                    return
                with self._lock:
                    self._pending.pop(project_id, None)
                    self._running[project_id] = time.time()
                try:
                    self._runner(project_id)
                except BaseException as exc:      # noqa: BLE001 - 一个任务炸掉不能停池子
                    if self._on_error is not None:
                        self._on_error(project_id, exc)
                finally:
                    with self._lock:
                        self._running.pop(project_id, None)
            finally:
                self._queue.task_done()

    # ---- 重启恢复 ----
    @staticmethod
    def recover_orphans(
        projects: Iterable,
        *,
        save: Callable[[object], None],
        reason: str = "服务重启中断（排队或运行中），请重新反推",
    ) -> list[str]:
        """把**卡在"分析中"**的项目标成失败。

        进程一重启，内存里的队列就没了 —— 那些项目会永远停在"分析中"，
        前端一直转圈。团队使用时会不断有人来问"怎么不动了"，
        所以启动时必须扫一遍并如实标失败（**不假装还在跑**）。
        """
        recovered: list[str] = []
        for project in projects:
            if str(getattr(project, "status", "")) != "analyzing":
                continue
            project.status = "failed"
            project.stage = "已中断"
            project.error = reason
            save(project)
            recovered.append(str(getattr(project, "project_id", "")))
        return recovered

    @property
    def running_count(self) -> int:
        with self._lock:
            return len(self._running)

    @property
    def pending_ids(self) -> Sequence[str]:
        with self._lock:
            return tuple(self._pending)
