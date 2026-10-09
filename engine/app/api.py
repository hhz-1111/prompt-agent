"""HTTP 接口层。

设计纪律：

1. **门禁只在核心里判**：API 不重复实现"哪个字段能改"，
   一律调 ``framework.can_write`` —— 否则 UI 拦住了、API 没拦，
   锁定位就形同虚设（PRD F2.3 要求 UI 与 API 两层都不可写）。
2. **反推是长任务**：同步返回会让请求超时、也无法反馈进度。
   所以提交后立即返回，前端轮询状态。
3. **接口薄**：只做参数解析与调用用例，业务逻辑全在 core / pipeline。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from engine.core import (
    alignment,
    assembler,
    framework as fw,
    gates,
    phrases as ph,
    positions,
)
from engine.core.types import OutputForm

from .config import Settings
from .inference_queue import InferenceQueue, QueueFull
from .pipeline.infer import build_inferencer
from .store import Project, ProjectStore

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # PRD F1.1：≤500MB


# ----------------------------------------------------------------------
# 请求体
# ----------------------------------------------------------------------
class CreateProjectRequest(BaseModel):
    name: str = Field(default="", description="项目名")


class WriteRequest(BaseModel):
    path: str = Field(description="IR 位置路径，如 locked.B2_person[0].name")
    value: Any = Field(default=None, description="新值")


class AnswerRequest(BaseModel):
    """回答 Agent 提出的选择题。"""

    id: str = Field(description="题目 id（assemble 响应 questions[].id）")
    choice: str = Field(description="选项原文，必须与 options 里的一项完全一致")


class BindAssetRequest(BaseModel):
    reference_index: int = Field(ge=0)
    asset_id: str = Field(description="上游回填的素材句柄")


class AlignmentRequest(BaseModel):
    content: str = Field(description="用户粘贴的台词稿，支持纯脚本 / 说话人：内容 / 00:03 内容")
    apply: bool = Field(default=True, description="是否把把握足够的绑定写进提示词")
    chars_per_sec: float = Field(default=5.0, gt=0, description="语速估算，中文常规 5 字/秒")


class ManualAlignRequest(BaseModel):
    slot_path: str = Field(description="要调整的位置路径")
    block_index: int | None = Field(default=None, description="绑定的内容块下标；null 表示解绑")
    content: str | None = Field(default=None, description="直接给内容（忽略 block_index）")
    speaker: str | None = None


class DeliveryRequest(BaseModel):
    mode: str | None = None
    strategy: str | None = None
    target_model: str | None = None
    first_frame_mode: str | None = None
    include_param_header: bool | None = None


class DialogueWriteRequest(BaseModel):
    shot_id: int = Field(ge=1)
    speaker: str = ""
    line: str = ""
    tone: str = ""


class ForbiddenGroupsRequest(BaseModel):
    """勾选要写进提示词的禁止项组（用户在「固定提示词」表里筛）。

    ``groups`` 是**完整选择集**，不是增量 —— 传什么就是什么。
    只允许从库里已有的组里选；未知组会被忽略并如实回报，
    避免前端传错时静默写成空。
    """

    groups: list[str] = Field(default_factory=list)


class ReviseRequest(BaseModel):
    """用户的修改意见（PRD F8.4「按建议重写」）。

    只有一句话：**用户想把这个提示词改成什么样**。
    它落成字段级修改、重新组装、重跑闸门，全程由后端把关。
    """

    instruction: str = Field(description="修改意见，如「场景换成咖啡馆，再多加一个镜头」")


# ----------------------------------------------------------------------
# 应用工厂
# ----------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.load()
    store = ProjectStore(resolved.workspace_dir / "projects")
    library = ph.load_library()

    # ---- 反推队列（用户 2026-10-07：未来给团队用，多人会同时上传）----
    # 固定工作池 + FIFO + 位置可见；进程重启时把卡在"分析中"的项目如实标失败。
    queue = InferenceQueue(
        runner=lambda project_id: _run_inference(store, resolved, project_id),
        workers=resolved.max_concurrent_infer,
        max_queued=resolved.max_queued_infer,
        on_error=lambda project_id, exc: store.update_status(
            project_id, status="failed", stage="失败",
            error=f"{type(exc).__name__}: {exc}",
        ),
    )
    queue.start()
    InferenceQueue.recover_orphans(
        store.list(), save=store.save,
        reason="服务重启中断（排队或运行中），请重新反推",
    )

    app = FastAPI(
        title="爆款视频复刻提示词 Agent",
        version="0.1.0",
        description="上传爆款视频 → 反推提示词结构 → 换素材填内容 → 组装可交付提示词",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = resolved
    app.state.store = store
    # 运维/测试都要能拿到池子（测试会替换它的 runner 来验证 503 与排队位置）
    app.state.queue = queue

    _register_routes(app, resolved, store, library, queue)
    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """若前端已构建，就由同一个端口提供静态页面。

    单端口的好处：不必开两个服务、不必配跨域、团队里发一个地址即可用。
    没构建时（开发期）走 Vite 的 5173，跨域已在上面的中间件放开。
    """
    dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if not dist.is_dir():
        return

    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = dist / "index.html"

    @app.get("/", include_in_schema=False)
    def _index() -> HTMLResponse:
        return HTMLResponse(index.read_text(encoding="utf-8"))


def _register_routes(
    app: FastAPI,
    settings: Settings,
    store: ProjectStore,
    library: ph.PhraseLibrary,
    queue: InferenceQueue,
) -> None:
    # 用显式 router 装配，而不是在闭包里直接挂 @app.get：
    # 后者依赖装饰器在闭包里的注册时机，路由表可能不完整。
    router = APIRouter(prefix="/api")

    # ==================================================================
    # 能力与资产
    # ==================================================================
    @router.get("/health")
    def health() -> dict[str, Any]:
        from .providers.cv import detect_capabilities
        from .providers.scene_detect import SceneDetectShots

        return {
            "ok": True,
            "framework_version": fw.FRAMEWORK_VERSION,
            "phrase_lib_version": library.version,
            "capabilities": settings.capability_report(),
            "cv": detect_capabilities().to_dict(),
            "shot_detector": (
                "scenedetect" if SceneDetectShots(settings).available else "ffmpeg-mean"
            ),
            # 运维可见：池子里谁在跑（跑了多久）、谁在等。
            # 团队使用时要能一眼看出"是不是有任务卡住了"。
            "queue": queue.state().to_dict(),
        }

    @router.get("/schema")
    def schema() -> dict[str, Any]:
        """框架契约清单 —— 前端据此渲染可编辑面板，不硬编码字段。"""
        return {
            "framework_version": fw.FRAMEWORK_VERSION,
            "fields": fw.schema_digest(),
            "sections": {
                "params": fw.SEC_PARAMS,
                "refs": fw.SEC_REFS,
                "locked": fw.SEC_LOCKED,
                "logline": fw.SEC_LOGLINE,
                "beats": fw.SEC_BEATS,
                "dialogue": fw.SEC_DIALOGUE,
                "forbidden": fw.SEC_FORBIDDEN,
            },
        }

    @router.get("/phrases")
    def phrase_library() -> dict[str, Any]:
        """短语库：模板、禁止项、维度、降级规则。"""
        return {
            "phrase_lib_version": library.version,
            "mandatory_dimensions": list(library.mandatory_dimensions),
            "templates": {
                key: {
                    "label": asset.label,
                    "dimension": asset.dimension,
                    "tier": asset.tier,
                    "source": asset.source,
                    "sentence": asset.sentence,
                    "variants": dict(asset.variants or {}),
                    "mandatory_fragments": list(asset.mandatory_fragments),
                }
                for key, asset in library.templates.items()
            },
            "forbidden_groups": library.forbidden_groups,
            "dimensions": list(library.dimensions.values()),
            "downgrade_rules": list(library.downgrade_rules),
        }

    # ==================================================================
    # 项目
    # ==================================================================
    @router.get("/projects")
    def list_projects() -> dict[str, Any]:
        # 列表里也带上**队列位置** —— 团队使用时要能在左侧栏一眼看出
        # "我这个任务前面还有几条"（前端本来就把 stage 显示在项目名下）。
        out: list[dict[str, Any]] = []
        for project in store.list():
            item = project.to_dict()
            position = queue.position(project.project_id)
            if position and position.get("state") == "queued":
                item["stage"] = f"排队中（前面还有 {position['ahead']} 条）"
                item["queue"] = position
            out.append(item)
        return {"projects": out}

    @router.post("/projects", status_code=201)
    def create_project(payload: CreateProjectRequest) -> dict[str, Any]:
        project = store.create(payload.name, video_path="")
        return project.to_dict()

    @router.get("/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        project = _require(store, project_id)
        return {
            **project.to_dict(),
            "report": project.report,
            "positions": (
                positions.build_positions(project.ir, _prov(project)).to_dict()
                if project.ir else None
            ),
            "reference_manifest": (
                positions.blue_slot_manifest(project.ir) if project.ir else []
            ),
        }

    @router.delete("/projects/{project_id}")
    def delete_project(project_id: str) -> dict[str, Any]:
        if not store.delete(project_id):
            raise HTTPException(404, "项目不存在")
        return {"ok": True}

    # ==================================================================
    # 上传与反推
    # ==================================================================
    @router.post("/projects/{project_id}/video")
    async def upload_video(project_id: str, file: UploadFile) -> dict[str, Any]:
        project = _require(store, project_id)
        suffix = Path(file.filename or "video.mp4").suffix.lower()
        if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}:
            raise HTTPException(400, f"不支持的格式 {suffix}；请上传 mp4 / mov")

        media = store.media_dir(project_id)
        target = media / f"source{suffix}"
        size = 0
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, "文件超过 500MB 上限（PRD F1.1）")
                handle.write(chunk)

        project.video_path = str(target)
        project.status = "created"
        store.save(project)
        return {"video_path": str(target), "bytes": size}

    @router.post("/projects/{project_id}/infer", status_code=202)
    def start_infer(project_id: str, review: bool | None = None) -> dict[str, Any]:
        """触发反推。

        ``review=false`` = **快速反推**：跳过"自检-重写"（实测它一次调用就要 129 秒，
        加上取证与复检，这一段通常占几分钟）。按次生效并记在项目上；
        默认值由 ``REVIEW_ENABLED`` 决定。
        """
        project = _require(store, project_id)
        if not project.video_path:
            raise HTTPException(400, "请先上传视频")
        if review is not None and project.review_enabled != review:
            project.review_enabled = bool(review)
            store.save(project)
        if project.status == "analyzing" and queue.position(project_id) is not None:
            # 幂等：已经在排队/在跑就别重复提交（位置由 /status 实时算）
            return {
                "status": "analyzing", "stage": project.stage,
                "queue": queue.position(project_id),
            }
        # **僵尸状态**：状态写着"分析中"，但队列里根本没有它（进程被杀过、任务丢了）。
        # 这时如果照着旧状态返回，用户点多少次"重新反推"都不会有任何反应 ——
        # 实测就是这样让人觉得"按钮是假的"。所以往下走，**重新入队**。
        if project.status == "analyzing":
            store.update_status(
                project_id, stage="排队中",
                error="上一次反推没有跑完，已重新排队",
            )

        store.update_status(project_id, status="analyzing", stage="排队中", progress=0.0)
        try:
            ticket = queue.submit(project_id)
        except QueueFull as exc:
            # 收不下就如实拒绝，并**把状态回滚**，不留下一个永远"排队中"的项目
            store.update_status(project_id, status="idle", stage="未开始", progress=0.0)
            raise HTTPException(503, str(exc)) from exc
        return {"status": "analyzing", "queue": ticket}

    @router.get("/projects/{project_id}/status")
    def infer_status(project_id: str) -> dict[str, Any]:
        project = _require(store, project_id)
        position = queue.position(project_id)
        stage = project.stage
        # 排队中就把"前面还有几条"直接写进 stage —— 前端不用改就能看见
        if position and position.get("state") == "queued":
            stage = f"排队中（前面还有 {position['ahead']} 条）"
        return {
            "status": project.status,
            "stage": stage,
            "progress": project.progress,
            "error": project.error,
            "has_ir": bool(project.ir),
            "queue": position,
        }

    # ==================================================================
    # IR 读写（门禁在 core 里判）
    # ==================================================================
    @router.put("/projects/{project_id}/field")
    def write_field(project_id: str, payload: WriteRequest) -> dict[str, Any]:
        project = _require(store, project_id)
        if not project.ir:
            raise HTTPException(409, "还没有反推结果")

        current = fw.get_in(project.ir, payload.path)
        if current is None and fw.spec_for(payload.path) is None:
            raise HTTPException(404, f"契约中不存在该位置：{payload.path}")

        decision = fw.can_write(payload.path, value=payload.value, current=current)
        if not decision.allowed:
            # 锁定位在 API 层也必须拦住（PRD F2.3）
            raise HTTPException(403, decision.reason)

        # 旧值必须在写入**之前**取 —— 写完再取拿到的已经是新值，
        # 拿它当旧名字去做级联替换会完全失效（实测踩过）。
        previous = str(current or "")
        try:
            fw.set_in(project.ir, payload.path, payload.value)
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(400, f"路径不可写：{payload.path}（{exc}）") from exc

        # 改人物名要带动一串引用，否则交付物自相矛盾、闸门误报
        cascaded: list[str] = []
        if re.search(r"B2_person\[\d+\]\.name$", payload.path):
            match = re.search(r"B2_person\[(\d+)\]", payload.path)
            if match:
                cascaded = fw.cascade_person_rename(
                    project.ir, int(match.group(1)), previous, str(payload.value or "")
                )

        store.save(project)
        table = positions.build_positions(project.ir, _prov(project))
        pos = table.by_path(payload.path)
        return {
            "ok": True,
            "path": payload.path,
            "value": payload.value,
            "position": pos.to_dict() if pos else None,
            "cascaded": cascaded,
        }

    @router.put("/projects/{project_id}/dialogue")
    def write_dialogue(project_id: str, payload: DialogueWriteRequest) -> dict[str, Any]:
        """填台词。台词是用户的创作，Agent 绝不代写（红线 3）。"""
        project = _require(store, project_id)
        beats = project.ir.get("creative", {}).get("C2_beats", [])
        target = next(
            (b for b in beats if int(b.get("shot_id") or 0) == payload.shot_id), None
        )
        if target is None:
            raise HTTPException(404, f"没有镜头 {payload.shot_id}")
        dialogue = target.setdefault("dialogue", {})
        dialogue["speaker"] = payload.speaker
        dialogue["line"] = payload.line
        dialogue["tone"] = payload.tone
        store.save(project)
        return {"ok": True, "shot_id": payload.shot_id, "dialogue": dialogue}

    @router.post("/projects/{project_id}/bind")
    def bind_asset(project_id: str, payload: BindAssetRequest) -> dict[str, Any]:
        """绑定上游回填的素材句柄（PRD F5.2）。"""
        project = _require(store, project_id)
        references = project.ir.get("references", [])
        if payload.reference_index >= len(references):
            raise HTTPException(404, "素材位不存在")
        references[payload.reference_index]["asset_id"] = payload.asset_id
        store.save(project)
        return {"ok": True, "manifest": positions.blue_slot_manifest(project.ir)}

    @router.put("/projects/{project_id}/delivery")
    def update_delivery(project_id: str, payload: DeliveryRequest) -> dict[str, Any]:
        project = _require(store, project_id)
        delivery = project.ir.setdefault("delivery", {})
        updates = {
            "mode": payload.mode,
            "strategy": payload.strategy,
            "first_frame_mode": payload.first_frame_mode,
            "include_param_header": payload.include_param_header,
        }
        for key, value in updates.items():
            if value is not None:
                delivery[key] = value
        if payload.target_model is not None:
            meta = project.ir.setdefault("meta", {})
            meta["target_model"] = payload.target_model
        store.save(project)
        return {"ok": True, "delivery": delivery}

    @router.put("/projects/{project_id}/forbidden-groups")
    def set_forbidden_groups(
        project_id: str, payload: ForbiddenGroupsRequest
    ) -> dict[str, Any]:
        """按用户勾选重建禁止项清单（「固定提示词」表格页用）。

        **为什么不做成普通字段写入**：``ir.forbidden`` 在契约里是
        "可增不可减"（删条会破坏崩坏兜底）。而用户在表格里取消勾选
        本质就是"减少"，直接写字段会被 G4/D0 的不变量挡住。
        所以这里走专用入口：重建时**强制保留永不裁剪的 D0/D7**，
        用户取消不掉它们（那是红线，不是偏好）。
        """
        project = _require(store, project_id)
        if not project.ir:
            raise HTTPException(409, "还没有反推结果")

        available = set(library.forbidden_groups)
        unknown = [g for g in payload.groups if g not in available]
        # D0/D7 永不可裁（框架 §9.2）：用户取消不掉。
        # 顺序按库里定义的 D 组次序，保证渲染稳定、不因勾选顺序变化。
        picked = {g for g in payload.groups if g in available}
        picked |= {"D0", "D7"}
        enforced = [g for g in ("D0", "D7") if g not in payload.groups]
        selected = [g for g in library.forbidden_groups if g in picked]

        project.ir["forbidden"] = library.forbidden_items(selected)
        store.save(project)
        return {
            "ok": True,
            "groups": selected,
            "enforced": enforced,
            "unknown": unknown,
            "item_count": len(project.ir["forbidden"]),
        }

    # ==================================================================
    # 内容对齐（PRD F6）
    # ==================================================================
    @router.post("/projects/{project_id}/align")
    def align_content(project_id: str, payload: AlignmentRequest) -> dict[str, Any]:
        """把用户粘贴的台词稿对齐到各位置。

        默认把**把握足够**的绑定直接写进提示词；标黄的必须用户确认 ——
        未经确认的猜测不许污染交付物（F6.3 + F6.6）。
        """
        project = _require(store, project_id)
        if not project.ir:
            raise HTTPException(409, "还没有反推结果")

        blocks = alignment.parse_user_content(payload.content)
        if not blocks:
            raise HTTPException(400, "没有解析出任何内容行")
        slots = alignment.collect_slots(project.ir)
        report = alignment.align(blocks, slots, chars_per_sec=payload.chars_per_sec)

        written: list[str] = []
        if payload.apply:
            written = alignment.apply_alignment(project.ir, report)

        project.report = {**project.report, "alignment": report.to_dict()}
        store.save(project)
        return {
            "report": report.to_dict(),
            "written": written,
            "blocks": [b.to_dict() for b in blocks],
            "slots": [s.to_dict() for s in slots],
        }

    @router.get("/projects/{project_id}/align")
    def get_alignment(project_id: str) -> dict[str, Any]:
        project = _require(store, project_id)
        stored = (project.report or {}).get("alignment")
        if stored is None:
            return {"report": None, "slots": (
                [s.to_dict() for s in alignment.collect_slots(project.ir)]
                if project.ir else []
            )}
        return {"report": stored, "slots": (
            [s.to_dict() for s in alignment.collect_slots(project.ir)]
            if project.ir else []
        )}

    @router.put("/projects/{project_id}/align/manual")
    def align_manual(project_id: str, payload: ManualAlignRequest) -> dict[str, Any]:
        """手动对齐：绑定某一段、或解绑清空（PRD F6.5）。

        解绑后该位置回到"未分配"，交付物里会保留原片结构 ——
        这是"标为仅参考"的落地方式。
        """
        project = _require(store, project_id)
        if not project.ir:
            raise HTTPException(409, "还没有反推结果")

        dialogue_path = payload.slot_path
        if fw.spec_for(f"{dialogue_path}.line") is None:
            raise HTTPException(404, f"不是可对齐的位置：{payload.slot_path}")

        if payload.content is not None:
            line, speaker = payload.content, (payload.speaker or "")
        elif payload.block_index is None:
            line, speaker = "", ""      # 解绑
        else:
            stored = (project.report or {}).get("alignment") or {}
            bindings = stored.get("bindings") or []
            block = None
            for item in bindings:
                candidate = item.get("block")
                if candidate and int(candidate.get("index", -1)) == payload.block_index:
                    block = candidate
                    break
            if block is None:
                raise HTTPException(404, f"对齐结果里没有内容块 {payload.block_index}")
            line = str(block.get("content") or "")
            speaker = payload.speaker or str(block.get("speaker") or "")

        fw.set_in(project.ir, f"{dialogue_path}.line", line)
        if fw.spec_for(f"{dialogue_path}.speaker") is not None:
            fw.set_in(project.ir, f"{dialogue_path}.speaker", speaker)

        # 手动确认后清掉该位置的未决状态
        stored = (project.report or {}).get("alignment")
        if stored:
            for item in stored.get("bindings") or []:
                if item.get("slot", {}).get("path") == dialogue_path:
                    item["status"] = "bound"
                    item["requires_confirmation"] = False
                    item["reason"] = "已由用户手动确认"
            _recount_alignment(stored)
            project.report = {**project.report, "alignment": stored}

        store.save(project)
        return {"ok": True, "slot_path": dialogue_path, "line": line, "speaker": speaker}

    # ==================================================================
    # 组装与交付
    # ==================================================================
    @router.get("/projects/{project_id}/assemble")
    def assemble_project(project_id: str) -> dict[str, Any]:
        project = _require(store, project_id)
        if not project.ir:
            raise HTTPException(409, "还没有反推结果")

        doc = assembler.assemble(project.ir, library)
        report = gates.run_gates(
            project.ir, doc, library, alignment=_alignment_of(project)
        )

        warnings = list(doc.warnings)
        warnings.extend(_blocking_hints(project.ir, project.report))
        return {
            "document": {**doc.to_dict(), "warnings": warnings},
            "gates": report.to_dict(),
            "review": _review_summary(project, report),
            "questions": _open_questions(project),
            # 用户意见改写的记账（PRD F8.4）：刷新页面后仍能看到"我当时说了什么、
            # 它改了哪几处、哪条没做到"
            "revisions": list(project.revisions),
            "can_undo": bool(project.ir_history),
            # 逐阶段耗时：用户报过"自检与重写卡很久"，先让他能看见时间花在哪
            "timings": (project.report or {}).get("timings") or {},
            "model_calls": (project.report or {}).get("model_calls") or [],            "compression": [e.to_dict() for e in doc.compression],
            "split_plan": [seg.to_dict() for seg in doc.plan],
            "positions": positions.build_positions(project.ir, _prov(project)).to_dict(),
            "apply_targets": positions.apply_targets(project.ir),
        }

    @router.post("/projects/{project_id}/answer")
    def answer_question(project_id: str, payload: AnswerRequest) -> dict[str, Any]:
        """回答 Agent 提出的选择题：**答案直接落进 IR**，交付物当场更新。

        写入走的是和 ``PUT /field`` 完全相同的那道门禁（``framework.can_write``）——
        UI 拦、API 也拦，两边同一判据，不给"绕过后门改锁定位"开口子。
        """
        from engine.core import review as review_core

        project = _require(store, project_id)
        if not project.ir:
            raise HTTPException(409, "还没有反推结果")
        questions = [q for q in (project.report or {}).get("questions") or []
                     if isinstance(q, dict)]
        target = next((q for q in questions if q.get("id") == payload.id), None)
        if target is None:
            raise HTTPException(404, "没有这道题（可能已经回答过，或换了一次反推）")
        try:
            path, value = review_core.answer_question(target, payload.choice)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        current = fw.get_in(project.ir, path)
        decision = fw.can_write(path, value=value, current=current)
        if not decision.allowed:
            raise HTTPException(403, decision.reason)
        try:
            fw.set_in(project.ir, path, value)
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(400, f"路径不可写：{path}（{exc}）") from exc

        target["answer"] = value
        (project.report or {})["questions"] = questions
        store.save(project)
        return {"path": path, "value": value}

    @router.post("/projects/{project_id}/revise")
    def revise_project(project_id: str, payload: ReviseRequest) -> dict[str, Any]:
        """按用户的一句修改意见改写提示词（PRD F8.4「按建议重写」）。

        **为什么是同步的**：一次改写是一两次模型调用（实测 30~90 秒），
        用户在对话框前面等着，直接返回比让他轮询更好懂；服务端单次调用
        有 ``REVISE_CALL_TIMEOUT_SEC`` 封顶，不会无限挂着。

        **改动落 IR 而不是改文本**：AI 提的是字段级修改，
        经 ``core/revision`` 按契约校验 → 重新组装 → 重跑 14 项闸门。
        改坏了会整体回滚并告诉他为什么（用户意志优先，但不给他一个不能用的版本）。
        """
        import copy as _copy
        from dataclasses import replace as _replace

        from .pipeline import revision as revision_pipeline
        from .providers import mimo as mimo_module

        project = _require(store, project_id)
        if not project.ir:
            raise HTTPException(409, "还没有反推结果")
        instruction = str(payload.instruction or "").strip()
        if not instruction:
            raise HTTPException(400, "请先写一句修改意见")
        if not settings.has_revise_llm:
            raise HTTPException(
                400,
                "改写用的模型 Key 没配：请在 .env 里填 REVISE_API_KEY"
                "（或 MIMO_API_KEY），并可用 REVISE_BASE_URL / MODEL_REVISE 换端点与模型",
            )

        # **清掉本线程的旧账，好让这次改写只留下它自己的调用记录** ——
        # 用户 2026-10-08 要求"只让 DeepSeek 改、不要再送 MiMo 自检"，
        # 这份账就是证据：正常应当恰好一条（`按意见改写 · deepseek-flash`）。
        mimo_module.take_calls()
        run = revision_pipeline.run_revision(
            project.ir, project.report, build_reviser(settings), library, instruction,
            # 默认只调一次模型：改完直接拼出新正文填回去（用户 2026-10-08）。
            # 想恢复"跑一遍 14 项交付闸门 + 按闸门反馈再修一轮"，把
            # .env 里的 REVISE_CHECK_GATES 设成 1。
            check_gates=bool(getattr(settings, "revise_check_gates", False)),
        )
        calls = tuple(mimo_module.take_calls())
        record = _replace(run.record, model_calls=calls) if calls else run.record
        cap = max(1, int(getattr(settings, "revise_max_history", 10) or 10))

        if run.applied:
            # 撤销用的快照：存**改写前**的 IR（存整份最省心，几 KB）。
            # **必须和记账对上号**（实测踩过）：没落地的轮次（例如"与现状相同"）
            # 会记一笔账但不推快照，两者若各弹各的，撤销就会"删掉 A 的记录、
            # 退回 B 的版本"。所以快照带上是哪条记录的。
            project.ir_history.append({"id": record.id, "ir": _copy.deepcopy(project.ir)})
            project.ir_history = project.ir_history[-cap:]
            project.ir = run.ir

        project.revisions.append(record.to_dict())
        project.revisions = project.revisions[-cap:]
        store.save(project)

        return {
            "ok": run.applied,
            "record": record.to_dict(),
            "changes": [item.to_dict() for item in record.changes],
            "refusals": [item.to_dict() for item in record.refusals],
            "rejected": [item.to_dict() for item in record.rejected],
            "notes": [item.to_dict() for item in record.notes],
            "verdict_before": record.verdict_before,
            "verdict_after": record.verdict_after,
            "can_undo": bool(project.ir_history),
        }

    @router.post("/projects/{project_id}/revise/undo")
    def undo_revision(project_id: str) -> dict[str, Any]:
        """撤销最近一次**真的改了东西**的那一轮：回到那一轮之前的提示词。

        记账与快照**成套弹出**：撤销后把该轮之后那些"没落地"的记账一起去掉，
        否则历史里会留着一条声称改了、实际已经回退的记录。
        """
        project = _require(store, project_id)
        if not project.ir or not project.ir_history:
            raise HTTPException(409, "没有可以撤销的改写")
        entry = project.ir_history.pop()
        # 兼容两种形态：新的 {"id", "ir"} 与早期的裸 IR
        if isinstance(entry, dict) and "ir" in entry:
            previous, target = dict(entry.get("ir") or {}), str(entry.get("id") or "")
        else:
            previous, target = dict(entry or {}), ""
        if previous:
            project.ir = previous
        undone: dict[str, Any] = {}
        if target:
            while project.revisions and str(project.revisions[-1].get("id") or "") != target:
                project.revisions.pop()
        if project.revisions:
            undone = project.revisions.pop()
        store.save(project)
        return {"ok": True, "undone": undone, "can_undo": bool(project.ir_history)}

    @router.get("/projects/{project_id}/export")
    def export_project(
        project_id: str,
        form: str = Query(default="plain", pattern="^(plain|markdown|json)$"),
    ) -> Any:
        project = _require(store, project_id)
        if not project.ir:
            raise HTTPException(409, "还没有反推结果")
        doc = assembler.assemble(project.ir, library)
        report = gates.run_gates(
            project.ir, doc, library, alignment=_alignment_of(project)
        )

        # 红线 10：闸门未全绿时不许声称可交付。
        # **三种形态都要带判定**（原来只有 markdown 带，而 plain 是默认形态、
        # 也正是"直接粘进生成平台"的那一份 —— 全红也照样给一份干净正文）。
        markdown_header = (
            f"# 交付自检：{report.verdict}"
            f"（阻断 {len(report.blockers)} / 告警 {len(report.warnings)}）\n\n"
        )
        plain_note = ""
        if report.verdict != "ALL_GREEN":
            names = "、".join(g.name for g in (report.blockers or report.warnings)[:3])
            plain_note = (
                f"【交付自检】{report.verdict}"
                f"（阻断 {len(report.blockers)} / 告警 {len(report.warnings)}）：{names}\n"
                f"——这一行不是提示词内容，粘进平台前请删掉。\n\n"
            )
        verdict_headers = {
            "X-Delivery-Verdict": report.verdict,
            "X-Delivery-Blockers": str(len(report.blockers)),
            "X-Delivery-Warnings": str(len(report.warnings)),
        }
        text = doc.render(OutputForm(form))
        if form == "json":
            payload = doc.to_dict()
            payload["gates"] = report.to_dict()
            return PlainTextResponse(
                json.dumps(payload, ensure_ascii=False, indent=2),
                media_type="application/json; charset=utf-8",
                headers=verdict_headers,
            )
        if form == "markdown":
            return PlainTextResponse(
                markdown_header + text,
                media_type="text/markdown; charset=utf-8",
                headers=verdict_headers,
            )
        return PlainTextResponse(
            plain_note + text,
            media_type="text/plain; charset=utf-8",
            headers=verdict_headers,
        )

    @router.get("/projects/{project_id}/video")
    def stream_video(project_id: str) -> FileResponse:
        project = _require(store, project_id)
        path = Path(project.video_path)
        if not path.exists():
            raise HTTPException(404, "视频文件已丢失")
        return FileResponse(path)

    app.include_router(router)


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------
def _require(store: ProjectStore, project_id: str) -> Project:
    project = store.get(project_id)
    if project is None:
        raise HTTPException(404, "项目不存在")
    return project

def _recount_alignment(stored: dict[str, Any]) -> None:
    """手动确认后重算汇总 —— 否则门禁还在按旧状态判定。"""
    bindings = stored.get("bindings") or []
    bound = sum(1 for b in bindings if b.get("status") == "bound")
    pending = sum(1 for b in bindings if b.get("status") == "pending")
    unmatched = sum(1 for b in bindings if b.get("status") == "unmatched")
    orphans = len(stored.get("orphans") or [])
    stored["summary"] = {
        "total_slots": len(bindings),
        "bound": bound,
        "pending": pending,
        "unmatched": unmatched,
        "orphans": orphans,
        "has_unresolved": bool(pending or unmatched or orphans),
    }


def _alignment_of(project: Project) -> dict[str, Any] | None:
    """取出本次项目的对齐结果，供门禁 G15 判定。

    没用过内容对齐就返回 None —— 那样 G15 不会出现在报告里，
    不给不用对齐的用户添噪声。
    """
    return (project.report or {}).get("alignment")


def _prov(project: Project) -> dict[str, Any]:
    """把已存的 provenance 还原成对象，供位置表着色与"待确认"统计。

    还原失败的位置直接跳过 —— provenance 只是增强信息，
    缺了它位置表仍应正常工作，不能因此 500。
    """
    from engine.core.types import Provenance

    raw = (project.report or {}).get("provenance") or {}
    out: dict[str, Provenance] = {}
    for path, item in raw.items():
        try:
            out[path] = Provenance.from_dict(item)
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _open_questions(project: Any) -> list[dict[str, Any]]:
    """还没回答的选择题。已答的不再显示（它们已经落进 IR 了）。"""
    out: list[dict[str, Any]] = []
    for item in (project.report or {}).get("questions") or []:
        if not isinstance(item, dict) or str(item.get("answer") or "").strip():
            continue
        out.append({
            "id": item.get("id"),
            "path": item.get("path"),
            "text": item.get("text"),
            "options": list(item.get("options") or []),
            "reason": item.get("reason") or "",
        })
    return out


def _review_summary(project: Any, report: Any) -> dict[str, Any]:
    """把"用户需要知道/确认"的东西收成一条摘要。

    **为什么必须有这个出口**：置信度分级、反推降级说明、CV↔模型冲突、闸门判定
    以前**一个都没有出口** —— 前端只有一个提示词文本框。于是"低置信度标黄待确认"
    这条能力在真实数据上等于不存在（实测 5 个项目 `conflicts` 全空、
    ASR 置信度 0.65~0.69 落在"必须确认"档也没人看见），用户只能靠肉眼抓错。

    只汇总**用户能据此行动**的四类，按严重程度排序，超过 12 条截断并报出总数。
    """
    from engine.core import framework as fw
    from engine.core.types import CONFIDENCE_ACCEPT

    items: list[dict[str, Any]] = []

    # 结构化提问：**优先级最高**（它是唯一"用户点一下就能把交付物改对"的入口）
    for item in _open_questions(project):
        items.append({
            "kind": "question", "level": "review",
            "label": f"需要你定一下：{item['text']}",
            "detail": f"选项：{' / '.join(item['options'])}"
                      + (f"｜{item['reason']}" if item.get("reason") else ""),
            "suggestion": "在结果页点选即可（答案会直接写进提示词）。",
        })

    # 动作类但没绑参考视频：那两句 1:1 复刻句落不了地，**必须让用户看见**
    action_hint = _action_binding_hint(getattr(project, "ir", None) or {})
    if action_hint:
        items.append({
            "kind": "hint", "level": "review", "label": "需要你补一步",
            "detail": action_hint, "suggestion": "",
        })

    # **慢反推要说清时间花在哪**（用户 2026-10-07 报"自检与重写卡很久"）：
    # 一次反推要调十几次模型，用户看到 0.97 停着不动时最想知道的就是这件事。
    timings = {
        str(k): float(v)
        for k, v in ((getattr(project, "report", None) or {}).get("timings") or {}).items()
        if isinstance(v, (int, float))
    }
    total = sum(timings.values())
    if timings and total >= 300:
        slowest = sorted(timings.items(), key=lambda kv: -kv[1])[:3]
        # 阶段计时之外，再点出**最慢的那几次模型调用** ——
        # 光看"自检 563 秒"没法判断是"一次调用想了 4 分钟"还是"重试了两次"
        calls = sorted(
            (
                (str(item.get("purpose") or ""), float(item.get("seconds") or 0),
                 bool(item.get("ok")), str(item.get("model") or ""))
                for item in ((getattr(project, "report", None) or {}).get("model_calls") or [])
                if isinstance(item, dict)
            ),
            key=lambda row: -row[1],
        )[:3]
        detail = "最慢：" + "、".join(f"{name} {value:.0f}s" for name, value in slowest)
        if calls:
            detail += "｜最慢的模型调用：" + "、".join(
                f"{name}{'' if ok else '（失败）'} {value:.0f}s（{model}）"
                for name, value, ok, model in calls
            )
        # **token 也要能看见**（用户 2026-10-07 问"换模型是不是变贵了"）：
        # 快模型可能正是靠"烧更多 token"换来的，不显示就只能靠猜。
        usage_rows = [
            item for item in ((getattr(project, "report", None) or {}).get("model_calls") or [])
            if isinstance(item, dict)
        ]
        if usage_rows:
            prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in usage_rows)
            out_tokens = sum(int(item.get("completion_tokens") or 0) for item in usage_rows)
            reasoning = sum(int(item.get("reasoning_tokens") or 0) for item in usage_rows)
            detail += (
                f"｜共 {len(usage_rows)} 次模型调用，输入 {prompt_tokens} / 输出 {out_tokens} token"
                f"（其中推理 {reasoning}）"
            )
        items.append({
            "kind": "timing", "level": "review",
            "label": f"这次反推花了 {int(total // 60)} 分 {int(total % 60)} 秒",
            "detail": detail,
            "suggestion": "想更快：可以在设置里关掉自检-重写（REVIEW_ROUNDS=0）或调小它的时间预算。",
        })

    for gate in report.blockers:
        items.append({
            "kind": "gate", "level": "block", "label": gate.name,
            "detail": gate.detail, "suggestion": gate.suggestion,
        })
    for gate in report.warnings:
        items.append({
            "kind": "gate", "level": "warn", "label": gate.name,
            "detail": gate.detail, "suggestion": gate.suggestion,
        })

    stored = project.report or {}
    for text in stored.get("degradations") or []:
        items.append({
            "kind": "degradation", "level": "warn", "label": "反推降级",
            "detail": str(text), "suggestion": "",
        })
    for conflict in stored.get("conflicts") or []:
        items.append({
            "kind": "conflict", "level": "warn",
            "label": f"{conflict.get('field')} 镜头{conflict.get('shot_index')}",
            "detail": (
                f"CV={conflict.get('cv_value')}，模型={conflict.get('model_value')}；"
                f"{conflict.get('resolution')}"
            ),
            "suggestion": "",
        })

    # 自检-重写回路的记账：**交付物被重写过，用户必须知道改了什么**。
    # 没有这个出口，用户看到的是"Agent 复核过"还是"没复核"，完全无从判断。
    applied = 0
    for note in stored.get("reviews") or []:
        if not isinstance(note, dict):
            continue
        did_apply = bool(note.get("applied"))
        applied += 1 if did_apply else 0
        items.append({
            "kind": "critique",
            "level": "warn" if did_apply else "review",
            "label": ("自检已改写" if did_apply else "自检拒绝的修改") + f"：{note.get('path') or '—'}",
            "detail": f"{note.get('reason') or ''}｜{note.get('detail') or ''}".strip("｜"),
            "suggestion": "" if did_apply else "自检想改这一处但越界了，已在审计记录里。",
        })
    if applied:
        items.append({
            "kind": "critique", "level": "review", "label": "自检回路",
            "detail": f"交付前复核并改写了 {applied} 处（每次反推都会跑，可在设置里关掉）",
            "suggestion": "",
        })
    for path, entry in (stored.get("provenance") or {}).items():
        if not isinstance(entry, dict):
            continue
        try:
            confidence = float(entry.get("confidence") or 1.0)
        except (TypeError, ValueError):
            continue
        if confidence >= CONFIDENCE_ACCEPT:
            continue
        spec = fw.spec_for(path)
        items.append({
            "kind": "confidence", "level": "review",
            "label": spec.label if spec else path,
            "detail": (
                f"置信度 {confidence:.2f}（低于确认线 {CONFIDENCE_ACCEPT:g}），"
                f"依据：{entry.get('note') or entry.get('basis') or '—'}"
            ),
            "suggestion": "这一项把握不足，请对着原片核一遍。",
        })

    # 阻断排最前，其次告警，最后是"请核对"；同名项只留最严重的那条
    # （`meta.genre` 与 `locked.B1_scene.type` 的标签都叫「场景类型」，
    #  不去重的话用户会看到两条一模一样的，以为系统坏了）
    order = {"block": 0, "warn": 1, "review": 2}
    items.sort(key=lambda item: order.get(item["level"], 3))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item["kind"], item["label"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return {
        "count": len(unique),
        "items": unique[:12],
        "truncated": max(0, len(unique) - 12),
    }


def _action_binding_hint(ir: dict[str, Any]) -> str:
    """动作类片子的提示（``""`` 表示不用提示）。

    用户 2026-10-07 定下的用法：交付物里直接写 ``[参考动作视频1]``，
    **用户自己在生成平台把这个位置传上原片** —— 所以这不再是"要去绑定"的待办，
    只是一个提醒。

    只有在**老数据**（早于本轮、素材清单里没有这条位置）时才提示一句：
    重新反推一次就会自动补上。
    """
    from engine.core.assembler import needs_action_physics
    from engine.app.pipeline.to_ir import ACTION_VIDEO_LABEL

    if not ir or not needs_action_physics(ir):
        return ""
    declared = any(
        str(ref.get("token_label") or "").strip() == ACTION_VIDEO_LABEL
        for ref in ir.get("references", [])
    )
    if declared:
        return ""
    return (
        "这条片子是**动作类**，交付物里写的是 `[视频1]` —— "
        "重新反推一次会把它改成更清楚的 `[参考动作视频1]`（素材位会自动登记）。"
    )


def _blocking_hints(ir: dict[str, Any], report: dict[str, Any] | None = None) -> list[str]:
    """把"用户必须先补什么"直接列出来，而不是让用户去闸门列表里找。

    这些不是错误，是**待办**：反推不出人物时，交付物会缺"人物一致性"整句，
    而那句话决定"像不像原片"。只在 G13 里出现一条告警太容易被忽略。

    ``report`` 目前不参与判断，保留形参是为了让调用点不必再改一次 ——
    具体的"为什么没人物"由反推阶段写进 ``report.degradations``。
    """
    from engine.core.types import UNKNOWN

    hints: list[str] = []

    # 占位符泄漏检查：unknown 是内部标记，绝不该出现在用户界面上。
    # 实测出现过场景光线显示为 "unknown"，用户看到根本不知道该填什么。
    pending = [p for p in framework_paths(ir) if fw.get_in(ir, p) in (UNKNOWN, "unknown")]
    if pending:
        hints.append(
            f"{len(pending)} 个位置未反推出来（界面显示为「待填」，留空是刻意的 —— "
            f"低置信度宁可让你填，也不猜）："
            f"{'、'.join(fw.label_of(p) for p in pending[:6])}"
            + ("…" if len(pending) > 6 else "")
        )

    if not ir.get("locked", {}).get("B2_person"):
        hints.append(
            "未反推出人物清单：请补充角色名并绑定多视图素材，"
            "否则交付物缺少「人物一致性」段落 —— 这决定跨镜头会不会漂移。"
            "（反推报告里的「降级说明」会写清是画面里没人，还是有人但认不出身份）"
        )
    beats = ir.get("creative", {}).get("C2_beats", [])
    has_dialogue = any((b.get("dialogue") or {}).get("line") for b in beats)

    # 动作类片子：**那两句动作复刻句引用的是"上传的原片"**，
    # 没绑定参考视频的话，生成平台根本看不到原片 —— 这句就是空的。
    # 用户 2026-10-07 明确要求动作类要有 1:1 复刻句，所以这里必须说清怎么让它生效。
    action_hint = _action_binding_hint(ir)
    if action_hint:
        hints.append(action_hint)
    # 音色只在**有台词**时才提示：静音片子里没有声音可以克隆，
    # 这时候提示"缺音色绑定"与下面那条"台词为空"自相矛盾（实测踩过）。
    if has_dialogue and not ir.get("locked", {}).get("B6_audio"):
        hints.append(
            "未反推出音色绑定：每个有台词的角色都需要一条音色参考，"
            "否则口型与音色对不上（闸门 G6 会阻断）。"
        )
    if beats and not has_dialogue:
        hints.append(
            "台词为空：请粘贴你自己的台词。台词是你的创作，Agent 不代写（红线 3）。"
        )
    return hints


def framework_paths(ir: dict[str, Any]) -> list[str]:
    """列出 IR 里所有属于契约的具体位置路径（排除元数据与非位置字段）。"""
    from engine.core.contracts import NON_POSITION_KEYS

    paths: list[str] = []
    for path, _ in fw.flatten(ir):
        if path.split(".", 1)[0].split("[", 1)[0] in NON_POSITION_KEYS:
            continue
        if fw.spec_for(path) is not None:
            paths.append(path)
    return paths


def build_reviser(settings: Settings) -> Any:
    """按配置组装「用户意见改写」的模型适配器。

    **单独抽一个函数**是为了测试能注入替身（与 ``build_inferencer`` 同一套路）——
    改写是同步接口，端到端测试不该真烧一次模型调用。
    """
    from .providers.mimo_analysis import MimoPromptReviser

    return MimoPromptReviser(settings)


def _run_inference(store: ProjectStore, settings: Settings, project_id: str) -> None:
    """后台线程跑反推。状态与进度都写进项目，供前端轮询。"""
    import time as _time

    project = store.get(project_id)
    if project is None:
        return

    # **逐阶段计时**（用户 2026-10-07 报"自检与重写卡很久"之后加的）：
    # 进度回调在每个阶段开始时打点，用它累计每个阶段实际花了多久 ——
    # 一次反推要调十几次模型，没有这个只能靠猜"时间花在哪"。
    timings: dict[str, float] = {}
    last = {"stage": "", "at": _time.monotonic()}

    def on_progress(stage: str, progress: float) -> None:
        # **协作式中止**（团队使用）：项目被删掉之后，正在跑的反推没必要继续 ——
        # 它还会调十几次模型、白烧钱。进度回调是各阶段之间的天然检查点。
        if store.get(project_id) is None:
            raise RuntimeError("项目已被删除，反推中止")
        now = _time.monotonic()
        if last["stage"]:
            timings[last["stage"]] = round(
                timings.get(last["stage"], 0.0) + (now - last["at"]), 1
            )
        last["stage"], last["at"] = stage, now
        store.update_status(project_id, stage=stage, progress=progress)

    try:
        # 快速反推（按项目/按次）：跳过自检-重写。实测这一段一次调用就要 129 秒，
        # 跳过它是最直接的"快几分钟"手段；**但必须如实说明**，不能假装复核过。
        from dataclasses import replace as _replace

        from .providers import mimo as mimo_module

        mimo_module.take_calls()          # 清掉本线程的旧账，避免串到上一次
        run_settings = settings
        if not getattr(project, "review_enabled", True):
            run_settings = _replace(settings, review_rounds=0)
        inferencer = build_inferencer(run_settings)
        workspace = store.media_dir(project_id) / "work"
        ir, report = inferencer.infer_to_ir(
            project.video_path, workspace=workspace, on_progress=on_progress
        )
        if not getattr(project, "review_enabled", True):
            on_progress("跳过交付前自检（快速反推）", 0.99)
            report.degradations = tuple(report.degradations) + (
                "本次为快速反推：跳过了「自检-重写」，交付前**没有**复核环节"
                "（少花几分钟，代价是没人回头看一遍）。",
            )
        # 收尾把最后一个阶段也记上（它不会再有下一次回调）
        if last["stage"]:
            timings[last["stage"]] = round(
                timings.get(last["stage"], 0.0) + (_time.monotonic() - last["at"]), 1
            )
        report.timings = dict(timings)
        # **每一次模型调用的记账**：阶段计时回答不了"自检那 563 秒到底花在哪一次调用"
        report.model_calls = tuple(mimo_module.take_calls())
        # **收尾时必须重新取一次**（实测踩过）：原来写 `store.get(...) or project`，
        # 项目在中途被删除时 `get` 返回 None，于是回退到内存里的旧对象并 `save()` ——
        # **把已经删掉的项目又写了回来**，它会以"完成"的姿态复活。
        latest = store.get(project_id)
        if latest is None:
            return                      # 已删除：结果丢弃，不复活
        latest.ir = ir
        latest.report = report.to_dict()
        latest.status = "ready"
        latest.stage = "完成"
        latest.progress = 1.0
        latest.error = ""
        store.save(latest)
    except Exception as exc:  # 后台线程不能静默死掉
        store.update_status(
            project_id, status="failed", stage="失败", error=f"{type(exc).__name__}: {exc}"
        )

app = create_app
