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
import threading
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


# ----------------------------------------------------------------------
# 应用工厂
# ----------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.load()
    store = ProjectStore(resolved.workspace_dir / "projects")
    library = ph.load_library()

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

    _register_routes(app, resolved, store, library)
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
        return {"projects": [p.to_dict() for p in store.list()]}

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
    def start_infer(project_id: str) -> dict[str, Any]:
        project = _require(store, project_id)
        if not project.video_path:
            raise HTTPException(400, "请先上传视频")
        if project.status == "analyzing":
            return {"status": "analyzing", "stage": project.stage}

        store.update_status(project_id, status="analyzing", stage="排队中", progress=0.0)
        thread = threading.Thread(
            target=_run_inference,
            args=(store, settings, project_id),
            daemon=True,
        )
        thread.start()
        return {"status": "analyzing"}

    @router.get("/projects/{project_id}/status")
    def infer_status(project_id: str) -> dict[str, Any]:
        project = _require(store, project_id)
        return {
            "status": project.status,
            "stage": project.stage,
            "progress": project.progress,
            "error": project.error,
            "has_ir": bool(project.ir),
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
        warnings.extend(_blocking_hints(project.ir))
        return {
            "document": {**doc.to_dict(), "warnings": warnings},
            "gates": report.to_dict(),
            "compression": [e.to_dict() for e in doc.compression],
            "split_plan": [seg.to_dict() for seg in doc.plan],
            "positions": positions.build_positions(project.ir, _prov(project)).to_dict(),
            "apply_targets": positions.apply_targets(project.ir),
        }

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

        # 红线 10：闸门未全绿时不许声称可交付，导出时如实带上判定
        header = (
            f"# 交付自检：{report.verdict}"
            f"（阻断 {len(report.blockers)} / 告警 {len(report.warnings)}）\n\n"
        )
        text = doc.render(OutputForm(form))
        if form == "json":
            payload = doc.to_dict()
            payload["gates"] = report.to_dict()
            return PlainTextResponse(
                json.dumps(payload, ensure_ascii=False, indent=2),
                media_type="application/json; charset=utf-8",
            )
        if form == "markdown":
            return PlainTextResponse(
                header + text, media_type="text/markdown; charset=utf-8"
            )
        return PlainTextResponse(text, media_type="text/plain; charset=utf-8")

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


def _blocking_hints(ir: dict[str, Any]) -> list[str]:
    """把"用户必须先补什么"直接列出来，而不是让用户去闸门列表里找。

    这些不是错误，是**待办**：反推不出人物时，交付物会缺"人物一致性"整句，
    而那句话决定"像不像原片"。只在 G13 里出现一条告警太容易被忽略。
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
        )
    if not ir.get("locked", {}).get("B6_audio"):
        hints.append(
            "未反推出音色绑定：每个有台词的角色都需要一条音色参考，"
            "否则口型与音色对不上（闸门 G6 会阻断）。"
        )
    beats = ir.get("creative", {}).get("C2_beats", [])
    if beats and not any((b.get("dialogue") or {}).get("line") for b in beats):
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


def _run_inference(store: ProjectStore, settings: Settings, project_id: str) -> None:
    """后台线程跑反推。状态与进度都写进项目，供前端轮询。"""
    project = store.get(project_id)
    if project is None:
        return

    def on_progress(stage: str, progress: float) -> None:
        store.update_status(project_id, stage=stage, progress=progress)

    try:
        inferencer = build_inferencer(settings)
        workspace = store.media_dir(project_id) / "work"
        ir, report = inferencer.infer_to_ir(
            project.video_path, workspace=workspace, on_progress=on_progress
        )
        project = store.get(project_id) or project
        project.ir = ir
        project.report = report.to_dict()
        project.status = "ready"
        project.stage = "完成"
        project.progress = 1.0
        project.error = ""
        store.save(project)
    except Exception as exc:  # 后台线程不能静默死掉
        store.update_status(
            project_id, status="failed", stage="失败", error=f"{type(exc).__name__}: {exc}"
        )


app = create_app
