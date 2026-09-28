import { useEffect } from "react";
import { DeliveryPanel } from "./DeliveryPanel";
import { PositionPanel } from "./PositionPanel";
import { ProjectRail } from "./ProjectRail";
import { Workbench } from "./Workbench";
import { api } from "./api";
import { useAppState } from "./useAppState";

export default function App() {
  const state = useAppState();
  const activeId = state.project?.project_id ?? null;
  const hasIr = Boolean(state.project?.has_ir);

  // 未选中项目时自动选最新一个，避免打开就是空屏
  useEffect(() => {
    if (!state.project && state.projects.length > 0) {
      void state.selectProject(state.projects[0].project_id);
    }
  }, [state.project, state.projects, state.selectProject]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          <strong>爆款视频复刻 · 提示词 Agent</strong>
          {state.health && (
            <span className="muted tiny">
              框架 {state.health.framework_version} · 短语库 {state.health.phrase_lib_version}
            </span>
          )}
        </div>
        <div className="topbar__status">
          {state.health ? (
            <>
              <span className={state.health.capabilities.llm.configured ? "pill pill--ok" : "pill pill--bad"}>
                {state.health.capabilities.llm.configured
                  ? `模型 ${state.health.capabilities.llm.vision_model}`
                  : "未配置模型 Key"}
              </span>
              {state.health.cv.missing.length > 0 && (
                <span
                  className="pill pill--warn"
                  title={`CV 缺口：${state.health.cv.missing.join("、")}；这些维度当前由模型单独判断，置信度已降档`}
                >
                  CV 缺口 {state.health.cv.missing.length} 项
                </span>
              )}
            </>
          ) : (
            <span className="pill">连接后端中…</span>
          )}
        </div>
      </header>

      <div className="layout">
        <ProjectRail
          projects={state.projects}
          activeId={activeId}
          busy={state.busy}
          onSelect={(id) => void state.selectProject(id)}
          onCreate={(name) => void state.createProject(name)}
          onDelete={(id) => void state.removeProject(id)}
        />

        <main className="main">
          {state.error && (
            <div className="banner banner--error">
              <span>{state.error}</span>
              <button className="icon-btn" onClick={() => window.location.reload()}>
                ✕
              </button>
            </div>
          )}

          {!state.project ? (
            <div className="empty">
              <h2>开始一条复刻</h2>
              <p className="muted">
                左侧新建项目 → 上传爆款视频 → 开始反推。Agent 会拆出它的提示词结构，
                并标明哪些位置能改。
              </p>
            </div>
          ) : (
            <>
              <Workbench
                project={state.project}
                assembled={state.assembled}
                analyzing={state.analyzing}
                stage={state.stage}
                progress={state.progress}
                busy={state.busy}
                videoUrl={api.videoUrl(state.project.project_id)}
                onUpload={(file) => void state.uploadVideo(state.project!.project_id, file)}
                onInfer={() => void state.startInfer(state.project!.project_id)}
                onBind={(index, assetId, slot) => {
                  if (!assetId) return;
                  void state.bindAsset(index, assetId);
                  void slot;
                }}
              />

              <div className="split">
                {state.assembled ? (
                  <>
                    <PositionPanel
                      assembled={state.assembled}
                      onWrite={(path, value) => void state.writeField(path, value)}
                    />
                    <DeliveryPanel
                      assembled={state.assembled}
                      project={state.project}
                      onDelivery={(payload) => void state.updateDelivery(payload)}
                    />
                  </>
                ) : (
                  <div className="empty">
                    <p className="muted">
                      {hasIr ? "正在组装…" : "反推完成后，这里会出现三色位置标注与最终提示词。"}
                    </p>
                  </div>
                )}
              </div>
            </>
          )}
        </main>
      </div>

      {state.toast && (
        <button className="toast" onClick={state.dismissToast}>
          {state.toast}
        </button>
      )}
    </div>
  );
}
