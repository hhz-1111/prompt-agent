/**
 * 主流程：**新建项目 → 上传 → 反推中 → 看提示词**。
 *
 * 用户 2026-09-29 明确要求：
 *   - 固定提示词表格**不放前端**（已经在后台，要打开时再说）；
 *   - 不要「反推详情」页；
 *   - 结果页只留提示词文本框，不要颜色区分、不要字数行；
 *   - 历史项目**搬到左侧常驻**，并且要有「新建项目」入口
 *     （原来只能靠删掉当前项目才能回到上传页）。
 *
 * 所以结果页现在只有一个东西：提示词。
 */

import { useEffect, useRef } from "react";
import { ProjectSidebar } from "./ProjectSidebar";
import { PromptView } from "./PromptView";
import { reviseSourceLabel } from "./ReviseDialog";
import { UploadPage } from "./UploadPage";
import { useAppState } from "./useAppState";

export default function App() {
  const state = useAppState();
  /**
   * "进哪个项目"只在启动时决定一次。
   *
   * 不用这个闸门的话会打架：点侧栏项目 → ``draft`` 置 false、``project`` 还没回来，
   * 那一瞬间回落逻辑会重新读一次地址栏（此时还是 ``?new=1``）把人拽回上传页。
   */
  const decided = useRef(false);

  const project = state.project;
  const assembled = state.assembled;
  const ready = Boolean(project?.has_ir) && Boolean(assembled);
  const activeId = state.draft ? null : project?.project_id ?? null;
  const projectId = project?.project_id ?? null;

  // ① 地址栏里写明了去哪（?new=1 新建 / ?project=xxx 某个项目）就听地址栏的。
  useEffect(() => {
    if (decided.current) return;
    const params = new URLSearchParams(window.location.search);
    const wanted = params.get("project");
    if (params.has("new")) {
      decided.current = true;
      state.beginNewProject();
    } else if (wanted) {
      decided.current = true;
      void state.selectProject(wanted);
    }
  }, [state.selectProject, state.beginNewProject]);

  // ② 地址栏没写明，才回落到"最新项目"（等列表回来再决定）。
  useEffect(() => {
    if (decided.current || state.draft || state.project) return;
    if (state.projects.length === 0) return;
    decided.current = true;
    void state.selectProject(state.projects[0].project_id);
  }, [state.draft, state.project, state.projects, state.selectProject]);

  // 地址栏跟着状态走：刷新后停在原地（不管是在看某个项目，还是在新建项目）。
  useEffect(() => {
    if (state.draft) {
      writeUrl("?new=1");
    } else if (projectId) {
      writeUrl(`?project=${projectId}`);
    }
  }, [state.draft, projectId]);

  /** 首次上传：建项目 → 传视频 → 自动开始反推。用户只点一次。 */
  const handleUpload = async (file: File) => {
    const id = await state.createProject(file.name.replace(/\.[^.]+$/, ""));
    if (!id) return;
    await state.uploadVideo(id, file);
    await state.startInfer(id, state.reviewEnabled);
  };

  const handleNewProject = () => state.beginNewProject();

  const showUpload = state.draft || !project || !project.video_path;
  /**
   * 只要在反推就显示进度 —— **包括"重新反推"**。
   *
   * 这里原来多了个 `!ready` 条件，于是"点重新反推"时：已有结果 → ready 仍为 true
   * → 进度面板不显示、旧提示词继续挂在屏幕上，只有按钮变灰。
   * 用户 2026-10-08 的原话就是「右上角重新反推的按钮是假的」——
   * 它确实在跑（2~3 分钟），但界面上一点反应都没有。
   */
  const showProgress = Boolean(project?.video_path) && state.analyzing;
  const showResult = ready && !state.analyzing;

  return (
    <div className="app">
      <ProjectSidebar
        projects={state.projects}
        activeId={activeId}
        draft={state.draft}
        busy={state.busy}
        onNew={handleNewProject}
        onSelect={(id) => void state.selectProject(id)}
        onRemove={(id) => void state.removeProject(id)}
      />

      <div className="shell">
        <header className="top">
          <div className="top__title">
            <strong>{state.draft ? "新建项目" : project?.name || "未命名项目"}</strong>
            {state.health && !state.health.capabilities.llm.configured && (
              <span className="pill pill--bad">未配置模型 Key</span>
            )}
          </div>
          <div className="top__right">
            {/* 交付前自检：默认开。关掉=快速反推（实测那一段一次模型调用就要 129 秒）。 */}
            <label className="top__switch" title="关掉可快几分钟；关掉后交付物里会如实写明「没有复核」">
              <input
                type="checkbox"
                checked={state.reviewEnabled}
                onChange={(event) => state.setReviewEnabled(event.target.checked)}
              />
              交付前自检
            </label>
            {showResult && (
              <button
                className="btn"
                onClick={() =>
                  void state.startInfer(project!.project_id, state.reviewEnabled)
                }
              >
                重新反推
              </button>
            )}
            {showProgress && (
              <button className="btn" disabled>
                正在反推…
              </button>
            )}
          </div>
        </header>

        {state.error && (
          <div className="banner">
            <span>{state.error}</span>
            <button className="btn btn--tiny" onClick={() => window.location.reload()}>
              ✕
            </button>
          </div>
        )}

        <main className="main">
          {showUpload && (
            <UploadPage onPick={(file) => void handleUpload(file)} busy={state.busy} />
          )}

          {showProgress && (
            <div className="prog">
              <div className="prog__inner">
                <h2>{ready ? "正在重新反推这条视频" : "正在拆这条视频"}</h2>
                <p className="prog__stage">{state.stage || "准备中"}</p>
                <div className="prog__track">
                  <div
                    className="prog__fill"
                    style={{ width: `${Math.round((state.progress || 0) * 100)}%` }}
                  />
                </div>
                {ready && (
                  <p className="muted tiny">
                    旧的提示词先收起来了 —— 跑完这里会自动换成新的一版。
                  </p>
                )}
                <p className="muted tiny">
                  通常 3~10 分钟（看模型此刻快慢）：镜头切分、运镜与景别量测、人物识别、
                  语音识别、结构归纳，交付前还会自己回头检查一遍并去补看没看清的地方。
                  {state.stage.includes("自检") && "现在正在「自检与重写」—— 这一段最慢。"}
                  {!state.reviewEnabled && "（本次已关掉交付前自检，会快几分钟。）"}
                </p>
              </div>
            </div>
          )}

          {!showUpload && !showProgress && !showResult && (
            <div className="prog">
              <div className="prog__inner">
                <h2>这条视频还没反推</h2>
                <button
                  className="btn btn--primary"
                  onClick={() => void state.startInfer(project!.project_id)}
                >
                  开始反推
                </button>
              </div>
            </div>
          )}

          {showResult && (
            <PromptView
              document={assembled!.document}
              review={assembled!.review}
              questions={assembled!.questions}
              onAnswer={state.answerQuestion}
              revisions={assembled!.revisions ?? []}
              revising={state.revising}
              canUndo={assembled!.can_undo}
              reviseSource={reviseSourceLabel(state.health)}
              onRevise={(instruction) => void state.revise(instruction)}
              onUndoRevision={() => void state.undoRevision()}
            />
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

/**
 * 把地址栏换成新的查询串（不新增历史记录）。
 *
 * 为什么必须同步：地址栏是刷新后唯一还记得的状态。不同步的话，
 * 点「新建项目」再刷新会被 URL 里的 ?project 拽回上一个项目，
 * 看起来就像"新建没生效"。
 */
function writeUrl(search: string) {
  const next = `${window.location.pathname}${search}`;
  if (next !== `${window.location.pathname}${window.location.search}`) {
    window.history.replaceState(null, "", next);
  }
}
