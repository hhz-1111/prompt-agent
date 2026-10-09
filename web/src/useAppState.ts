/**
 * 应用状态：只在这里管数据加载与副作用，组件保持纯展示。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  api,
  type AssembleResponse,
  type HealthInfo,
  type ProjectDetail,
  type ProjectSummary,
  type ReviseResponse,
} from "./api";

const POLL_INTERVAL_MS = 1200;

export interface AppState {
  health: HealthInfo | null;
  projects: ProjectSummary[];
  project: ProjectDetail | null;
  assembled: AssembleResponse | null;
  /** 回答 Agent 的选择题：答案写进 IR 后自动刷新交付物。 */
  answerQuestion: (questionId: string, choice: string) => void;
  busy: boolean;
  analyzing: boolean;
  stage: string;
  progress: number;
  error: string;
  toast: string;
  /**
   * 正在"新建项目"：当前不挂任何项目，界面停在上传页。
   *
   * 为什么需要一个显式状态，而不是把 ``project`` 置空就行：
   * ``project`` 为空同时也是"还没选项目"的信号，会被自动回落到最新项目的逻辑
   * 立刻抢回来，于是点「新建项目」会瞬间被弹回上一个项目。
   */
  draft: boolean;

  refreshProjects: () => Promise<void>;
  selectProject: (id: string) => Promise<void>;
  /** 回到上传页开始一个新项目（**不建空项目**，选到文件才建）。 */
  beginNewProject: () => void;
  createProject: (name: string) => Promise<string | null>;
  removeProject: (id: string) => Promise<void>;
  uploadVideo: (id: string, file: File) => Promise<void>;
  startInfer: (id: string, review?: boolean) => Promise<void>;
  reloadProject: () => Promise<void>;
  reloadAssembled: () => Promise<void>;
  writeField: (path: string, value: unknown) => Promise<void>;
  writeDialogue: (shotId: number, payload: { speaker: string; line: string; tone: string }) => Promise<void>;
  bindAsset: (index: number, assetId: string) => Promise<void>;
  updateDelivery: (payload: Record<string, unknown>) => Promise<void>;
  setForbiddenGroups: (groups: string[]) => Promise<void>;
  /** 交付前是否跑「自检-重写」。关掉=快速反推（实测这一段要几分钟）。 */
  reviewEnabled: boolean;
  setReviewEnabled: (value: boolean) => void;
  /**
   * 按用户的一句修改意见改写提示词（PRD F8.4）。
   *
   * 同步等待（实测 30~90 秒），所以组件那边要显示"已等多少秒"。
   * 返回后端结果，失败也返回 —— 界面要把"哪条没做到、为什么"显示出来。
   */
  revise: (instruction: string) => Promise<ReviseResponse | null>;
  undoRevision: () => Promise<void>;
  revising: boolean;
  dismissToast: () => void;
}

export function useAppState(): AppState {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [assembled, setAssembled] = useState<AssembleResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [stage, setStage] = useState("");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [draft, setDraft] = useState(false);
  // 交付前自检：**记住选择**（存在浏览器里）。关掉就是快速反推。
  const [reviewEnabled, setReviewEnabledState] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem("mod.review") !== "0";
    } catch {
      return true;
    }
  });
  const setReviewEnabled = useCallback((value: boolean) => {
    setReviewEnabledState(value);
    try {
      window.localStorage.setItem("mod.review", value ? "1" : "0");
    } catch {
      /* 隐私模式下写不了，忽略 */
    }
  }, []);

  const currentId = useRef<string | null>(null);

  const report = useCallback((err: unknown) => {
    const message =
      err instanceof ApiError ? err.message : err instanceof Error ? err.message : String(err);
    setError(message);
  }, []);

  const refreshProjects = useCallback(async () => {
    try {
      setProjects(await api.listProjects());
    } catch (err) {
      report(err);
    }
  }, [report]);

  const reloadProject = useCallback(async () => {
    const id = currentId.current;
    if (!id) return;
    try {
      setProject(await api.getProject(id));
    } catch (err) {
      report(err);
    }
  }, [report]);

  const reloadAssembled = useCallback(async () => {
    const id = currentId.current;
    if (!id) return;
    try {
      setAssembled(await api.assemble(id));
    } catch (err) {
      // 还没反推完时 409 是正常的，不当错误弹
      if (err instanceof ApiError && err.status === 409) {
        setAssembled(null);
        return;
      }
      report(err);
    }
  }, [report]);

  useEffect(() => {
    void api.health().then(setHealth).catch(report);
    void refreshProjects();
  }, [refreshProjects, report]);

  /**
   * 回答 Agent 的选择题。
   *
   * 为什么值得单独做一条：这是**唯一一个"用户点一下就把交付物改对"的入口** ——
   * 以前置信度低只进「待确认」列表，用户看完还得自己回去改正文。
   */
  const answerQuestion = useCallback(
    async (questionId: string, choice: string) => {
      const id = currentId.current;
      if (!id) return;
      try {
        await api.answerQuestion(id, questionId, choice);
        setToast(`已按「${choice}」更新提示词`);
        await reloadAssembled();
      } catch (err) {
        report(err);
      }
    },
    [reloadAssembled, report],
  );

  const selectProject = useCallback(
    async (id: string) => {
      currentId.current = id;
      setDraft(false);
      setAssembled(null);
      setError("");
      try {
        const detail = await api.getProject(id);
        setProject(detail);
        setAnalyzing(detail.status === "analyzing");
        setStage(detail.stage);
        setProgress(detail.progress);
        if (detail.has_ir) await reloadAssembled();
      } catch (err) {
        report(err);
      }
    },
    [reloadAssembled, report],
  );

  /** 回上传页开新项目。旧项目原样留在列表里，什么都不删。 */
  const beginNewProject = useCallback(() => {
    currentId.current = null;
    setDraft(true);
    setAnalyzing(false);
    setProject(null);
    setAssembled(null);
    setError("");
    setStage("");
    setProgress(0);
  }, []);

  const createProject = useCallback(
    async (name: string) => {
      setBusy(true);
      try {
        const created = await api.createProject(name);
        await refreshProjects();
        await selectProject(created.project_id);
        return created.project_id;
      } catch (err) {
        report(err);
        return null;
      } finally {
        setBusy(false);
      }
    },
    [refreshProjects, report, selectProject],
  );

  const removeProject = useCallback(
    async (id: string) => {
      setBusy(true);
      try {
        await api.deleteProject(id);
        // 删掉的正好是当前项目 → 直接回上传页。
        // 不能只是把 project 置空等自动回落：列表这时还没刷新，
        // 回落逻辑会先挑到刚删掉的那个 id，白挨一个 404。
        if (currentId.current === id) beginNewProject();
        await refreshProjects();
      } catch (err) {
        report(err);
      } finally {
        setBusy(false);
      }
    },
    [beginNewProject, refreshProjects, report],
  );

  const uploadVideo = useCallback(
    async (id: string, file: File) => {
      setBusy(true);
      setError("");
      try {
        await api.uploadVideo(id, file);
        setToast(`已上传 ${file.name}`);
        await reloadProject();
      } catch (err) {
        report(err);
      } finally {
        setBusy(false);
      }
    },
    [reloadProject, report],
  );

  const startInfer = useCallback(
    async (id: string, review?: boolean) => {
      setError("");
      try {
        await api.startInfer(id, review);
        setAnalyzing(true);
        setStage(review === false ? "排队中（快速反推，跳过自检）" : "排队中");
        setProgress(0);
      } catch (err) {
        report(err);
      }
    },
    [report],
  );

  // 反推进度轮询：完成后自动拉取结果与组装
  useEffect(() => {
    if (!analyzing) return;
    const id = currentId.current;
    if (!id) return;

    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const status = await api.inferStatus(id);
        if (cancelled) return;
        setStage(status.stage);
        setProgress(status.progress);
        if (status.status === "failed") {
          setAnalyzing(false);
          setError(status.error || "反推失败");
          return;
        }
        if (status.status === "ready") {
          setAnalyzing(false);
          await reloadProject();
          await reloadAssembled();
          setToast("反推完成");
        }
      } catch (err) {
        if (!cancelled) report(err);
      }
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [analyzing, reloadAssembled, reloadProject, report]);

  const writeField = useCallback(
    async (path: string, value: unknown) => {
      const id = currentId.current;
      if (!id) return;
      try {
        await api.writeField(id, path, value);
        await reloadAssembled();
      } catch (err) {
        report(err);
      }
    },
    [reloadAssembled, report],
  );

  const writeDialogue = useCallback(
    async (shotId: number, payload: { speaker: string; line: string; tone: string }) => {
      const id = currentId.current;
      if (!id) return;
      try {
        await api.writeDialogue(id, { shot_id: shotId, ...payload });
        await reloadAssembled();
      } catch (err) {
        report(err);
      }
    },
    [reloadAssembled, report],
  );

  const bindAsset = useCallback(
    async (index: number, assetId: string) => {
      const id = currentId.current;
      if (!id) return;
      try {
        const result = await api.bindAsset(id, index, assetId);
        setProject((prev) => (prev ? { ...prev, reference_manifest: result.manifest } : prev));
        await reloadAssembled();
        setToast(`已绑定素材 ${assetId}`);
      } catch (err) {
        report(err);
      }
    },
    [reloadAssembled, report],
  );

  const updateDelivery = useCallback(
    async (payload: Record<string, unknown>) => {
      const id = currentId.current;
      if (!id) return;
      try {
        await api.updateDelivery(id, payload);
        await reloadAssembled();
      } catch (err) {
        report(err);
      }
    },
    [reloadAssembled, report],
  );

  const dismissToast = useCallback(() => setToast(""), []);

  const [revising, setRevising] = useState(false);

  /**
   * 按用户的修改意见改写提示词。
   *
   * 结果**不写在前端**：后端改的是 IR，改完重新组装，
   * 所以这里只负责"发起 + 拉最新交付物 + 提示一句话"。
   * 详细的"改了哪几处、哪条没做到"从 `assembled.revisions` 里读（后端落的账）。
   */
  const revise = useCallback(
    async (instruction: string) => {
      const id = currentId.current;
      if (!id) return null;
      const text = instruction.trim();
      if (!text) return null;
      setRevising(true);
      setError("");
      try {
        const result = await api.revise(id, text);
        await reloadAssembled();
        setToast(
          result.ok
            ? result.record.note || `已按你的意见改了 ${result.changes.length} 处`
            : result.record.note || "这次没有能落地的改动",
        );
        return result;
      } catch (err) {
        report(err);
        return null;
      } finally {
        setRevising(false);
      }
    },
    [reloadAssembled, report],
  );

  const undoRevision = useCallback(async () => {
    const id = currentId.current;
    if (!id) return;
    try {
      await api.undoRevision(id);
      await reloadAssembled();
      setToast("已撤销这次修改，提示词回到上一版");
    } catch (err) {
      report(err);
    }
  }, [reloadAssembled, report]);

  const setForbiddenGroups = useCallback(
    async (groups: string[]) => {
      const id = currentId.current;
      if (!id) return;
      try {
        const result = await api.setForbiddenGroups(id, groups);
        await reloadAssembled();
        if (result.enforced.length) {
          setToast(
            `${result.enforced.join("、")} 是永不裁剪的底线，已自动保留`,
          );
        }
      } catch (err) {
        report(err);
      }
    },
    [reloadAssembled, report],
  );

  return {
    health,
    projects,
    project,
    assembled,
    answerQuestion,
    busy,
    analyzing,
    stage,
    progress,
    error,
    toast,
    draft,
    refreshProjects,
    selectProject,
    beginNewProject,
    createProject,
    removeProject,
    uploadVideo,
    startInfer,
    reloadProject,
    reloadAssembled,
    writeField,
    writeDialogue,
    bindAsset,
    updateDelivery,
    setForbiddenGroups,
    reviewEnabled,
    setReviewEnabled,
    revise,
    undoRevision,
    revising,
    dismissToast,
  };
}
