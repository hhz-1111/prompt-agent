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
} from "./api";

const POLL_INTERVAL_MS = 1200;

export interface AppState {
  health: HealthInfo | null;
  projects: ProjectSummary[];
  project: ProjectDetail | null;
  assembled: AssembleResponse | null;
  busy: boolean;
  analyzing: boolean;
  stage: string;
  progress: number;
  error: string;
  toast: string;

  refreshProjects: () => Promise<void>;
  selectProject: (id: string) => Promise<void>;
  createProject: (name: string) => Promise<string | null>;
  removeProject: (id: string) => Promise<void>;
  uploadVideo: (id: string, file: File) => Promise<void>;
  startInfer: (id: string) => Promise<void>;
  reloadProject: () => Promise<void>;
  reloadAssembled: () => Promise<void>;
  writeField: (path: string, value: unknown) => Promise<void>;
  writeDialogue: (shotId: number, payload: { speaker: string; line: string; tone: string }) => Promise<void>;
  bindAsset: (index: number, assetId: string) => Promise<void>;
  updateDelivery: (payload: Record<string, unknown>) => Promise<void>;
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

  const selectProject = useCallback(
    async (id: string) => {
      currentId.current = id;
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
        if (currentId.current === id) {
          currentId.current = null;
          setProject(null);
          setAssembled(null);
        }
        await refreshProjects();
      } catch (err) {
        report(err);
      } finally {
        setBusy(false);
      }
    },
    [refreshProjects, report],
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
    async (id: string) => {
      setError("");
      try {
        await api.startInfer(id);
        setAnalyzing(true);
        setStage("排队中");
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

  return {
    health,
    projects,
    project,
    assembled,
    busy,
    analyzing,
    stage,
    progress,
    error,
    toast,
    refreshProjects,
    selectProject,
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
    dismissToast,
  };
}
