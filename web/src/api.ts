/**
 * 后端接口的唯一出口。
 *
 * 前端**不重复实现任何业务规则** —— 哪个字段能改、闸门怎么判，
 * 一律由后端回答。这里只做类型声明与请求封装。
 */

export type PositionColor = "blue" | "orange" | "gray";
export type PositionCategory = "LOCK" | "PARAM" | "FREE" | "AUTO";
export type RiskLevel = "low" | "medium" | "high" | "critical";
export type GateStatus = "ok" | "warn" | "block";

export interface Provenance {
  confidence: number;
  basis: string;
  disposition: "certain" | "accept" | "review" | "blank";
  span: { start_ms: number; end_ms: number } | null;
  note: string;
  evidence: string;
}

export interface Position {
  path: string;
  label: string;
  category: PositionCategory;
  color: PositionColor;
  kind: string;
  risk: RiskLevel;
  value: unknown;
  choices: string[];
  editable: boolean;
  apply_target: string;
  provenance: Provenance | null;
  note: string;
}

export interface ReferenceSlot {
  index: number;
  slots: string[];
  type: string;
  placeholder: string;
  asset_id: string | null;
  bound: boolean;
  seedance_role: string;
  prep_step: number | null;
  prep_hint: string;
}

export interface GateResult {
  code: string;
  name: string;
  status: GateStatus;
  detail: string;
  suggestion: string;
  autofixed: boolean;
  evidence: string[];
}

export interface GateReport {
  verdict: "ALL_GREEN" | "PASS_WITH_WARNINGS" | "BLOCKED";
  deliverable: boolean;
  all_green: boolean;
  blocker_count: number;
  warning_count: number;
  results: GateResult[];
}

export interface PromptSection {
  title: string;
  lines: string[];
  text: string;
  sources: string[];
}

export interface SegmentPlan {
  index: number;
  strategy: string;
  beat_ids: number[];
  duration_sec: number;
  note: string;
}

export interface CompressionEntry {
  section: string;
  what: string;
  reason: string;
  saved_chars: number;
}

export interface PromptDocument {
  strategy: string;
  param_header: string;
  char_count: number;
  sections: PromptSection[];
  plan: SegmentPlan[];
  compression: CompressionEntry[];
  warnings: string[];
  text: string;
}

export interface AssembleResponse {
  document: PromptDocument;
  gates: GateReport;
  compression: CompressionEntry[];
  split_plan: SegmentPlan[];
  positions: { counts: Record<PositionColor, number>; positions: Position[] };
  apply_targets: Record<string, string[]>;
}

export interface ProjectSummary {
  project_id: string;
  name: string;
  video_path: string;
  created_at: number;
  updated_at: number;
  framework_version: string;
  status: string;
  stage: string;
  progress: number;
  error: string;
  has_ir: boolean;
}

export interface ProjectDetail extends ProjectSummary {
  report: {
    video?: VideoInfo;
    shots?: ShotSignals[];
    degradations?: string[];
    conflicts?: Conflict[];
    model_id?: string;
    transcript?: TranscriptSegment[];
  };
  positions: { counts: Record<PositionColor, number>; positions: Position[] } | null;
  reference_manifest: ReferenceSlot[];
}

export interface VideoInfo {
  duration_sec: number;
  width: number;
  height: number;
  fps: number;
  has_audio: boolean;
  aspect_ratio: string;
}

export interface ShotSignals {
  boundary: { index: number; start_ms: number; end_ms: number; duration_sec: number };
  motion: { kind: string; confidence: number; note: string } | null;
  framing: { kind: string; confidence: number; note: string } | null;
}

export interface Conflict {
  field: string;
  shot_index: number;
  cv_value: string;
  model_value: string;
  resolution: string;
  confidence: number;
}

export interface TranscriptSegment {
  text: string;
  span: { start_ms: number; end_ms: number };
  speaker: string;
  confidence: number;
  has_timestamps: boolean;
}

export interface HealthInfo {
  ok: boolean;
  framework_version: string;
  phrase_lib_version: string;
  capabilities: {
    llm: { configured: boolean; base_url: string; vision_model: string; text_model: string; asr_model: string };
    media: { ffmpeg: string; ffprobe: string };
  };
  cv: { missing: string[]; to_dict?: unknown } & Record<string, unknown>;
}

/** 后端返回的业务错误：把 detail 原样带出来，不要吞掉原因。 */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload?.detail ?? detail;
    } catch {
      /* 响应不是 JSON，就用状态文本 */
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthInfo>("/api/health"),

  listProjects: () =>
    request<{ projects: ProjectSummary[] }>("/api/projects").then((r) => r.projects),

  createProject: (name: string) =>
    request<ProjectSummary>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  getProject: (id: string) => request<ProjectDetail>(`/api/projects/${id}`),

  deleteProject: (id: string) =>
    request<{ ok: boolean }>(`/api/projects/${id}`, { method: "DELETE" }),

  uploadVideo: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ video_path: string; bytes: number }>(`/api/projects/${id}/video`, {
      method: "POST",
      body: form,
    });
  },

  startInfer: (id: string) =>
    request<{ status: string }>(`/api/projects/${id}/infer`, { method: "POST" }),

  inferStatus: (id: string) =>
    request<{
      status: string;
      stage: string;
      progress: number;
      error: string;
      has_ir: boolean;
    }>(`/api/projects/${id}/status`),

  writeField: (id: string, path: string, value: unknown) =>
    request<{ ok: boolean; position: Position | null }>(`/api/projects/${id}/field`, {
      method: "PUT",
      body: JSON.stringify({ path, value }),
    }),

  writeDialogue: (
    id: string,
    payload: { shot_id: number; speaker: string; line: string; tone: string },
  ) =>
    request<{ ok: boolean; dialogue: unknown }>(`/api/projects/${id}/dialogue`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  bindAsset: (id: string, referenceIndex: number, assetId: string) =>
    request<{ ok: boolean; manifest: ReferenceSlot[] }>(`/api/projects/${id}/bind`, {
      method: "POST",
      body: JSON.stringify({ reference_index: referenceIndex, asset_id: assetId }),
    }),

  updateDelivery: (
    id: string,
    payload: Partial<{
      mode: string;
      strategy: string;
      target_model: string;
      first_frame_mode: string;
      include_param_header: boolean;
    }>,
  ) =>
    request<{ ok: boolean }>(`/api/projects/${id}/delivery`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  assemble: (id: string) => request<AssembleResponse>(`/api/projects/${id}/assemble`),

  exportUrl: (id: string, form: "plain" | "markdown" | "json") =>
    `/api/projects/${id}/export?form=${form}`,

  videoUrl: (id: string) => `/api/projects/${id}/video`,
};
