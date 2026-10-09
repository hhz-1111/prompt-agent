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
  hints: string[];
  /** 给用户看的操作提示（如「↳ 可替换：[图1] = 王姐多视图」）。
   *  **不进正文** —— 后端已经把它们与 lines 分开，导出时也不会带上。 */
}

export interface ForbiddenGroup {
  label: string;
  default_on: boolean;
  note: string;
  items: string[];
}

export interface ForbiddenGroupState {
  groups: string[];
  enforced: string[];
  unknown: string[];
  item_count: number;
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

export interface ReviewItem {
  kind: "gate" | "degradation" | "conflict" | "confidence";
  level: "block" | "warn" | "review";
  label: string;
  detail: string;
  suggestion: string;
}

/** 本次反推里"需要你知道/确认"的东西。
 *
 *  以前置信度、降级说明、CV↔模型冲突、闸门判定**一个出口都没有** ——
 *  结果页只有一个文本框，只能靠肉眼抓错。 */
export interface ReviewSummary {
  count: number;
  items: ReviewItem[];
  truncated: number;
}

export interface AssembleResponse {
  document: PromptDocument;
  gates: GateReport;
  review: ReviewSummary;
  /** Agent 交给人裁决的选择题（最多 2 条）。点选后答案**直接写进提示词**。 */
  questions: AgentQuestion[];
  /** 用户意见改写的记账（最近若干轮）：刷新页面后仍能看到改过什么。 */
  revisions: RevisionRecord[];
  /** 还有没有可以撤销的改写（后端持有上一版 IR 快照）。 */
  can_undo: boolean;
  compression: CompressionEntry[];
  split_plan: SegmentPlan[];
  positions: { counts: Record<PositionColor, number>; positions: Position[] };
  apply_targets: Record<string, string[]>;
}

/** Agent 提出的选择题。`options` 里的值就是能直接写进字段的值。 */
export interface AgentQuestion {
  id: string;
  path: string;
  text: string;
  options: string[];
  reason: string;
}

/** 用户意见改写：一处已生效的改动（给人看的原句 → 新句）。 */
export interface RevisionChange {
  path: string;
  label: string;
  before: string;
  after: string;
  reason: string;
}

/** 一条**做不到**的意见，以及为什么。必须显示出来 —— 不能让用户以为改好了。 */
export interface RevisionRefusal {
  instruction: string;
  reason: string;
}

export interface RevisionNote {
  path: string;
  action: string;
  reason: string;
  applied: boolean;
  detail: string;
}

/** 一轮改写里实际发生的模型调用（用途/型号/耗时/token）。 */
export interface ModelCall {
  purpose: string;
  model: string;
  seconds: number;
  ok: boolean;
  detail: string;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number;
}

/** 一轮改写的账：说了什么、改了哪几处、哪条没做到、调了什么模型。 */
export interface RevisionRecord {
  id: string;
  instruction: string;
  note: string;
  model: string;
  created_at: number;
  seconds: number;
  changes: RevisionChange[];
  notes: RevisionNote[];
  rejected: RevisionNote[];
  refusals: RevisionRefusal[];
  verdict_before: string;
  verdict_after: string;
  ok: boolean;
  /** 这一轮到底调了哪些模型。用户要求"只让 DeepSeek 改、不要再送 MiMo 自检"，
   *  这份账就是证据：正常恰好一条「按意见改写 · deepseek-flash」。 */
  model_calls: ModelCall[];
}

export interface ReviseResponse {
  ok: boolean;
  record: RevisionRecord;
  changes: RevisionChange[];
  refusals: RevisionRefusal[];
  rejected: RevisionNote[];
  notes: RevisionNote[];
  verdict_before: string;
  verdict_after: string;
  can_undo: boolean;
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
    llm: {
      configured: boolean;
      base_url: string;
      vision_model: string;
      text_model: string;
      asr_model: string;
      /** 下面几项是改写（PRD F8.4）用的：端点可换，界面上要如实显示是谁改的。 */
      review_model?: string;
      revise_model?: string;
      revise_base_url?: string;
      revise_configured?: boolean;
    };
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

export interface AlignmentScore {
  total: number;
  similarity: number;
  duration_fit: number;
  speaker_match: number;
}

export interface ContentBlock {
  index: number;
  content: string;
  speaker: string;
  char_start: number;
  char_end: number;
  estimated_seconds: number;
  timecode_hint_sec: number | null;
}

export interface AlignmentSlot {
  index: number;
  path: string;
  label: string;
  timecode: string;
  span: { start_ms: number; end_ms: number };
  reference_line: string;
  speaker: string;
  capacity_chars: number;
}

export interface AlignmentBinding {
  slot: AlignmentSlot;
  block: ContentBlock | null;
  score: AlignmentScore;
  confidence: number;
  status: "bound" | "pending" | "unmatched" | "orphan";
  requires_confirmation: boolean;
  reason: string;
  conflicts: string[];
}

export interface AlignmentSummary {
  total_slots: number;
  bound: number;
  pending: number;
  unmatched: number;
  orphans: number;
  has_unresolved: boolean;
}

export interface AlignmentReport {
  bindings: AlignmentBinding[];
  orphans: ContentBlock[];
  chars_per_sec: number;
  notes: string[];
  summary: AlignmentSummary;
}

export interface AlignmentResponse {
  report: AlignmentReport;
  written: string[];
  blocks: ContentBlock[];
  slots: AlignmentSlot[];
}

export interface AlignmentState {
  report: AlignmentReport | null;
  slots: AlignmentSlot[];
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

  /** ``review=false`` = 快速反推：跳过"交付前自检"（实测那一段要几分钟）。 */
  startInfer: (id: string, review?: boolean) =>
    request<{ status: string }>(
      `/api/projects/${id}/infer${review === undefined ? "" : `?review=${review ? 1 : 0}`}`,
      { method: "POST" },
    ),

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

  /**
   * 按用户的一句修改意见改写提示词（PRD F8.4）。
   *
   * 同步接口：后端改完才返回（实测 30~90 秒，个别情况更久），
   * 所以界面上要显示"已等多少秒"，别让用户以为卡死了。
   */
  revise: (id: string, instruction: string) =>
    request<ReviseResponse>(`/api/projects/${id}/revise`, {
      method: "POST",
      body: JSON.stringify({ instruction }),
    }),

  /** 撤销最近一轮改写：后端把上一份 IR 放回去。 */
  undoRevision: (id: string) =>
    request<{ ok: boolean; can_undo: boolean }>(`/api/projects/${id}/revise/undo`, {
      method: "POST",
    }),

  /** 回答 Agent 的选择题：答案会写进 IR，之后再调 assemble 就能看到更新后的正文。 */
  answerQuestion: (id: string, questionId: string, choice: string) =>
    request<{ path: string; value: string }>(`/api/projects/${id}/answer`, {
      method: "POST",
      body: JSON.stringify({ id: questionId, choice }),
    }),

  /** 固定提示词表格页：按勾选重建禁止项清单（D0/D7 由后端强制保留）。 */
  setForbiddenGroups: (id: string, groups: string[]) =>
    request<ForbiddenGroupState>(`/api/projects/${id}/forbidden-groups`, {
      method: "PUT",
      body: JSON.stringify({ groups }),
    }),

  phraseLibrary: () =>
    request<{ forbidden_groups: Record<string, ForbiddenGroup> }>("/api/phrases"),

  exportUrl: (id: string, form: "plain" | "markdown" | "json") =>
    `/api/projects/${id}/export?form=${form}`,

  videoUrl: (id: string) => `/api/projects/${id}/video`,

  // ---- 内容对齐（PRD F6）----
  getAlignment: (id: string) => request<AlignmentState>(`/api/projects/${id}/align`),

  align: (id: string, content: string, apply = true, charsPerSec = 5) =>
    request<AlignmentResponse>(`/api/projects/${id}/align`, {
      method: "POST",
      body: JSON.stringify({ content, apply, chars_per_sec: charsPerSec }),
    }),

  manualAlign: (
    id: string,
    payload: { slot_path: string; block_index?: number | null; content?: string; speaker?: string },
  ) =>
    request<{ ok: boolean; line: string; speaker: string }>(
      `/api/projects/${id}/align/manual`,
      { method: "PUT", body: JSON.stringify(payload) },
    ),
};
