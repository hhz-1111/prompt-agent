import { useState } from "react";
import type { AssembleResponse, ProjectDetail, ReferenceSlot } from "./api";
import { colorLegend } from "./format";

interface Props {
  project: ProjectDetail;
  assembled: AssembleResponse | null;
  analyzing: boolean;
  stage: string;
  progress: number;
  busy: boolean;
  videoUrl: string;
  onUpload: (file: File) => void;
  onInfer: () => void;
  onBind: (index: number, assetId: string, slot: ReferenceSlot) => void;
}

export function Workbench({
  project,
  assembled,
  analyzing,
  stage,
  progress,
  busy,
  videoUrl,
  onUpload,
  onInfer,
  onBind,
}: Props) {
  const report = project.report ?? {};
  const video = report.video;
  const shots = report.shots ?? [];
  const degradations = report.degradations ?? [];
  const conflicts = report.conflicts ?? [];
  const counts = assembled?.positions.counts;

  return (
    <section className="workbench">
      <header className="workbench__head">
        <div>
          <h2>{project.name}</h2>
          <p className="muted mono">
            {video
              ? `${video.duration_sec.toFixed(2)}s · ${video.width}×${video.height} · ${video.fps}fps · ${video.aspect_ratio}`
              : "尚未解析视频"}
          </p>
        </div>
        <div className="workbench__actions">
          <label className={busy ? "btn btn--ghost is-disabled" : "btn btn--ghost"}>
            上传视频
            <input
              type="file"
              accept="video/mp4,video/quicktime"
              hidden
              disabled={busy}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onUpload(file);
                e.target.value = "";
              }}
            />
          </label>
          <button
            className="btn btn--primary"
            onClick={onInfer}
            disabled={busy || analyzing || !project.video_path}
          >
            {analyzing ? "反推中…" : project.has_ir ? "重新反推" : "开始反推"}
          </button>
        </div>
      </header>

      {analyzing && (
        <div className="progress">
          <div className="progress__bar" style={{ width: `${Math.round(progress * 100)}%` }} />
          <span className="progress__label">
            {stage || "处理中"} {Math.round(progress * 100)}%
          </span>
        </div>
      )}

      <div className="workbench__grid">
        <div className="card">
          <h3>原片与镜头</h3>
          {project.video_path ? (
            <video className="player" src={videoUrl} controls preload="metadata" />
          ) : (
            <p className="muted">先上传一条爆款视频</p>
          )}
          {shots.length > 0 && (
            <ol className="shots">
              {shots.map((shot) => (
                <li key={shot.boundary.index}>
                  <span className="shots__id">镜头{shot.boundary.index}</span>
                  <span className="mono muted">
                    {(shot.boundary.start_ms / 1000).toFixed(1)}–
                    {(shot.boundary.end_ms / 1000).toFixed(1)}s
                  </span>
                  <span className="shots__tags">
                    {shot.framing && <em>{shot.framing.kind}</em>}
                    {shot.motion && <em>{motionLabel(shot.motion.kind)}</em>}
                  </span>
                </li>
              ))}
            </ol>
          )}
        </div>

        <div className="card">
          <h3>素材清单（🔵 蓝位）</h3>
          {project.reference_manifest.length === 0 && (
            <p className="muted">反推后这里会列出需要上游准备的素材</p>
          )}
          <ul className="manifest">
            {project.reference_manifest.map((slot) => (
              <ManifestRow key={slot.index} slot={slot} onBind={onBind} />
            ))}
          </ul>
          <p className="muted tiny">
            素材的生成与合规由中台负责，Agent 只负责「指名」与「正确引用」。
          </p>
        </div>

        <div className="card">
          <h3>三色位置</h3>
          {counts ? (
            <div className="legend">
              {colorLegend(counts).map((item) => (
                <div key={item.color} className={`legend__row legend__row--${item.color}`}>
                  <span className="legend__dot" />
                  <span className="legend__label">{item.label}</span>
                  <span className="legend__count mono">{item.count}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">反推完成后生成位置标注</p>
          )}
        </div>

        <div className="card">
          <h3>诊断</h3>
          {report.model_id && (
            <p className="muted tiny mono">{report.model_id}</p>
          )}
          {conflicts.length > 0 && (
            <div className="diag diag--warn">
              <strong>CV 与模型判断冲突 {conflicts.length} 处</strong>
              <ul>
                {conflicts.slice(0, 5).map((c) => (
                  <li key={`${c.field}-${c.shot_index}`}>
                    镜头{c.shot_index} · {c.field}：CV 判「{c.cv_value}」，模型判「
                    {c.model_value}」→ {c.resolution}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {degradations.length > 0 && (
            <div className="diag diag--info">
              <strong>降级说明（如实告知，不藏）</strong>
              <ul>
                {degradations.map((d) => (
                  <li key={d}>{d}</li>
                ))}
              </ul>
            </div>
          )}
          {conflicts.length === 0 && degradations.length === 0 && (
            <p className="muted">无冲突、无降级</p>
          )}
        </div>
      </div>
    </section>
  );
}

function ManifestRow({
  slot,
  onBind,
}: {
  slot: ReferenceSlot;
  onBind: (index: number, assetId: string, slot: ReferenceSlot) => void;
}) {
  const [value, setValue] = useState(slot.asset_id ?? "");
  return (
    <li className={slot.bound ? "manifest__row is-bound" : "manifest__row"}>
      <div className="manifest__main">
        <span className="manifest__slots">{slot.slots.join("、") || slot.placeholder}</span>
        <span className="chip">{slot.type}</span>
        {slot.prep_step !== null && <span className="muted tiny">工序 #{slot.prep_step}</span>}
      </div>
      {slot.prep_hint && <p className="muted tiny">{slot.prep_hint}</p>}
      <div className="manifest__bind">
        <input
          value={value}
          placeholder="粘贴上游回填的素材句柄"
          onChange={(e) => setValue(e.target.value)}
        />
        <button
          className="btn btn--ghost btn--sm"
          onClick={() => onBind(slot.index, value.trim(), slot)}
          disabled={!value.trim()}
        >
          绑定
        </button>
      </div>
    </li>
  );
}

function motionLabel(kind: string): string {
  const map: Record<string, string> = {
    fixed: "固定",
    push_in: "推近",
    pan_or_track: "横摇/跟拍",
    handheld: "手持",
    unknown: "未知",
  };
  return map[kind] ?? kind;
}
