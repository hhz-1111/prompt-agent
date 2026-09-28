import { useEffect, useState } from "react";
import type { AlignmentBinding, AlignmentResponse, AlignmentState } from "./api";
import { api } from "./api";

interface Props {
  projectId: string;
  onApplied: () => void;
}

const PLACEHOLDER = `把你的台词粘进来，一行一句。三种写法都可以混用：

朋友都问我是不是做什么项目了
王姐：你这气色也太好了吧
00:03 哪有啊，我就是在家练普拉提`;

export function AlignmentPanel({ projectId, onApplied }: Props) {
  const [draft, setDraft] = useState("");
  const [state, setState] = useState<AlignmentState | null>(null);
  const [result, setResult] = useState<AlignmentResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void api
      .getAlignment(projectId)
      .then((data) => {
        setState(data);
        if (data.report) setResult(null);
      })
      .catch(() => setState(null));
  }, [projectId]);

  const report = result?.report ?? state?.report ?? null;

  const run = async () => {
    if (!draft.trim()) return;
    setBusy(true);
    setError("");
    try {
      const data = await api.align(projectId, draft);
      setResult(data);
      onApplied();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const manual = async (
    slotPath: string,
    payload: { block_index?: number | null; content?: string; speaker?: string },
  ) => {
    setBusy(true);
    try {
      await api.manualAlign(projectId, { slot_path: slotPath, ...payload });
      const fresh = await api.getAlignment(projectId);
      setState(fresh);
      setResult(null);
      onApplied();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel">
      <header className="panel__head">
        <h3>内容对齐</h3>
        {report && <SummaryChips summary={report.summary} />}
      </header>

      <p className="muted tiny panel__hint">
        粘贴你的台词稿，系统按「说话人 → 文字相似度 → 时长适配」三层判据
        自动分派到各个位置。<strong>把握不足的必须你确认</strong> ——
        不确定的猜测不会写进交付物。
      </p>

      <div className="align__input">
        <textarea
          value={draft}
          rows={5}
          placeholder={PLACEHOLDER}
          onChange={(e) => setDraft(e.target.value)}
        />
        <div className="align__actions">
          <button className="btn btn--primary" onClick={run} disabled={busy || !draft.trim()}>
            {busy ? "对齐中…" : "自动对齐"}
          </button>
          <span className="muted tiny">
            支持纯脚本 / <code>说话人：内容</code> / <code>00:03 内容</code>
          </span>
        </div>
      </div>

      {error && <div className="banner banner--error">{error}</div>}

      {report && report.notes.length > 0 && (
        <ul className="align__notes">
          {report.notes.map((note) => (
            <li key={note}>· {note}</li>
          ))}
        </ul>
      )}

      {report && report.bindings.length > 0 ? (
        <div className="align__table">
          <div className="align__row align__row--head">
            <span>原片时间码</span>
            <span>原片台词</span>
            <span>我的内容</span>
            <span>置信度</span>
            <span>操作</span>
          </div>
          {report.bindings.map((binding) => (
            <BindingRow key={binding.slot.path} binding={binding} onManual={manual} busy={busy} />
          ))}
        </div>
      ) : (
        <p className="muted pad">
          {state?.slots.length
            ? `反推出 ${state.slots.length} 个可放置台词的位置，粘贴内容后开始对齐。`
            : "还没有反推结果。"}
        </p>
      )}

      {report && report.orphans.length > 0 && (
        <details className="align__orphans">
          <summary>有 {report.orphans.length} 段内容没找到位置</summary>
          <ul>
            {report.orphans.map((block) => (
              <li key={block.index}>
                {block.content}
                <span className="muted tiny">（约 {block.estimated_seconds} 秒）</span>
              </li>
            ))}
          </ul>
          <p className="muted tiny">位置不够用。合并内容，或延长目标时长后重新拆条。</p>
        </details>
      )}
    </section>
  );
}

function SummaryChips({ summary }: { summary: AlignmentResponse["report"]["summary"] }) {
  return (
    <div className="panel__filters">
      <span className="chip chip--ready">已确认 {summary.bound}</span>
      {summary.pending > 0 && <span className="chip chip--analyzing">待确认 {summary.pending}</span>}
      {summary.unmatched > 0 && <span className="chip">保留原片 {summary.unmatched}</span>}
      {summary.orphans > 0 && <span className="chip chip--failed">无位置 {summary.orphans}</span>}
    </div>
  );
}

function BindingRow({
  binding,
  onManual,
  busy,
}: {
  binding: AlignmentBinding;
  onManual: (path: string, payload: { block_index?: number | null; content?: string }) => void;
  busy: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(binding.block?.content ?? "");
  const block = binding.block;

  const tone =
    binding.status === "bound"
      ? "ok"
      : binding.status === "pending"
        ? "warn"
        : "muted";

  return (
    <div className={`align__row align__row--${tone}`}>
      <span className="mono tiny">{binding.slot.timecode}</span>
      <span className="align__ref">
        {binding.slot.reference_line || <em className="muted">（原片无台词）</em>}
        {binding.slot.speaker && <span className="muted tiny"> — {binding.slot.speaker}</span>}
      </span>
      <span className="align__mine">
        {editing ? (
          <textarea
            autoFocus
            rows={2}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => {
              setEditing(false);
              if (draft !== (block?.content ?? "")) {
                onManual(binding.slot.path, { content: draft });
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                (e.target as HTMLTextAreaElement).blur();
              }
              if (e.key === "Escape") {
                setDraft(block?.content ?? "");
                setEditing(false);
              }
            }}
          />
        ) : (
          <button className="align__edit" onClick={() => setEditing(true)}>
            {block ? (
              block.content
            ) : (
              <em className="muted">（将保留原片台词）</em>
            )}
          </button>
        )}
      </span>
      <span className="align__conf">
        {block ? (
          <span className={`conf conf--${binding.confidence >= 0.85 ? "certain" : binding.confidence >= 0.7 ? "accept" : "review"}`}>
            {binding.confidence.toFixed(2)}
          </span>
        ) : (
          <span className="muted tiny">—</span>
        )}
        {binding.requires_confirmation && <span className="chip chip--analyzing">待确认</span>}
      </span>
      <span className="align__ops">
        {block && block.index >= 0 && (
          <button
            className="icon-btn"
            title="解绑，该位置保留原片台词"
            disabled={busy}
            onClick={() => onManual(binding.slot.path, { block_index: null })}
          >
            解绑
          </button>
        )}
      </span>
      {binding.conflicts.length > 0 && (
        <p className="align__conflict">⚠️ {binding.conflicts.join("；")}</p>
      )}
      {block && binding.reason && (
        <p className="align__reason muted tiny">{binding.reason}</p>
      )}
    </div>
  );
}
