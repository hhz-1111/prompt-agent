import { useState } from "react";
import type { AssembleResponse, GateReport, ProjectDetail } from "./api";

interface Props {
  assembled: AssembleResponse;
  project: ProjectDetail;
  onDelivery: (payload: Record<string, unknown>) => void;
}

type OutputTab = "plain" | "markdown" | "json";

export function DeliveryPanel({ assembled, project, onDelivery }: Props) {
  const [tab, setTab] = useState<OutputTab>("plain");
  const [copied, setCopied] = useState(false);
  const document = assembled.document;
  const gates = assembled.gates;

  const copy = async () => {
    await navigator.clipboard.writeText(document.text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <section className="panel">
      <header className="panel__head">
        <h3>交付</h3>
        <div className="panel__filters">
          <GateBadge report={gates} />
        </div>
      </header>

      <div className="delivery__meta">
        <span className="mono">{document.char_count} 字</span>
        <span className="mono">拆条 {document.strategy} · {document.plan.length} 条</span>
        <label className="switch">
          <input
            type="checkbox"
            checked={document.param_header.length > 0}
            onChange={(e) =>
              onDelivery({
                mode: e.target.checked ? "phase1_manual" : "phase2_api",
                include_param_header: e.target.checked,
              })
            }
          />
          <span>写参数声明（手动复制模式）</span>
        </label>
      </div>

      <p className="muted tiny">
        同一份提示词，手动模式保留开头的参数声明；直调 API 模式移除它，
        正文内容完全一致（PRD §6.4.5）。
      </p>

      {document.compression.length > 0 && (
        <details className="compress">
          <summary>
            字数超预算时的压缩优先级（{document.compression.length} 条）
          </summary>
          <ul>
            {document.compression.map((entry) => (
              <li key={`${entry.section}-${entry.what}`}>
                <strong>{entry.section}</strong> · {entry.what}
                <span className="muted tiny">（约省 {entry.saved_chars} 字：{entry.reason}）</span>
              </li>
            ))}
          </ul>
          <p className="muted tiny">
            永不可裁：站位、音色、台词原文、D0 通用禁止项、D7 对白锁定。
          </p>
        </details>
      )}

      <div className="delivery__tabs">
        {(["plain", "markdown", "json"] as OutputTab[]).map((item) => (
          <button
            key={item}
            className={tab === item ? "tab is-active" : "tab"}
            onClick={() => setTab(item)}
          >
            {item === "plain" ? "纯文本" : item === "markdown" ? "Markdown" : "JSON"}
          </button>
        ))}
        <a className="btn btn--ghost btn--sm" href={`/api/projects/${project.project_id}/export?form=${tab}`} download>
          下载
        </a>
        <button className="btn btn--primary btn--sm" onClick={copy}>
          {copied ? "已复制" : "复制提示词"}
        </button>
      </div>

      <pre className="prompt">{tab === "json" ? JSON.stringify(assembled.document, null, 2) : document.text}</pre>

      <GateList report={gates} />
    </section>
  );
}

function GateBadge({ report }: { report: GateReport }) {
  const tone = report.all_green ? "ok" : report.deliverable ? "warn" : "block";
  const text = report.all_green
    ? "闸门全绿 · 允许交付"
    : report.deliverable
      ? `可交付（${report.warning_count} 项告警）`
      : `阻断交付（${report.blocker_count} 项）`;
  return <span className={`gate-badge gate-badge--${tone}`}>{text}</span>;
}

function GateList({ report }: { report: GateReport }) {
  return (
    <div className="gates">
      <h4>交付自检（14 项闸门）</h4>
      <ul>
        {report.results.map((item) => (
          <li key={item.code} className={`gate gate--${item.status}`}>
            <span className="gate__code mono">{item.code}</span>
            <span className="gate__name">{item.name}</span>
            <span className={`gate__mark gate__mark--${item.status}`}>
              {item.status === "ok" ? "✅" : item.status === "warn" ? "⚠️" : "❌"}
            </span>
            <span className="gate__detail">{item.detail}</span>
            {item.autofixed && <span className="chip chip--auto">已自动修复</span>}
            {item.suggestion && <p className="gate__fix muted tiny">→ {item.suggestion}</p>}
            {item.evidence.length > 0 && (
              <p className="gate__evidence mono tiny muted">{item.evidence.slice(0, 3).join(" ／ ")}</p>
            )}
          </li>
        ))}
      </ul>
      <p className={`gates__verdict gates__verdict--${report.deliverable ? "ok" : "block"}`}>
        → {report.deliverable ? "允许交付" : "阻断交付"}
      </p>
    </div>
  );
}
