import { useState } from "react";
import type { ProjectSummary } from "./api";

interface Props {
  projects: ProjectSummary[];
  activeId: string | null;
  busy: boolean;
  onSelect: (id: string) => void;
  onCreate: (name: string) => void;
  onDelete: (id: string) => void;
}

export function ProjectRail({ projects, activeId, busy, onSelect, onCreate, onDelete }: Props) {
  const [name, setName] = useState("");

  const submit = () => {
    const trimmed = name.trim() || `复刻任务 ${new Date().toLocaleString("zh-CN", { hour12: false })}`;
    onCreate(trimmed);
    setName("");
  };

  return (
    <aside className="rail">
      <header className="rail__head">
        <h1>爆款复刻</h1>
        <p className="muted">提示词专家 Agent</p>
      </header>

      <div className="rail__new">
        <input
          value={name}
          placeholder="新建项目名（可留空）"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
        <button onClick={submit} disabled={busy}>
          新建
        </button>
      </div>

      <ul className="rail__list">
        {projects.length === 0 && <li className="muted pad">还没有项目</li>}
        {projects.map((project) => (
          <li
            key={project.project_id}
            className={project.project_id === activeId ? "rail__item is-active" : "rail__item"}
          >
            <button className="rail__pick" onClick={() => onSelect(project.project_id)}>
              <span className="rail__name">{project.name}</span>
              <span className={`chip chip--${project.status}`}>{statusLabel(project.status)}</span>
            </button>
            <button
              className="icon-btn"
              title="删除项目"
              onClick={() => onDelete(project.project_id)}
              disabled={busy}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}

function statusLabel(status: string): string {
  const map: Record<string, string> = {
    created: "待上传",
    analyzing: "分析中",
    ready: "已反推",
    failed: "失败",
  };
  return map[status] ?? status;
}
