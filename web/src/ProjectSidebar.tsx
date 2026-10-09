/**
 * 左侧项目栏：**新建项目 + 历史项目列表**。
 *
 * 用户 2026-09-29 第三轮的两个要求：
 *   ① 历史项目从右上角抽屉**搬到左边**，常驻可见；
 *   ② 加一个「新建项目」—— 原来上传页的显示条件是"当前项目没有视频"，
 *      而项目一旦传过视频就永远有视频，**唯一的回上传页办法是把项目删掉**。
 *
 * 所以这一栏给的就是"换一条视频重新开始"的入口：点「＋ 新建项目」
 * 回到上传页（不建空项目，选到文件才建，避免堆一堆没有视频的空壳）。
 */

import type { ProjectSummary } from "./api";

export function ProjectSidebar({
  projects,
  activeId,
  draft,
  busy,
  onNew,
  onSelect,
  onRemove,
}: {
  projects: ProjectSummary[];
  activeId: string | null;
  /** 正在"新建项目"（还没选文件）—— 高亮新建按钮，列表不高亮任何一项。 */
  draft: boolean;
  busy: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  return (
    <aside className="side">
      <div className="side__brand">
        <strong>爆款视频复刻</strong>
        {/* 手动断行：侧栏窄，交给浏览器折行会在末尾甩出一个孤字 */}
        <span className="muted tiny">
          上传一条爆款视频
          <br />
          拿到能直接用的提示词
        </span>
      </div>

      <button
        className={`btn side__new${draft ? " is-active" : ""}`}
        onClick={onNew}
        disabled={busy}
      >
        ＋ 新建项目
      </button>

      <div className="side__label muted tiny">
        历史项目{projects.length > 0 ? `（${projects.length}）` : ""}
      </div>

      <div className="side__list">
        {projects.length === 0 && (
          <p className="muted tiny side__empty">还没有项目，点上面的「新建项目」开始</p>
        )}
        {projects.map((item) => (
          <div key={item.project_id} className="side__row">
            <button
              className={`side__item${
                !draft && item.project_id === activeId ? " is-active" : ""
              }`}
              title={item.name || item.project_id}
              onClick={() => onSelect(item.project_id)}
            >
              <span className="side__name">{item.name || item.project_id}</span>
              <span className="muted tiny">{statusText(item)}</span>
            </button>
            <button
              className="btn btn--tiny side__del"
              title="删除这个项目"
              onClick={() => onRemove(item.project_id)}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}

/** 一句话说清这个项目到哪一步了 —— 用户看不懂 `ready` / `phase1` 这类内部词。 */
function statusText(item: ProjectSummary): string {
  if (item.status === "ready") return "已完成";
  if (item.status === "failed") return "反推失败";
  if (item.status === "analyzing") return item.stage || "反推中";
  return item.video_path ? "待反推" : "还没传视频";
}
