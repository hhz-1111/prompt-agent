/**
 * 「修改提示词」对话框：**一个标题 + 一个输入框 + 一个提交按钮**，没有别的东西。
 *
 * 用户 2026-10-08 定稿的形态：
 *   - 不要常用意见 chips；
 *   - 不要那么多提示文字，**只保留标题**（标题「AI改写」，下面一行灰色小字
 *     说明"改写结果来自谁"）；
 *   - 蓝色按钮叫「提交」，结果页那个按钮叫「修改提示词」。
 *
 * 三件必须做到的事（都是这个项目踩过的教训）：
 *
 * 1. **要有等待感**：改写是同步的（实测 10~30 秒），
 *    等待时显示秒数，不给秒数用户会以为死机（第十八轮"自检与重写卡很久"）；
 * 2. **没改成要说清为什么**：模型返回空数组 / 意见落不了地时，
 *    绝不能显示成"已完成"（第十九轮的教训：空响应 ≠ 没问题）；
 * 3. **能撤销**：AI 改完必须能一键回到上一版，否则用户不敢用。
 */

import { useEffect, useRef, useState } from "react";
import type { HealthInfo, RevisionRecord } from "./api";

/**
 * 「改写结果来自谁」——按 `.env` 里配置的端点推，**不写死 Deepseek**。
 *
 * 用户明确要求"所有 key 都要可换端点"，所以哪天把 `REVISE_BASE_URL` 换成别家，
 * 这里必须跟着变；写死就等于在界面上说谎。
 */
export function reviseSourceLabel(health: HealthInfo | null): string {
  const llm = (health?.capabilities?.llm ?? {}) as Record<string, unknown>;
  const base = String(llm.revise_base_url || llm.base_url || "");
  const host = base.split("//").pop()?.split("/")[0] || "";
  if (host.includes("deepseek")) return "Deepseek";
  if (host.includes("xiaomimimo") || host.includes("mimo")) return "MiMo";
  return host || "配置的模型";
}

export function ReviseDialog({
  open,
  busy,
  revisions,
  canUndo,
  sourceLabel = "配置的模型",
  onClose,
  onSubmit,
  onUndo,
}: {
  open: boolean;
  busy: boolean;
  revisions: RevisionRecord[];
  canUndo: boolean;
  /** 改写结果来自哪家（由 :func:`reviseSourceLabel` 按端点推出来）。 */
  sourceLabel?: string;
  onClose: () => void;
  onSubmit: (instruction: string) => void;
  onUndo: () => void;
}) {
  const [text, setText] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const areaRef = useRef<HTMLTextAreaElement>(null);

  // 打开时清空并聚焦；关闭时把秒数归零
  useEffect(() => {
    if (open) {
      setText("");
      window.setTimeout(() => areaRef.current?.focus(), 0);
    } else {
      setElapsed(0);
    }
  }, [open]);

  // 等待秒数：让"正在改"和"卡死了"能区分开
  useEffect(() => {
    if (!busy) {
      setElapsed(0);
      return;
    }
    const timer = window.setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [busy]);

  if (!open) return null;

  const last = revisions.length ? revisions[revisions.length - 1] : null;

  return (
    <div className="rev" role="dialog" aria-modal="true">
      <div className="rev__box">
        <div className="rev__head">
          <div className="rev__titles">
            <strong className="rev__title">AI改写</strong>
            {/* 灰色小字：说清这一版是谁改的 */}
            <span className="rev__from muted tiny">改写结果来自{sourceLabel}</span>
          </div>
          <button className="btn btn--tiny" onClick={onClose} disabled={busy}>
            ✕
          </button>
        </div>

        <textarea
          ref={areaRef}
          className="rev__input"
          value={text}
          disabled={busy}
          spellCheck={false}
          onChange={(event) => setText(event.target.value)}
        />

        <div className="rev__bar">
          {/* 平时不留任何说明文字；只有等待时才显示秒数 —— 不给秒数用户会以为死机 */}
          <span className="muted tiny">
            {busy ? `已等 ${elapsed} 秒` : ""}
          </span>
          <button
            className="btn btn--primary"
            disabled={busy || !text.trim()}
            onClick={() => onSubmit(text)}
          >
            {busy ? "提交中…" : "提交"}
          </button>
        </div>

        {last && <RevisionReport record={last} canUndo={canUndo} onUndo={onUndo} />}
      </div>
    </div>
  );
}

/** 一轮改写的账单：改了哪几处、哪条没做到、代价是什么。 */
function RevisionReport({
  record,
  canUndo,
  onUndo,
}: {
  record: RevisionRecord;
  canUndo: boolean;
  onUndo: () => void;
}) {
  const failed = record.rejected.filter((item) => !item.applied);
  const changed = record.verdict_before && record.verdict_after &&
    record.verdict_before !== record.verdict_after;

  return (
    <div className="rev__report">
      <div className={`rev__result ${record.ok ? "rev__result--ok" : "rev__result--no"}`}>
        {record.ok ? "✅ " : "⚠️ "}
        {record.ok
          ? `改了 ${record.changes.length} 处`
          : "这次没能落地的改动"}
        {record.note && <span className="rev__note">{record.note}</span>}
        {record.seconds > 0 && (
          <span className="muted tiny">（用了 {Math.round(record.seconds)} 秒）</span>
        )}
      </div>

      {record.changes.length > 0 && (
        <ul className="rev__list">
          {record.changes.map((change, index) => (
            <li key={`${change.path}-${index}`} className="rev__item">
              <span className="rev__label">{change.label || change.path}</span>
              <span className="rev__diff">
                <span className="rev__before">{change.before}</span>
                <span className="rev__arrow"> → </span>
                <span className="rev__after">{change.after}</span>
              </span>
              {change.reason && <span className="muted tiny">（{change.reason}）</span>}
            </li>
          ))}
        </ul>
      )}

      {record.refusals.length > 0 && (
        <div className="rev__refusals">
          <div className="rev__sub">这几条我做不到，原因是：</div>
          <ul className="rev__list">
            {record.refusals.map((item, index) => (
              <li key={`refusal-${index}`} className="rev__item rev__item--no">
                <span className="rev__before">{item.instruction}</span>
                <span className="rev__why">{item.reason || "改不了"}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {failed.length > 0 && (
        <div className="rev__refusals">
          <div className="rev__sub">这几处没动：</div>
          <ul className="rev__list">
            {failed.map((item, index) => (
              <li key={`rejected-${index}`} className="rev__item rev__item--no">
                <span className="rev__before">{item.path || "（未指明位置）"}</span>
                <span className="rev__why">{item.detail || item.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {record.notes.map((item, index) => (
        <p key={`note-${index}`} className="rev__note-item muted tiny">
          {item.detail}
        </p>
      ))}

      {record.model_calls.length > 0 && (
        <p className="rev__note-item muted tiny">
          模型调用 {record.model_calls.length} 次：
          {record.model_calls
            .map((call) => `${call.purpose || "改写"} · ${call.model}（${Math.round(call.seconds)} 秒）`)
            .join("；")}
        </p>
      )}

      {changed && (
        <p className="rev__note-item muted tiny">
          交付自检：{record.verdict_before} → {record.verdict_after}
        </p>
      )}

      {canUndo && (
        <button className="btn btn--ghost rev__undo" onClick={onUndo}>
          撤销最近一次改动（回到上一版提示词）
        </button>
      )}
    </div>
  );
}
