/**
 * 交付提示词：**一个可编辑的大文本框**，按产物顺序一行一行排列。
 *
 * 用户 2026-09-29 明确要求：
 *   - 不要颜色区分（原来灰色的"固定"、黄色的"可改"、蓝色的"@素材"全部去掉）；
 *   - 不要字数那一行；
 *   - 就是干净的提示词正文，一行一行读下来。
 *
 * 所以这里刻意做得很素：段落标题 + 正文行，外加**一条可折叠的"待确认"摘要**。
 * 摘要只在真有问题时出现 —— 反推的把握（置信度）、降级说明、CV 与模型的分歧、
 * 闸门判定以前**全都没有出口**，用户只能靠肉眼发现错误。
 */

import { useEffect, useMemo, useState } from "react";
import type {
  AgentQuestion,
  PromptDocument,
  ReviewSummary,
  RevisionRecord,
} from "./api";
import { ReviseDialog } from "./ReviseDialog";

export function PromptView({
  document,
  review,
  questions,
  onAnswer,
  revisions = [],
  revising = false,
  canUndo = false,
  reviseSource = "配置的模型",
  onRevise,
  onUndoRevision,
}: {
  document: PromptDocument;
  review: ReviewSummary | null;
  questions?: AgentQuestion[];
  onAnswer?: (questionId: string, choice: string) => void;
  /** 用户意见改写的账（后端落的盘）：对话框里显示"上次改了什么"。 */
  revisions?: RevisionRecord[];
  revising?: boolean;
  canUndo?: boolean;
  /** 改写结果来自哪家（按 .env 里的端点推出来的，不写死）。 */
  reviseSource?: string;
  onRevise?: (instruction: string) => void;
  onUndoRevision?: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [draft, setDraft] = useState("");
  const [reviseOpen, setReviseOpen] = useState(false);

  /** 正文：按段落顺序拼成纯文本，段落之间空一行。 */
  const plain = useMemo(() => {
    const parts: string[] = [];
    for (const section of document.sections) {
      parts.push(`【${section.title}】`);
      parts.push(...section.lines);
      parts.push("");
    }
    return parts.join("\n").trim();
  }, [document.sections]);

  // 换项目/重新反推时同步文本框内容；用户编辑过程中不覆盖（用 value 受控但要能改）
  useEffect(() => {
    setDraft(plain);
  }, [plain]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(draft);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const items = review?.items ?? [];
  const worst = items[0]?.level ?? "review";
  /**
   * **只在真有阻断项时**才提一句（用户 2026-10-08：「这次反推有 xx 处要你确认，
   * 这个隐藏，不需要出现在前端画面」）。
   *
   * 结果页要的就是一份能直接复制的提示词 —— 那行"N 处待确认"是统计，
   * 不是交付物的一部分，去掉。但**阻断项不能静默**（提示词当下不能用这种事
   * 必须让人看见），所以留着一条最简提示。
   */
  const blockers = items.filter((item) => item.level === "block");

  return (
    <div className="pv">
      {/*
        Agent 的选择题：**它唯一会主动打扰用户的地方**。
        点一个选项，答案直接写进提示词（走后端同一道门禁），正文当场更新。
        放在最上面、且做得极简 —— 用户点完就能继续往下看提示词。
      */}
      {(questions?.length ?? 0) > 0 && (
        <div className="ask">
          {questions!.map((question) => (
            <div className="ask__item" key={question.id}>
              <div className="ask__q">
                <span className="ask__tag">要我定一下</span>
                {question.text}
                {question.reason && (
                  <span className="muted tiny">（{question.reason}）</span>
                )}
              </div>
              <div className="ask__opts">
                {question.options.map((option) => (
                  <button
                    key={option}
                    className="btn btn--ghost ask__opt"
                    onClick={() => onAnswer?.(question.id, option)}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      {/*
        待确认列表：
        - **有问题（阻断）时**才出现在画面上，一句话说清，点开看原因；
        - 没有阻断项时**整块不渲染** —— 用户要的是"结果页只有提示词"。
        数据仍在接口里（`review.items`），没有丢，只是不占画面。
      */}
      {blockers.length > 0 && (
        <details className={`rv rv--${worst}`}>
          <summary className="rv__head">
            <span className="rv__dot" />
            这份提示词还有问题，点开看怎么处理（{blockers.length}）
          </summary>
          <ul className="rv__list">
            {items.map((item, index) => (
              <li key={`${item.kind}-${index}`} className={`rv__item rv__item--${item.level}`}>
                <span className="rv__label">{item.label}</span>
                <span className="rv__detail">{item.detail}</span>
                {item.suggestion && (
                  <span className="rv__tip muted tiny">{item.suggestion}</span>
                )}
              </li>
            ))}
          </ul>
          {(review?.truncated ?? 0) > 0 && (
            <p className="rv__more muted tiny">还有 {review?.truncated} 条未列出</p>
          )}
        </details>
      )}
      <div className="pv__bar">
        <span className="muted tiny">可以直接编辑，改完再复制</span>
        <div className="pv__actions">
          {/* 「修改提示词」：写一句话，AI 按它改提示词，改完直接填回上面的文本框。 */}
          <button
            className="btn"
            onClick={() => setReviseOpen(true)}
            disabled={revising}
          >
            {revising ? "AI 改写中…" : "修改提示词"}
          </button>
          <button className="btn btn--primary" onClick={copy}>
            {copied ? "已复制" : "复制提示词"}
          </button>
        </div>
      </div>
      <textarea
        className="pv__text"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        spellCheck={false}
      />
      <ReviseDialog
        open={reviseOpen}
        busy={revising}
        revisions={revisions}
        canUndo={canUndo}
        sourceLabel={reviseSource}
        onClose={() => setReviseOpen(false)}
        onSubmit={(instruction) => onRevise?.(instruction)}
        onUndo={() => onUndoRevision?.()}
      />
    </div>
  );
}
