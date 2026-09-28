import { useMemo, useState } from "react";
import type { AssembleResponse, Position, PositionColor } from "./api";
import { confidenceLabel, formatRisk, shortPath } from "./format";

interface Props {
  assembled: AssembleResponse;
  onWrite: (path: string, value: unknown) => void;
}

type Filter = "all" | PositionColor;

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "全部" },
  { key: "blue", label: "🔵 素材位" },
  { key: "orange", label: "🟠 文案位" },
  { key: "gray", label: "⚫ 锁定区" },
];

export function PositionPanel({ assembled, onWrite }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const [onlyPending, setOnlyPending] = useState(false);

  const positions = assembled.positions.positions;
  const pending = useMemo(
    () => positions.filter((p) => p.provenance && p.provenance.confidence < 0.7),
    [positions],
  );

  const visible = useMemo(() => {
    let list = positions;
    if (filter !== "all") list = list.filter((p) => p.color === filter);
    if (onlyPending) list = list.filter((p) => p.provenance && p.provenance.confidence < 0.7);
    return list;
  }, [positions, filter, onlyPending]);

  const grouped = useMemo(() => {
    const map = new Map<string, Position[]>();
    for (const position of visible) {
      const key = position.apply_target || "其他";
      const bucket = map.get(key);
      if (bucket) bucket.push(position);
      else map.set(key, [position]);
    }
    return [...map.entries()];
  }, [visible]);

  return (
    <section className="panel">
      <header className="panel__head">
        <h3>位置标注</h3>
        <div className="panel__filters">
          {FILTERS.map((item) => (
            <button
              key={item.key}
              className={filter === item.key ? "tab is-active" : "tab"}
              onClick={() => setFilter(item.key)}
            >
              {item.label}
            </button>
          ))}
          {pending.length > 0 && (
            <button
              className={onlyPending ? "tab tab--warn is-active" : "tab tab--warn"}
              onClick={() => setOnlyPending((v) => !v)}
              title="只看置信度低于 0.70、需要你确认的位置"
            >
              待确认 {pending.length}
            </button>
          )}
        </div>
      </header>

      <p className="muted tiny panel__hint">
        蓝色换素材，橙色改文字，灰色别动。灰色是决定「像不像原片」的部分。
      </p>

      <div className="panel__body">
        {grouped.length === 0 && <p className="muted pad">没有符合条件的位置</p>}
        {grouped.map(([section, items]) => (
          <div key={section} className="group">
            <h4 className="group__title">【{section}】</h4>
            <ul className="positions">
              {items.map((position) => (
                <PositionRow key={position.path} position={position} onWrite={onWrite} />
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

function PositionRow({
  position,
  onWrite,
}: {
  position: Position;
  onWrite: (path: string, value: unknown) => void;
}) {
  const [draft, setDraft] = useState<string>(() => stringify(position.value));
  const [editing, setEditing] = useState(false);

  const isText = position.kind === "text" || position.kind === "longtext";
  const isNumber = position.kind === "number";

  // 锁定位永不进入编辑态 —— 门禁在 UI 这一层先挡住
  const canEdit = position.editable && (isText || isNumber);

  const commit = () => {
    setEditing(false);
    const next = isNumber ? Number(draft) : draft;
    if (next === position.value) return;
    // 台词行直接写 dialogue.line —— 说话人由相邻的 speaker 位置单独维护，
    // 这里不该顺手改它（改了会覆盖用户已选定的说话人）。
    onWrite(position.path, next);
  };

  return (
    <li className={`pos pos--${position.color}`}>
      <div className="pos__head">
        <span className={`dot dot--${position.color}`} />
        <span className="pos__label">{position.label}</span>
        <span className="pos__cat mono">{position.category}</span>
        {position.risk !== "low" && (
          <span className={`risk risk--${position.risk}`} title="改动风险">
            {formatRisk(position.risk)}
          </span>
        )}
        {position.provenance && (
          <span
            className={`conf conf--${position.provenance.disposition}`}
            title={`${position.provenance.basis} · ${position.provenance.note}`}
          >
            {confidenceLabel(position.provenance.disposition)}{" "}
            {position.provenance.confidence.toFixed(2)}
          </span>
        )}
        <span className="pos__path mono" title={position.path}>
          {shortPath(position.path)}
        </span>
      </div>

      <div className="pos__value">
        {position.choices.length > 0 ? (
          <select
            value={stringify(position.value)}
            disabled={!position.editable}
            onChange={(e) => onWrite(position.path, e.target.value)}
          >
            <option value={stringify(position.value)}>{stringify(position.value)}</option>
            {position.choices
              .filter((choice) => choice !== stringify(position.value))
              .map((choice) => (
                <option key={choice} value={choice}>
                  {choice}
                </option>
              ))}
          </select>
        ) : canEdit ? (
          editing ? (
            <textarea
              autoFocus
              value={draft}
              rows={isText && position.kind === "longtext" ? 3 : 1}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commit}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  commit();
                }
                if (e.key === "Escape") {
                  setDraft(stringify(position.value));
                  setEditing(false);
                }
              }}
            />
          ) : (
            <button
              className="pos__readonly"
              onClick={() => {
                setDraft(stringify(position.value));
                setEditing(true);
              }}
            >
              {stringify(position.value) || <em className="muted">（空，点击填写）</em>}
            </button>
          )
        ) : (
          <span className="pos__readonly is-locked">
            {stringify(position.value) || <em className="muted">（空）</em>}
          </span>
        )}
      </div>

      {(position.note || position.apply_target) && (
        <p className="pos__note muted tiny">
          {position.apply_target && <span>落在：{position.apply_target}　</span>}
          {position.note}
        </p>
      )}
    </li>
  );
}

function stringify(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.join("、");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
