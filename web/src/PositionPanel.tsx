import { useMemo, useState } from "react";
import type { AssembleResponse, Position, PositionColor } from "./api";
import {
  actionableBlueFields,
  confidenceLabel,
  displayValue,
  formatRisk,
  groupBlueSlots,
  shortPath,
} from "./format";

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
  const blueSlots = useMemo(() => groupBlueSlots(positions), [positions]);

  const visible = useMemo(() => {
    let list = positions;
    if (filter !== "all") list = list.filter((p) => p.color === filter);
    if (onlyPending) list = list.filter((p) => p.provenance && p.provenance.confidence < 0.7);
    // 蓝位在下面按素材分组单独渲染，这里排除，避免同一批字段出现两次
    if (filter === "blue" || filter === "all") {
      list = list.filter((p) => p.color !== "blue");
    }
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

  const showBlue = filter === "all" || filter === "blue";

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
        {showBlue && blueSlots.length > 0 && (
          <div className="group">
            <h4 className="group__title">【参考素材】要换的素材（🔵）</h4>
            <ul className="manifest">
              {blueSlots.map((slot) => (
                <BlueSlotCard key={slot.index} slot={slot} onWrite={onWrite} />
              ))}
            </ul>
            <p className="muted tiny">
              素材的生成与合规由中台负责，这里只声明「需要哪些」与「绑定哪一个」。
            </p>
          </div>
        )}

        {grouped.length === 0 && !showBlue && <p className="muted pad">没有符合条件的位置</p>}
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

/** 一张素材卡片：把该素材的字段收在一处，并按需给出绑定入口。 */
function BlueSlotCard({
  slot,
  onWrite,
}: {
  slot: ReturnType<typeof groupBlueSlots>[number];
  onWrite: (path: string, value: unknown) => void;
}) {
  const assetField = slot.fields.find((f) => f.path.endsWith(".asset_id"));
  const [handle, setHandle] = useState(() =>
    assetField ? displayValue(assetField.value) : "",
  );
  const bound = Boolean(assetField && displayValue(assetField.value));
  const typeField = slot.fields.find((f) => f.path.endsWith(".type"));

  return (
    <li className={`slot-card${bound ? " is-bound" : ""}`}>
      <div className="slot-card__head">
        <span className={`dot dot--${bound ? "blue" : "gray"}`} />
        <strong>{slot.name}</strong>
        <select
          className="slot-card__type"
          value={displayValue(typeField?.value)}
          disabled={!typeField?.editable}
          onChange={(e) => typeField && onWrite(typeField.path, e.target.value)}
        >
          {(typeField?.choices.length
            ? typeField.choices
            : [displayValue(typeField?.value)]
          ).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
        <span className="chip">{bound ? "已绑定" : "待绑定"}</span>
      </div>
      {assetField && (
        <div className="slot-card__bind">
          <input
            value={handle}
            placeholder="粘贴上游回填的素材句柄"
            onChange={(e) => setHandle(e.target.value)}
          />
          <button
            className="btn btn--ghost btn--sm"
            disabled={!handle.trim() || !assetField.editable}
            onClick={() => onWrite(assetField.path, handle.trim())}
          >
            {bound ? "更新绑定" : "绑定"}
          </button>
        </div>
      )}
      <details className="slot-card__more">
        <summary>查看该素材的全部字段（{slot.fields.length}）</summary>
        <ul className="positions">
          {actionableBlueFields(slot).map((field) => (
            <li key={field.path} className="pos pos--blue">
              <span className="pos__label">{field.label}</span>
              <span className="pos__path mono">{shortPath(field.path)}</span>
              <div className="pos__value mono">{displayValue(field.value) || "（空）"}</div>
            </li>
          ))}
        </ul>
      </details>
    </li>
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
              {stringify(position.value) || (
                <em className="pos__pending">待填 —— 反推未能确定，请按原片填写</em>
              )}
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
  return displayValue(value);
}
