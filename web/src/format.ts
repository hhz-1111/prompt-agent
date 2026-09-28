/**
 * 展示层的小工具。**不含业务规则** —— 三色语义与门禁判定都由后端给。
 */

import type { Position, PositionColor } from "./api";

export interface ColorLegendItem {
  color: PositionColor;
  label: string;
  hint: string;
  count: number;
}

/** 用户的唯一心智模型：蓝色换素材，橙色改文字，灰色别动。 */
export function colorLegend(counts: Record<PositionColor, number>): ColorLegendItem[] {
  return [
    { color: "blue", label: "可替换素材位", hint: "点击上传 / 选择", count: counts.blue ?? 0 },
    { color: "orange", label: "可编辑文案位", hint: "直接输入", count: counts.orange ?? 0 },
    { color: "gray", label: "锁定区（只读）", hint: "改了复刻就失败", count: counts.gray ?? 0 },
  ];
}

export function confidenceLabel(disposition: string): string {
  const map: Record<string, string> = {
    certain: "确定",
    accept: "高",
    review: "待确认",
    blank: "留空",
  };
  return map[disposition] ?? disposition;
}

export function motionLabel(kind: string): string {
  const map: Record<string, string> = {
    fixed: "固定机位",
    push_in: "推近",
    pan_or_track: "横摇/跟拍",
    handheld: "手持",
    unknown: "未知",
  };
  return map[kind] ?? kind;
}

export function shortPath(path: string): string {
  return path.replace(/creative\.|locked\./g, "").replace(/\[\d+\]/g, "[]");
}

export function formatRisk(risk: string): string {
  const map: Record<string, string> = {
    low: "低",
    medium: "中",
    high: "高",
    critical: "致命",
  };
  return map[risk] ?? risk;
}

/** 内部占位符 → 给用户看的说法。
 *
 * ``unknown`` 是"没反推出来"的内部标记（红线 7「不猜」）。
 * 直接把它显示给用户毫无意义 —— 用户不知道 unknown 是什么意思、
 * 更不知道该填什么。实测界面里出现过"光线 = unknown"。
 */
export function displayValue(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = Array.isArray(value) ? value.join("、") : String(value);
  const trimmed = text.trim();
  if (["unknown", "未知", "none", "null"].includes(trimmed.toLowerCase())) return "";
  if (trimmed === "未知") return "";
  return trimmed;
}

/** 该位置是否属于"未反推到、等用户补"。 */
export function isPendingValue(value: unknown): boolean {
  return displayValue(value) === "";
}

export interface BlueSlot {
  /** 素材在 references 数组里的下标 */
  index: number;
  /** 素材名，取第一个槽位名 */
  name: string;
  /** 该素材下的全部字段位置 */
  fields: Position[];
}

/** 把蓝位按**素材**分组，而不是按字段平铺。
 *
 * 原因：一个素材在契约里有 7 个字段（槽位名、类型、占位符、绑定句柄、
 * Seedance 归属、工序号、工序建议）。平铺给用户看会变成 21 行噪声，
 * 而用户的心智模型是「我要准备 3 个素材」，不是「我要填 21 个字段」。
 * 所以界面按素材出卡片，字段折叠在卡片里。
 */
export function groupBlueSlots(positions: Position[]): BlueSlot[] {
  const buckets = new Map<number, Position[]>();
  for (const position of positions) {
    if (position.color !== "blue") continue;
    const match = /references\[(\d+)\]/.exec(position.path);
    if (!match) continue;
    const index = Number(match[1]);
    const bucket = buckets.get(index);
    if (bucket) bucket.push(position);
    else buckets.set(index, [position]);
  }

  return [...buckets.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([index, fields]) => {
      const slots = fields.find((f) => f.path.endsWith(".slots"));
      const raw = slots?.value;
      const name = Array.isArray(raw) ? raw.join("、") : String(raw ?? "");
      return { index, name: name || `素材 ${index + 1}`, fields };
    });
}

/** 单张蓝位卡片里真正需要用户操作的字段（其余是自动位，折叠展示）。 */
export function actionableBlueFields(slot: BlueSlot): Position[] {
  return slot.fields.filter(
    (f) => f.path.endsWith(".asset_id") || f.path.endsWith(".slots") || f.path.endsWith(".type"),
  );
}

export function timecode(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

/** 位置按输出段落分组，便于用户"从提示词倒着找位置"。 */
export function groupByApplyTarget(
  positions: { path: string; apply_target: string }[],
): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const position of positions) {
    const key = position.apply_target || "其他";
    (groups[key] ??= []).push(position.path);
  }
  return groups;
}
