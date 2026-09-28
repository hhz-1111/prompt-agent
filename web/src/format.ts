/**
 * 展示层的小工具。**不含业务规则** —— 三色语义与门禁判定都由后端给。
 */

import type { PositionColor } from "./api";

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
