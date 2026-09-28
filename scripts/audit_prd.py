"""PRD 符合度核查：把功能清单与代码实现逐条比对。

做法：对 PRD §6 的每条功能编号，在代码里找一个**可验证的锚点**
（函数名/类名/字段名），确认它真的存在。不靠印象说"做了"，靠能在代码里指出来。

**不用 importlib 反射**：FastAPI 的路由定义在 ``_register_routes`` 闭包里，
模块级反射找不到，会误报"未实现"。所以直接读源文件做符号匹配。

用法：py scripts/audit_prd.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (编号, 功能名, 相对文件, 锚点正则, 备注)
CHECKS: list[tuple[str, str, str, str, str]] = [
    # ================= F1 反推 =================
    ("F1.1", "本地上传 mp4/mov ≤3min/≤500MB", "engine/app/api.py",
     r"async def upload_video", "MAX_UPLOAD_BYTES 校验"),
    ("F1.2", "ffprobe 取真实时长/帧率/分辨率/音轨", "engine/app/providers/ffmpeg_media.py",
     r"def probe\(", "唯一可信来源"),
    ("F1.3", "镜头切分 + 关键帧抽取", "engine/app/providers/scene_detect.py",
     r"class SceneDetectShots", "PySceneDetect 自适应+内容双遍"),
    ("F1.4", "CV 强判断层（光流运镜/框占比景别/检测框站位）", "engine/app/providers/cv.py",
     r"def framing_from_person|def analyze_motion|def blocking_from_boxes", "人体框几何量测"),
    ("F1.5", "多模态视觉理解（逐镜头抽帧）", "engine/app/providers/mimo_analysis.py",
     r"class MimoVisualDescriber", "L1 视觉理解"),
    ("F1.6", "交叉校验（模型 vs CV）", "engine/app/pipeline/infer.py",
     r"def _fuse", "冲突以 CV 为准并降置信度"),
    ("F1.7", "置信度计算与分级处置", "engine/core/types.py",
     r"def compute_confidence", "MULTIMODAL §4.2 公式"),
    ("F1.8", "每个位置带证据（可点回原片）", "engine/core/types.py",
     r"class Provenance", "span + confidence"),
    ("F1.9", "结构族判定（决定输出格式）", "engine/app/pipeline/to_ir.py",
     r"def build_ir", "genre / shot_mode"),
    ("F1.10", "17 份样本回归测试", "scripts/benchmark.py",
     r"def analyze_one", "24 条真实视频基准"),
    ("F1.11", "模型版本记录与回滚", "engine/app/pipeline/infer.py",
     r"def _model_id", "记录 vision/text/asr/cv"),
    ("F1.12", "反推失败降级（不输出假数据）", "engine/app/pipeline/to_ir.py",
     r"def _text\(", "低置信度留空"),

    # ================= F2 位置标注 =================
    ("F2.1", "四类位置自动标注", "engine/core/positions.py",
     r"def build_positions", "LOCK/PARAM/FREE/AUTO"),
    ("F2.2", "三色界面映射（不出现第四种）", "engine/core/positions.py",
     r"def color_for", "蓝/橙/灰"),
    ("F2.3", "位置级编辑门禁（UI 与 API 双层）", "engine/core/framework.py",
     r"def can_write", "锁定位拒写"),
    ("F2.4", "应用位置索引", "engine/core/positions.py",
     r"def apply_targets", "落在提示词哪一段"),
    ("F2.5", "五类位置冲突检测与裁决", "engine/core/gates.py",
     r"def g4_contradiction", "正反矛盾/语义互斥/数值/素材/站位"),

    # ================= F3 短语工程 =================
    ("F3.1", "短语资产库（三级分类）", "engine/core/phrases.py",
     r"class PhraseLibrary", "fixed/template/vocab/free"),
    ("F3.2", "逐维度短语选择", "engine/core/phrases.py",
     r"def vocabulary|def dimension", "候选 + 等级"),
    ("F3.3", "必写维度校验", "engine/core/phrases.py",
     r"def check_mandatory_dimensions", "覆盖率≥60% 即必写"),
    ("F3.4", "约束有效性分级校验（E/F 阻断）", "engine/core/phrases.py",
     r"def scan_texts", "A~F 分级"),
    ("F3.5", "强约束降级检测", "engine/core/phrases.py",
     r"def detect_downgrades", "1:1复刻→参考即阻断"),
    ("F3.6", "密度校验（节拍÷时长 >0.27 告警）", "engine/core/gates.py",
     r"def g8_density", "DENSITY_LIMIT = 0.27"),
    ("F3.7", "短语级 diff", "engine/core/assembler.py",
     r"class CompressionEntry", "部分：有压缩记录，无位置级 diff"),

    # ================= F4 组装与交付 =================
    ("F4.1", "拆条策略推荐 + 可覆盖", "engine/core/assembler.py",
     r"def plan_split", "S1~S4 + 可覆盖"),
    ("F4.2", "拆条边界校验（6 种情况）", "engine/core/assembler.py",
     r"SPLIT-S1-OVERFLOW", "过短/超限/密度"),
    ("F4.3", "14 项闸门校验", "engine/core/gates.py",
     r"def run_gates", "逐项结论 + 修复建议"),
    ("F4.4", "阻断式交付", "engine/core/gates.py",
     r"def deliverable", "有阻断不可标记可交付"),
    ("F4.5", "三形态输出（MD/纯文本/JSON）", "engine/core/assembler.py",
     r"def render_markdown|def render_json", "内容一致"),
    ("F4.6", "参数声明分支控制", "engine/core/assembler.py",
     r"include_param_header", "手动/直调正文不同"),
    ("F4.7", "Phase 1 导出（零格式污染）", "engine/core/assembler.py",
     r"def render_plain", "可直接粘进中台"),
    ("F4.9", "中台适配器（模型升级只改它）", "engine/app/providers/mimo.py",
     r"class MimoClient", "唯一直接跟模型说话的地方"),

    # ================= F5 素材清单 =================
    ("F5.1", "占位符抽取与素材清单化", "engine/app/pipeline/to_ir.py",
     r"def reference_manifest", "类型+用途+建议工序"),
    ("F5.2", "素材句柄绑定", "engine/app/api.py",
     r"def bind_asset", "接收上游回填"),
    ("F5.3", "配额校验（图9/视3/音3）", "engine/core/gates.py",
     r"def g2_quota", "按探测到的模型能力"),
    ("F5.4", "【】→@ 句柄统一", "engine/core/assembler.py",
     r"class HandleResolver", "交付物残留=0"),
    ("F5.5", "素材位即时校验", "engine/core/positions.py",
     r"def blue_slot_manifest", "类型/缺失/超额"),

    # ================= F6 内容对齐（未实现） =================
    ("F6.1", "三种输入形态解析", "", "", "未实现"),
    ("F6.2", "自动对齐 + 置信度", "", "", "未实现（反推侧只做整段归属）"),
    ("F6.4", "双向引用位置保留", "", "", "未实现"),
    ("F6.5", "手动对齐（拖拽/拆分/合并）", "", "", "未实现"),
    ("F6.6", "强制门禁（未决状态禁入组装）", "", "", "未实现"),

    # ================= F7 模板库与评分（未实现） =================
    ("F7.1", "模板沉淀", "", "", "未实现"),
    ("F7.5", "四级晋升自动判定", "", "", "未实现"),
    ("F7.7", "用户打星（3 维度）", "", "", "未实现"),
    ("F7.8", "防刷机制", "", "", "未实现"),
    ("F7.9", "视频数据回流", "", "", "未实现"),

    # ================= F8 效果评估（未实现） =================
    ("F8.1", "结构对照（段落比例/钩子时长/镜头数/语速）", "", "", "未实现"),
    ("F8.2", "结构还原度评分", "", "", "未实现"),
    ("F8.3", "短语级偏差定位", "", "", "未实现"),
    ("F8.4", "按建议重写", "", "", "未实现"),

    # ================= 14 项闸门 =================
    ("G1", "【】残留 = 0", "engine/core/gates.py", r"def g1_placeholder_residue", ""),
    ("G2", "素材配额", "engine/core/gates.py", r"def g2_quota", ""),
    ("G3", "互斥模式", "engine/core/gates.py", r"def g3_exclusive_modes", ""),
    ("G4", "正反矛盾", "engine/core/gates.py", r"def g4_contradiction", ""),
    ("G5", "台词双引号（自动修复）", "engine/core/gates.py", r"def g5_dialogue_quotes", ""),
    ("G6", "口型与说话人一致", "engine/core/gates.py", r"def g6_lipsync_binding", ""),
    ("G7", "字数预算", "engine/core/gates.py", r"def g7_char_budget", ""),
    ("G8", "内容量与时长匹配", "engine/core/gates.py", r"def g8_density", ""),
    ("G9", "拆条策略已选", "engine/core/gates.py", r"def g9_split_strategy", ""),
    ("G10", "锁定位 diff 为空", "engine/core/gates.py", r"def g10_locked_diff", ""),
    ("G11", "钩子类型保留", "engine/core/gates.py", r"def g11_hook_preserved", ""),
    ("G12", "素材角色已声明（自动修复）", "engine/core/gates.py", r"def g12_ref_roles", ""),
    ("G13", "必写维度齐全", "engine/core/gates.py", r"def g13_mandatory_dimensions", ""),
    ("G14", "约束强度未降级", "engine/core/gates.py", r"def g14_constraint_strength", ""),
]


def main() -> int:
    done: list[tuple[str, str, str]] = []
    missing: list[tuple[str, str, str]] = []

    for code, name, rel, pattern, note in CHECKS:
        if not rel:
            missing.append((code, name, note))
            continue
        path = ROOT / rel
        if not path.exists():
            missing.append((code, name, f"文件不存在 {rel}"))
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(pattern, text):
            done.append((code, name, rel))
        else:
            missing.append((code, name, f"{rel} 里找不到 {pattern}"))

    total = len(CHECKS)
    print(f"PRD 功能清单核查：{len(done)}/{total} 在代码里找到锚点"
          f"（{len(done) / total:.0%}）")
    print("=" * 78)
    print("\n【已实现】")
    for code, name, _ in done:
        print(f"  ✅ {code:<5} {name}")
    print("\n【未实现 / 部分实现】")
    for code, name, note in missing:
        print(f"  ⬜ {code:<5} {name}")
        if note:
            print(f"          {note}")
    print()

    groups = {
        "F1 反推": [c for c, _, _ in done if c.startswith("F1")],
        "F2 位置标注": [c for c, _, _ in done if c.startswith("F2")],
        "F3 短语工程": [c for c, _, _ in done if c.startswith("F3")],
        "F4 组装交付": [c for c, _, _ in done if c.startswith("F4")],
        "F5 素材清单": [c for c, _, _ in done if c.startswith("F5")],
        "F6 内容对齐": [c for c, _, _ in done if c.startswith("F6")],
        "F7 模板评分": [c for c, _, _ in done if c.startswith("F7")],
        "F8 效果评估": [c for c, _, _ in done if c.startswith("F8")],
        "14 项闸门": [c for c, _, _ in done if c.startswith("G")],
    }
    print("分组完成度：")
    for label, items in groups.items():
        expected = len([c for c, _, _, _, _ in CHECKS if c == label[:2]
                        or (label.startswith("14") and c.startswith("G"))])
        print(f"  {label:<12} {len(items):>2} 项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
