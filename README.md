# 爆款视频复刻 · 提示词 Agent

上传一条爆款视频，Agent 反推出它的提示词结构；用户换上自己的素材、写自己的台词；
Agent 组装出一份能**直接生成**（一遍成功）的提示词。

---

## 一、这套代码是怎么长的

```
engine/core/        领域内核 —— 纯 Python，零第三方依赖，可独立测试
  types.py          枚举、置信度公式、溯源
  framework.py      框架契约 v1：Schema 即数据，字段规则单点定义
  positions.py      四类位置 → 三色映射
  phrases.py        约束有效性分级、强约束降级检测、必写维度覆盖
  assembler.py      IR → 提示词，含应用位置溯源与压缩记录
  gates.py          14 项交付闸门
  contracts.py      反推中间表示（视频 → IR 的桥）
  ports.py          Provider 端口（靠 Protocol 定义，实现可替换）

engine/app/         应用层 —— 依赖 core，反向不成立
  config.py         配置（密钥只从环境读）
  store.py          项目持久化
  api.py            HTTP 接口
  pipeline/infer.py 反推编排（六步）
  pipeline/to_ir.py 分析结论 → Framework IR
  providers/        ffmpeg / MiMo / CV 的具体实现

assets/             资产（语料 + 短语库，由脚本从规范文档重建）
web/                前端（React + TS）
scripts/            资产构建与运维脚本
```

### 三条架构纪律

**① 领域内核不依赖任何东西。**
`engine/core` 里没有一次网络调用、没有一次文件 IO（资产路径是装配时注入的）。
所以它能在毫秒级跑完全部单测，也能在没网、没 Key 的机器上验证。

**② Schema 即数据。**
`framework.py` 里的 `FIELDS` 既是字段定义，也是字段级锁定规则、
四类位置标注、应用位置追踪的唯一来源。于是"哪个字段能改"只有一个答案，
改契约只需改一处。

**③ 门禁只判一次，UI 与 API 都过它。**
`framework.can_write()` 是唯一的写入门禁。前端不重复实现规则，
后端也不绕过它 —— 锁定位在两层都写不进去（PRD F2.3）。

---

## 二、跑起来

```bash
# 1. 依赖
py -m venv .venv
.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" pydantic python-multipart httpx pytest

# 2. 配置（密钥不进代码库）
copy .env.example .env      # 然后填入 MIMO_API_KEY

# 3. 前端
cd web && npm install && npm run build && cd ..

# 4. 启动
.venv\Scripts\python.exe scripts\run.py
#   → 界面   http://127.0.0.1:8000
#   → 接口文档 http://127.0.0.1:8000/docs
```

开发期可以前后端分离：`py scripts/run.py` + `cd web && npm run dev`（5173 端口，已配好代理）。

### 验证

```bash
.venv\Scripts\python.exe -m pytest          # 142 项单测
.venv\Scripts\python.exe scripts\smoke_live.py --base http://127.0.0.1:8000
```

`smoke_live.py` 打真实端口、真实视频、真实模型调用，
用来确认"启动起来真的能用"——单测用替身保证逻辑，它保证联通。

---

## 三、资产是脚本生成的，不是手写的

```
MULTIMODAL / TEMPLATES / PHRASES 等规范文档
        │
        ├─ scripts/ingest_corpus.py   样本归档 + 跨样本覆盖率统计
        └─ scripts/build_phrases.py   句式模板 / 禁止项库 / 维度 / 降级规则
                    ▼
        assets/corpus/manifest.json     assets/phrases/phrases.json
```

**为什么不手写 JSON**：禁止项库与句式模板要求逐字一致（G14 的判据）。
手抄一遍就可能错一个字，而错一个字就毁掉约束强度。
脚本让资产可从规范文档重建，每条还记录来源章节。

覆盖率是**算出来的**，不是抄的。当前 10 份样本的结果：

| 维度 | 覆盖 | 含义 |
|---|---|---|
| 眨眼节拍 / 皮肤质感 / 景别 | 10/10 | 模型默认会做错的地方，必须逐字钉住 |
| 机位 / 运镜 | 10/10 | 画面失控的高风险项 |
| 禁止字幕 / 禁止BGM / 禁臆造道具 | 8/10 | 兜底约束 |
| 情绪 | 7/10 | 与 PRD 的"≥70% 即合理"一致 |

---

## 四、三个必须知道的工程取舍

### 「不猜」是硬约束，不是口号

低置信度字段**留空**而不是填一个像样的猜测。落地方式：

- `<0.50` → 不填，折叠展示请用户补；
- `0.50~0.70` → 标黄，必须用户确认；
- `≥0.70` → 直接采用。

代价是反推结果看着"不完整"。这是刻意的：猜错会导致后续全链路偏移，
而用户很难发现（PRD 红线 7）。

### CV 层有缺口，代码里写明了

规范要求用光流算运镜、用人体框占比算景别、用检测框算站位。
那需要 OpenCV + 姿态估计/人体检测 —— 当前环境（Python 3.14）装不上。

所以 `providers/cv.py` 的策略是：**能确定性算的认真算，算不了的返回
`unknown` 并登记缺口**，绝不编一个数字冒充 CV 结果。缺失的能力会体现在
`/api/health` 的 `cv.missing` 里，对应维度的置信度自动降档为"模型单独判断"。

接入 OpenCV / YOLO / 外部检测服务后，改 `detect_capabilities()` 打开开关即可，
其余代码不动。

### 台词不做比例切分

MiMo 的 ASR 返回整段文本，**不带词级时间戳**。此时把文本按镜头时长比例切开
是在编造对齐关系，还会切出"你这气色也太好了吧，是 / 不是偷偷做项目啦？"这种半句话。

现在的做法：整段归给**时序重叠最多**的那个镜头，其余留空。
`TranscriptSegment.has_timestamps` 如实标注时间码是真实对齐还是仅由音频时长推导 ——
这个标记缺失过一版，导致同一句台词被复制到每个重叠镜头。

切割交给对齐环节由用户定。

---

## 五、交付闸门（14 项）

`engine/core/gates.py` 是纯函数，逐项给结论 + 可执行修复建议。
有阻断项时 `deliverable = False`，导出时会如实带上判定（红线 10）。

几个容易忽略的：

| 闸门 | 为什么重要 |
|---|---|
| **G8 密度** | `节拍数 ÷ 时长 ≤ 0.27`。把 30 秒内容塞进 15 秒提示词，模型必然丢细节 —— 抽卡的头号原因 |
| **G10 LOCK diff** | 逐字段比对锁定位。实测踩过一个 bug：标量数组被摊平成 `positions[0]`，导致契约里声明的规则永远匹配不上、站位改动漏检 |
| **G13 必写维度** | 审的是**最终提示词**而不是 IR —— IR 里 `blink` 存的是键名 `"5s"`，逐字表述要到组装才展开 |
| **G14 约束强度** | 只体检约束段落。台词里说"你先参考一下我的动作"是正常说话，不是把 `1:1复刻` 降级 |

---

## 六、接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 能力矩阵 + CV 缺口 |
| GET | `/api/schema` | 框架契约（前端据此渲染面板，不硬编码字段） |
| GET | `/api/phrases` | 短语库 |
| POST | `/api/projects` | 建项目 |
| POST | `/api/projects/{id}/video` | 上传视频（≤500MB） |
| POST | `/api/projects/{id}/infer` | 提交反推（异步） |
| GET | `/api/projects/{id}/status` | 轮询进度 |
| PUT | `/api/projects/{id}/field` | 写单个位置（**门禁在这里拦**） |
| PUT | `/api/projects/{id}/dialogue` | 填台词 |
| POST | `/api/projects/{id}/bind` | 绑定上游素材句柄 |
| GET | `/api/projects/{id}/assemble` | 组装 + 14 项闸门 |
| GET | `/api/projects/{id}/export?form=` | 三形态导出 |

---

## 七、还没做的

按 PRD 的里程碑，当前完成 M0 + M1 的引擎部分与 M2 的链路。

| 里程碑 | 状态 |
|---|---|
| M0 框架定型 + 资产化 | ✅ 契约冻结 v1、资产脚本化、能力矩阵配置化 |
| M1 反推引擎 + 短语打通 | ✅ 六步编排、CV+VLM 交叉校验、置信度分级；**17 份样本回归待补** |
| M2 闭环打通 | ✅ 素材绑定 / 台词填写 / 组装 / 闸门 / 导出；**内容对齐环节待补** |
| M3 用户打星 + 模板晋升 | ⬜ |
| M4 一遍成功调优 | ⬜ 需要真实生成配额跑 A/B |
| M5 Phase 2 直调中台 | ⬜ 适配器接口已留好 |
| M6 视频数据回流 | ⬜ |

**明确未完成的三项**：

1. **回归测试未跑通**：`MULTIMODAL §5` 要求 17 份样本逐维度比对，
   当前只有 10 份文本样本，且缺视频源（没有 ground truth 的视频就无法测反推准确率）；
2. **内容对齐（F6.x）未做**：用户粘贴台词 → 自动对齐到位置 → 三列对照表（含置信度）；
   当前只能逐条填；
3. **AB-1/2/3 未做**：需要真实生成配额。

---

## 八、术语对齐

| 说法 | 指什么 |
|---|---|
| 框架 / 契约 | `framework.py` 里的四段式结构，冻结为 `mod-framework-v1` |
| IR | Framework IR，反推产出的结构化中间表示 |
| 锁定位 / LOCK | 改了复刻就失败的位置，只读 |
| 档位位 / PARAM | 有白名单，只能切换 |
| 创作位 / FREE | 用户自由改写 |
| 蓝位 / 橙位 / 灰位 | 用户视角的三种颜色，由四类位置单向映射 |
| 闸门 | 交付前 14 项自检 |
