# 视频高光剪辑 Skill — 技术方案

## 项目概述

基于火山引擎 OpenClaw / ArkClaw 框架，开发一个视频高光剪辑 Skill。输入一段视频和自然语言需求描述，自动识别高光片段并剪辑输出集锦视频。

## 技术架构

```
用户输入（视频 + 指令）
  → OpenClaw Skill 入口（SKILL.md 定义触发条件）
    → 视频预处理（下载/帧提取/音频提取）
    → 高光检测（多模态理解为主，规则引擎降级）
    → 片段定位 & 排序
    → 剪辑输出（LAS las_video_edit 为主，FFmpeg 降级）
  → 输出（高光集锦视频 + 时间戳说明）
```

## 项目目录结构

```
video-highlight-skill/
├── SKILL.md                    # Skill 定义
├── src/
│   ├── __init__.py
│   ├── main.py                 # 主入口
│   ├── video_fetcher.py        # 视频获取
│   ├── highlight_detector.py   # 高光检测引擎
│   ├── video_editor.py         # 剪辑 & 输出
│   ├── ark_client.py           # 火山方舟 API 封装
│   ├── las_client.py           # LAS 算子 API 封装
│   └── rule_engine.py          # 规则引擎降级
├── evaluation/
│   ├── test_cases/             # 30组测试用例
│   │   └── case_001/
│   │       ├── video.mp4
│   │       ├── ground_truth.json
│   │       └── metadata.yaml
│   ├── evaluator.py            # 自动评测
│   ├── llm_judge.py            # LLM-as-Judge
│   └── report.py               # 可视化报告
├── tests/
│   └── test_*.py
├── pyproject.toml
└── README.md
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 视频下载 | yt-dlp |
| 多模态理解 | 豆包 Seed/Seedance via Ark API |
| 视频剪辑(主) | LAS `las_video_edit` 算子 |
| 视频剪辑(降级) | FFmpeg |
| 音频分析 | librosa |
| 画面分析 | OpenCV |
| ASR | LAS `las_asr_pro` 或豆包语音模型 |
| 评测 | pytest + matplotlib + LLM-as-Judge |
| 包管理 | uv + pyproject.toml |

## API 参考

### LAS 视频智能剪辑算子

- 算子 ID：`las_video_edit`
- 提交接口：`POST https://operator.las.cn-beijing.volces.com/api/v1/submit`
- 轮询接口：`POST https://operator.las.cn-beijing.volces.com/api/v1/poll`
- 鉴权：Bearer Token（LAS API KEY）

### 火山方舟 Ark Chat Completion

- Endpoint：`https://ark.cn-beijing.volces.com/api/v3/chat/completions`
- 模型：`doubao-seed-2-0-pro`、`doubao-vision-*`
- 鉴权：API Key

---

## Phase 1：项目初始化 + API 封装

**目标：** 搭好项目骨架，封装火山引擎 API，能独立调用验证。

**产出：**
- `pyproject.toml` 项目配置与依赖
- `src/ark_client.py` — Ark Chat Completion API 封装（支持多模态输入）
- `src/las_client.py` — LAS 算子 API 封装（submit + poll + 轮询等待）
- `tests/` 对应单元测试，验证 API 连通性

**验证标准：**
- Ark API 能成功调用并返回多模态理解结果
- LAS API 能成功提交任务、轮询、获取结果

---

## Phase 2：视频获取模块

**目标：** 实现视频下载与预处理，支持多种来源。

**产出：**
- `src/video_fetcher.py`
  - 本地文件路径
  - URL 下载（B站/YouTube 通过 yt-dlp）
  - TOS 路径
  - 视频预处理：格式统一（mp4）、提取音频（wav）、关键帧采样

**验证标准：**
- 能从 B站/YouTube 下载视频
- 能提取音频和关键帧

---

## Phase 3：规则引擎（降级路径）

**目标：** 实现基于信号处理的高光检测，作为多模态路径的降级方案。

**产出：**
- `src/rule_engine.py`
  - 音频分析（librosa）：音量峰值、语速变化、静音段检测
  - 画面分析（OpenCV）：场景切换检测、运动强度、画面亮度变化
  - 综合打分：加权融合 → Top-K 片段输出

**验证标准：**
- 输入视频，输出 Top-K 高光片段的时间戳 + 分数
- 人工抽查，高光片段基本合理

---

## Phase 4：高光检测引擎（多模态主路径）

**目标：** 实现多模态高光检测，集成 Ark API，支持自动降级。

**产出：**
- `src/highlight_detector.py`
  - 视频抽帧（每2-3秒一帧）+ ASR 文本
  - 调用 Ark 多模态模型分析帧序列 + 文本 → 识别高光
  - Prompt 工程：输出结构化 JSON（时间戳 + 标签 + 精彩度评分）
  - 降级逻辑：Ark 失败/超时/配额不足 → 自动切换规则引擎

**验证标准：**
- 输入视频，输出结构化高光片段列表
- 降级逻辑可正常触发和恢复

---

## Phase 5：视频剪辑模块

**目标：** 实现视频剪辑输出，LAS 为主 + FFmpeg 降级。

**产出：**
- `src/video_editor.py`
  - 主路径：组装 task_description → 调用 LAS `las_video_edit` → 轮询 → 获取剪辑结果
  - 降级路径：FFmpeg 本地裁剪 + concat 拼接
  - 输出：集锦视频 + 时间戳说明

**验证标准：**
- 输入高光片段列表，输出剪辑后的集锦视频
- LAS 不可用时自动切换 FFmpeg

---

## Phase 6：主入口 + Skill 定义

**目标：** 串联全流程，编写 SKILL.md，实现端到端可用。

**产出：**
- `src/main.py` — 主入口，串联 Phase 1-5 所有模块
- `SKILL.md` — Skill 定义文件
  - YAML frontmatter（name/description/version）
  - 触发关键词
  - 工作流步骤（Step 0-5）
  - Gotchas 与审查标准

**验证标准：**
- 从用户输入到输出集锦视频，全流程跑通
- Skill 可被 OpenClaw 正确加载和调用

---

## Phase 7：评测体系

**目标：** 构建 30 组测试用例 + 自动评测流水线。

**产出：**
- `evaluation/test_cases/` — 30 组测试用例（体育/游戏/Vlog/演讲/综艺/纪录片）
  - 每组包含：视频、Ground Truth 时间戳、metadata
- `evaluation/evaluator.py` — 自动评测
  - 时间戳 IoU（预测区间 vs GT 区间的交并比）
  - Top-K 召回率
- `evaluation/llm_judge.py` — LLM-as-Judge 多维度打分
  - 节奏感、内容完整性、精彩程度
- `evaluation/report.py` — 可视化报告生成

**验证标准：**
- 一键运行评测流水线
- 输出评测报告（含指标图表）

---

## Phase 8：文档与收尾

**目标：** 编写完整项目文档。

**产出：**
- `README.md` — 项目说明、安装指南、使用示例
- 最终全流程回归测试

---

## 核心风险 & 应对

| 风险 | 应对 |
|------|------|
| 豆包多模态对长视频上下文有限 | 分片处理 + 摘要级联 |
| LAS 算子调用不稳定 | 本地 FFmpeg 降级路径 |
| 高光标准主观 | 30组测试用例覆盖多种视频类型，多维度评测 |
| B站/抖音下载可能受限 | 支持本地文件上传 + 公开数据集备选 |
