# 视频高光剪辑 Skill

基于火山引擎 Ark API（多模态理解）和 LAS 算子（智能剪辑），输入原始长视频 + 剪辑指令，自动识别高光片段并输出集锦视频。

## 核心能力

1. **高光识别** — 自动检测视频中的精彩片段（精彩动作、关键场景、情绪爆发、转场亮点）
2. **多模态筛选** — 结合画面变化、音频特征、语义理解、ASR 字幕等多维线索定位高光片段
3. **智能剪辑** — 片段裁剪 + 转场拼接，支持 LAS 云端剪辑和 FFmpeg 本地降级
4. **结构化输出** — 输出最终高光视频，附带关键时间戳和片段说明

## 输入输出

| 输入 | 说明 | 必需 |
|------|------|------|
| 原始视频 | 本地文件路径 / URL / TOS 路径 | 是 |
| 剪辑目标 | 如"精彩集锦""进球片段""高能时刻" | 否 |
| 时长要求 | 如"3 分钟内""不超过 30 秒" | 否 |
| 风格/主题 | 如"快节奏""电影感""卡点混剪" | 否 |
| 剪辑模式 | `fast`（快速，仅视觉特征）或 `full`（完整，多模态+ASR），默认 `full` | 否 |

| 输出 | 说明 |
|------|------|
| 高光视频 | 剪辑后的集锦视频（mp4, H.264 + AAC） |
| 时间戳说明 | 每个片段的起止时间、精彩度评分、标签 |
| JSON 导出 | 结构化片段数据，含时间戳和评分 |
| Token 用量 | 多模态检测消耗的 token 数 |

## 工作流

```
执行进度：
- [ ] Step 0: 前置检查（环境变量、输入可访问性、Region 一致性）
- [ ] Step 1: 视频获取与预处理（下载/转码/抽帧/提取音频）
- [ ] Step 2: Token 预估（多模态路径必须告知用户）
- [ ] Step 3: 高光检测（Ark 多模态 / 规则引擎降级）
- [ ] Step 4: 视频剪辑（LAS 云端 / FFmpeg 本地降级）
- [ ] Step 5: 结果呈现（视频 + 片段列表 + JSON + Token 用量）
```

## 架构

```
用户输入（视频 + 指令）
  → VideoFetcher（下载/预处理/抽帧/提取音频）
    → HighlightDetector（多模态 Ark API 主路径 / 规则引擎降级）
      → VideoEditor（LAS las_video_edit 主路径 / FFmpeg 降级）
        → 输出（集锦视频 + 时间戳 + JSON + Token 用量）
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/wh493757101-oss/Video-Highlight-Skill.git
cd Video-Highlight-Skill

# 安装依赖
pip install -e ".[dev]"

# 安装外部工具
# FFmpeg: https://ffmpeg.org/download.html
# yt-dlp: pip install yt-dlp (用于 URL 下载)
```

## 环境变量

```bash
export ARK_API_KEY="your-ark-api-key"          # 火山方舟 Ark API Key（必需）
export LAS_API_KEY="your-las-api-key"          # LAS 算子 API Key（可选，FFmpeg 可降级）
export TOS_ENDPOINT="https://tos-cn-beijing.volces.com"  # TOS 对象存储 Endpoint（TOS 路径时必需）
export TOS_ACCESS_KEY="your-access-key"        # TOS Access Key（TOS 路径时必需）
export TOS_SECRET_KEY="your-secret-key"        # TOS Secret Key（TOS 路径时必需）
```

## 使用示例

```python
from src.main import VideoHighlightPipeline

pipeline = VideoHighlightPipeline()

# 本地视频
result = pipeline.run_from_path(
    video_path="/path/to/video.mp4",
    description="剪辑最精彩的 60 秒",
)

# URL 视频
result = pipeline.run_from_url(
    url="https://example.com/video.mp4",
    description="进球集锦",
)

# TOS 视频（需配置 TOS 环境变量 + pip install boto3）
from src.video_fetcher import TosSource
result = pipeline.run(
    TosSource("tos://my-bucket/path/to/video.mp4"),
    description="精彩集锦",
)

# 查看结果
print(pipeline.format_result(result))
print(pipeline.export_json(result))
```

## 项目结构

```
video-highlight-skill/
├── SKILL.md                    # Skill 定义
├── README.md
├── pyproject.toml
├── src/
│   ├── main.py                 # 主入口 Pipeline
│   ├── video_fetcher.py        # 视频获取与预处理
│   ├── highlight_detector.py   # 高光检测引擎（多模态 + 降级）
│   ├── video_editor.py         # 视频剪辑（LAS + FFmpeg）
│   ├── ark_client.py           # 火山方舟 Ark API 封装
│   ├── las_client.py           # LAS 算子 API 封装
│   └── rule_engine.py          # 规则引擎降级（音频 + 画面分析）
├── evaluation/
│   ├── evaluator.py            # IoU 自动评测
│   ├── llm_judge.py            # LLM-as-Judge 多维度打分
│   ├── report.py               # 可视化报告生成
│   ├── runner.py               # 评测流程编排器
│   └── test_cases/
│       ├── open_data/          # 35 组本地视频用例（SumMe 数据集）
│       └── self-built_data/    # 10 组远程视频用例
├── scripts/
│   └── verify_e2e.py           # 端到端验证脚本
└── tests/
    └── test_*.py
```

## 运行测试

```bash
# 全部测试
pytest tests/ -v

# 含覆盖率
pytest tests/ -v --cov=src --cov=evaluation --cov-report=term-missing
```

## 运行评测

```bash
# 1. 放入视频文件到 evaluation/test_cases/open_data/case_XXX/
# 2. 填充 ground_truth.json（高光时间戳标注）
# 3. 填充 self-built_data/cases.yaml 中的 source_url
# 4. 运行评测
python -c "
from evaluation.runner import EvalRunner, EvalRunConfig

config = EvalRunConfig(
    test_cases_root='evaluation/test_cases',
    output_dir='reports',
    skip_edit=True,
)
runner = EvalRunner(config)
eval_report, judge_report, report_text = runner.run()
print(report_text)
"
```

详细评测框架说明见下方 [评测体系](#评测体系) 章节。

## 评测体系

完整的评测框架支持自动化质量评估，包含量化评测（tIoU 匹配）和 LLM Judge 主观评分。

### 目录结构

```
evaluation/test_cases/
├── open_data/              # 35 组本地视频（SumMe 数据集）
│   ├── cases.yaml          # 用例注册表
│   └── case_XXX/
│       ├── video.mp4
│       ├── instruction.json
│       ├── ground_truth.json
│       └── metadata.yaml
└── self-built_data/        # 10 组远程视频
    ├── cases.yaml
    └── case_XXX/
        ├── instruction.json
        └── ground_truth.json
```

### 运行评测

```python
from evaluation.runner import EvalRunner, EvalRunConfig

config = EvalRunConfig(
    test_cases_root="evaluation/test_cases",
    output_dir="reports",
    skip_edit=True,          # 跳过 LAS 剪辑，仅评测检测
    skip_llm_judge=False,    # 启用 LLM Judge
    iou_threshold=0.5,
    case_filter=[],          # 空 = 全部用例，可指定 ["case_001", "case_002"]
)
runner = EvalRunner(config)
eval_report, judge_report, report_text = runner.run()
print(report_text)
```

### 评测指标

| 指标 | 说明 |
|------|------|
| IoU / F1 | 片段时间戳匹配精度 |
| Hit Rate @1/@3 | Top-K 片段命中率 |
| MAE | 平均时间偏差（秒） |
| LLM Judge | 节奏感/完整性/精彩度/指令契合度（1-5 分） |

### 添加测试用例

1. 在 `cases.yaml` 中注册用例（id / category / difficulty / instruction）
2. 创建 `case_XXX/` 目录，放入视频文件
3. 编写 `instruction.json`（含 `prompt` 字段）
4. 编写 `ground_truth.json`（含 `highlights` 数组，每项含 `start_time` / `end_time` / `label` / `score`）

```json
// instruction.json
{"prompt": "帮我把精彩片段剪成60秒集锦，节奏要快"}

// ground_truth.json
{
  "highlights": [
    {"start_time": 10.0, "end_time": 25.0, "label": "精彩动作", "score": 0.8},
    {"start_time": 45.0, "end_time": 60.0, "label": "关键场景", "score": 0.7}
  ]
}
```

## 降级策略

| 环节 | 主路径 | 降级路径 | 触发条件 |
|------|--------|----------|----------|
| 高光检测 | Ark 多模态 API | 规则引擎（librosa + OpenCV） | Ark API 不可用或异常 |
| 视频剪辑 | LAS las_video_edit | FFmpeg 本地裁剪拼接 | LAS API 不可用或异常 |

降级自动触发，无需手动切换。可通过配置关闭降级：

```python
from src.highlight_detector import DetectorConfig
from src.video_editor import EditorConfig

detector_cfg = DetectorConfig(fallback_enabled=False)  # Ark 失败直接抛异常
editor_cfg = EditorConfig(fallback_enabled=False)      # LAS 失败直接抛异常
```

## 审查标准

**运维层面：**
- [ ] 环境变量是否正确配置（ARK_API_KEY 已设置）
- [ ] 输入文件是否成功加载（非空、可解码）
- [ ] Token 用量是否已告知用户
- [ ] 输出结果是否正确呈现（视频路径 + 片段列表 + JSON）

**业务层面：**
- [ ] 高光片段时长合理（建议 2-10 秒/段）
- [ ] 集锦总时长不超过原视频的 30%
- [ ] 片段之间过渡自然（0.5s 转场）
- [ ] 输出视频格式为 mp4（H.264 + AAC）

## 故障排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `ARK_API_KEY 未设置` | 环境变量缺失 | `export ARK_API_KEY=your-key` |
| `TOS 配置不完整` | TOS 环境变量缺失 | 设置 `TOS_ENDPOINT`/`TOS_ACCESS_KEY`/`TOS_SECRET_KEY` |
| `TOS 下载需要 boto3` | boto3 未安装 | `pip install boto3` |
| `视频下载失败` | URL 不可访问或 yt-dlp 版本过旧 | 检查 URL 可访问性，升级 yt-dlp |
| `视频下载超时` | 网络慢或文件过大 | 增大 `UrlSource` timeout 或手动下载 |
| `无法打开视频` | 文件损坏或格式不支持 | 检查文件完整性，确认 FFmpeg 已安装 |
| `视频文件为空` | 0 字节文件 | 检查视频源文件 |
| `视频转码失败/超时` | FFmpeg 版本不兼容或磁盘空间不足 | 检查 FFmpeg 版本，清理磁盘空间 |
| `音频提取失败` | 视频无音频流 | 正常行为，Pipeline 降级为纯视觉检测 |
| `关键帧采样为空` | 视频无法解码 | 检查视频编码格式，确认 OpenCV 支持 |
| `frames_dir 为空` | 预处理步骤失败 | 检查上游 VideoFetcher 输出日志 |
| `未检测到高光片段` | 视频内容过于静态或无变化 | 调整 `frame_interval` 或规则引擎阈值 |
| `LAS 任务超时` | 视频过大或 LAS 服务繁忙 | 增大轮询超时，或使用 FFmpeg 降级 |
| `LAS Region 不一致` | LAS Region 与 TOS Bucket 区域不同 | 确保两者区域一致（如均为 cn-beijing） |
| `LLM Judge 降级` | Ark API 不可用 | 检查网络和 API Key，或设置 `skip_llm_judge=True` |
