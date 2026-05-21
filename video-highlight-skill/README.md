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

| 输出 | 说明 |
|------|------|
| 高光视频 | 剪辑后的集锦视频（mp4, H.264 + AAC） |
| 时间戳说明 | 每个片段的起止时间、精彩度评分、标签 |
| JSON 导出 | 结构化片段数据，含时间戳和评分 |

## 架构

```
用户输入（视频 + 指令）
  → VideoFetcher（下载/预处理/抽帧/提取音频）
    → HighlightDetector（多模态 Ark API 主路径 / 规则引擎降级）
      → VideoEditor（LAS las_video_edit 主路径 / FFmpeg 降级）
        → 输出（集锦视频 + 时间戳 + JSON）
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
export ARK_API_KEY="your-ark-api-key"    # 火山方舟 Ark API Key
export LAS_API_KEY="your-las-api-key"    # LAS 算子 API Key
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
│   └── test_cases/
│       ├── local/              # 20 组本地视频用例
│       └── remote/             # 10 组 URL 视频用例
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
# 1. 放入视频文件到 evaluation/test_cases/local/case_XXX/
# 2. 填充 ground_truth.json（高光时间戳标注）
# 3. 填充 remote/cases.yaml 中的 source_url
# 4. 运行评测
python -c "
from evaluation.evaluator import TestCaseLoader, HighlightEvaluator
from evaluation.report import ReportGenerator, ReportConfig

loader = TestCaseLoader('evaluation/test_cases')
cases = loader.load_all()
# ... 对每个 case 跑 pipeline，收集 predicted 结果 ...
evaluator = HighlightEvaluator()
report = evaluator.evaluate_all(results)
gen = ReportGenerator(ReportConfig(output_dir='reports'))
gen.generate(report, judge_report)
"
```

## 降级策略

| 环节 | 主路径 | 降级路径 |
|------|--------|----------|
| 高光检测 | Ark 多模态 API | 规则引擎（librosa + OpenCV） |
| 视频剪辑 | LAS las_video_edit | FFmpeg 本地裁剪拼接 |

降级自动触发，无需手动切换。可通过配置关闭降级：

```python
from src.highlight_detector import DetectorConfig
from src.video_editor import EditorConfig

detector_cfg = DetectorConfig(fallback_enabled=False)  # Ark 失败直接抛异常
editor_cfg = EditorConfig(fallback_enabled=False)      # LAS 失败直接抛异常
```
