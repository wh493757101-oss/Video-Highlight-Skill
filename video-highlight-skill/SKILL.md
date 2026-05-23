---
name: video-highlight
description: >
  视频高光剪辑 — 输入长视频+剪辑指令，自动识别高光片段并输出集锦视频。
  触发词: 视频高光、高光剪辑、精彩集锦、视频剪辑、剪辑视频、highlight、highlight reel、video highlight。
  使用场景: 用户需要从长视频中提取精彩片段、制作集锦、高光时刻剪辑。
version: 1.0.0
metadata:
  openclaw:
    requires:
      env:
        - ARK_API_KEY
      bins:
        - ffmpeg
    primaryEnv: ARK_API_KEY
    env:
      - name: ARK_API_KEY
        description: "火山方舟 Ark API Key，用于多模态高光检测"
        required: true
      - name: LAS_API_KEY
        description: "LAS 算子 API Key，用于云端视频剪辑"
        required: false
      - name: TOS_ENDPOINT
        description: "TOS 对象存储 Endpoint（TOS 路径时必需）"
        required: false
      - name: TOS_ACCESS_KEY
        description: "TOS Access Key（TOS 路径时必需）"
        required: false
      - name: TOS_SECRET_KEY
        description: "TOS Secret Key（TOS 路径时必需）"
        required: false
---

# 视频高光剪辑 Skill

基于火山引擎 Ark API（多模态理解）和 LAS 算子（智能剪辑），自动识别视频中的高光片段并生成集锦视频。

## 输入输出

### 输入

| 字段 | 说明 | 必需 |
|------|------|------|
| 原始视频 | 长视频文件（本地路径 / URL / TOS 路径） | 是 |
| 剪辑目标 | 如"精彩集锦""进球片段""高能时刻" | 否 |
| 时长要求 | 如"3 分钟内""不超过 30 秒" | 否 |
| 风格/主题 | 如"快节奏""电影感""卡点混剪" | 否 |
| 剪辑模式 | `fast`（快速，仅视觉特征）或 `full`（完整，多模态+ASR），默认 `full` | 否 |

### 输出

| 字段 | 说明 |
|------|------|
| 高光视频 | 剪辑后的集锦视频（mp4, H.264 + AAC） |
| 时间戳说明 | 每个片段的起止时间、精彩度评分、标签 |
| JSON 导出 | 结构化片段数据，含时间戳和评分 |
| Token 用量 | 多模态检测消耗的 token 数（prompt_tokens + completion_tokens） |

## 核心能力

1. **高光识别** — 自动检测视频中的精彩片段（精彩动作、关键场景、情绪爆发、转场亮点）
2. **多模态筛选** — 结合画面变化、音频特征（音量/语速/静音）、语义理解、ASR 字幕等多维线索定位高光片段
3. **智能剪辑** — 片段裁剪 + 转场拼接，支持 LAS 云端剪辑和 FFmpeg 本地降级
4. **结构化输出** — 输出最终高光视频，附带关键时间戳和片段说明

## 触发条件

当用户消息包含以下关键词时触发：
- 视频高光、高光剪辑、精彩集锦、视频剪辑、剪辑视频
- highlight、highlight reel、video highlight

## 工作流（严格按步骤执行）

复制此清单并跟踪进度：

```
执行进度：
- [ ] Step 0: 前置检查
- [ ] Step 1: 视频获取与预处理
- [ ] Step 2: Token 预估（⚠️ 多模态路径必须告知用户）
- [ ] Step 3: 高光检测
- [ ] Step 4: 视频剪辑
- [ ] Step 5: 结果呈现
- [ ] Step 6: 失败排查（仅在出错时）
```

### Step 0: 前置检查（⚠️ 必须在第一轮对话中完成）

在接受用户任务后，不要立即开始执行，必须首先进行以下检查：

**环境变量检查：**
- 确认 `ARK_API_KEY` 已配置（多模态路径必需）
- 如用户要求 LAS 剪辑，确认 `LAS_API_KEY` 已配置
- 如输入为 TOS 路径，确认 `TOS_ENDPOINT` / `TOS_ACCESS_KEY` / `TOS_SECRET_KEY` 均已配置
- 若无，必须立即向用户索要

**输入来源检查：**
- 飞书上传视频：确认文件已通过飞书消息接收，格式为常见视频格式（mp4/mov/avi 等）
- URL 链接：确认链接可访问，支持 yt-dlp 兼容平台（YouTube、B站等）
- 评测模式：确认 TOS 中测试用例路径有效，`TOS_ENDPOINT` / `TOS_ACCESS_KEY` / `TOS_SECRET_KEY` 已配置

**输出路径检查：**
- 飞书返回：输出视频直接通过飞书消息回复用户（文件或 TOS 下载链接）
- LAS 剪辑时，`output_tos_path` 必须为 `tos://` 前缀的目录（不能以文件名结尾）
- FFmpeg 降级时，确认本地磁盘空间充足（建议预留视频大小 3x 的空间）

**Region 一致性检查：**
- LAS Region 必须与 TOS Bucket 所在区域一致，否则会导致权限异常

### Step 1: 视频获取与预处理

根据来源类型下载/加载视频：

- **本地文件**: 直接加载，非 mp4 格式通过 FFmpeg 转码
- **URL**: 使用 yt-dlp 下载（默认 600 秒超时）
- **TOS 路径**: 使用 boto3/S3 协议下载

预处理步骤：
- 格式统一为 mp4
- 提取音频（16kHz 单声道 wav，无音频流时跳过）
- 关键帧采样（默认每 2 秒一帧，最多取 16 帧送多模态）

### Step 2: Token 预估（⚠️ 多模态路径必须告知用户）

多模态检测会发送图像帧到 Ark API，token 消耗取决于帧数和图片分辨率：

- **预估公式**: 每帧约 500-1500 tokens（取决于分辨率），加上 prompt 文本约 500 tokens
- **示例**: 16 帧 + prompt ≈ 8,500-24,500 tokens（输入），输出约 500-2000 tokens
- **告知用户**: 预估 token 消耗量，提示"实际以 API 返回的 usage 为准"
- **用户确认**: 如果预估 token 量较大（>50,000 tokens），必须等待用户确认后再继续

> 规则引擎降级路径不消耗 API tokens，但精度较低。

### Step 3: 高光检测

**主路径 — 多模态理解（Ark API）:**
- 将关键帧序列 + ASR 文本发送给豆包多模态模型
- 模型分析画面变化、内容精彩度、节奏感
- 返回结构化 JSON（时间戳 + 标签 + 精彩度评分）
- 记录实际 token 用量（从 `usage` 字段提取）

**降级路径 — 规则引擎:**
- Ark API 不可用时自动切换
- 音频分析：音量峰值、语速变化、静音段检测（librosa）
- 画面分析：场景切换检测、运动强度、亮度变化（OpenCV）
- 综合打分 → Top-K 片段输出

### Step 4: 视频剪辑

**主路径 — LAS `las_video_edit` 算子:**
- 组装 task_description（用户需求 + 片段列表 + 转场要求）
- 提交 LAS 异步任务 → 轮询等待完成（默认 600 秒超时）
- 获取剪辑结果（TOS 输出路径）

**降级路径 — FFmpeg 本地剪辑:**
- LAS 不可用时自动切换
- 逐片段裁剪 → concat 拼接 → 输出集锦视频

### Step 5: 结果呈现

向用户展示：

1. **视频信息**: 原视频时长、分辨率、帧率
2. **检测摘要**: 检测方式（多模态/规则引擎）、片段数量、Token 用量
3. **片段列表**: 每个片段的起止时间、精彩度评分、标签
4. **输出视频**: 集锦视频文件路径或下载链接
5. **JSON 导出**: 结构化片段数据

### Step 6: 失败排查（仅在出错时）

按以下顺序排查：

1. **检查环境变量**: 确认 `ARK_API_KEY` / `LAS_API_KEY` 已正确配置
2. **检查输入可访问性**: 确认视频文件存在、URL 可访问、TOS 凭证有效
3. **检查任务状态**: LAS 任务可通过 `las_cli task status <task_id>` 查询
4. **检查 FFmpeg**: 确认 FFmpeg 已安装且版本兼容
5. **检查磁盘空间**: FFmpeg 降级路径需要足够磁盘空间
6. **检查日志**: 查看 Pipeline 日志定位具体错误

## 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `ARK_API_KEY` | 火山方舟 Ark API Key | 是（多模态路径） |
| `LAS_API_KEY` | LAS 算子 API Key | 否（FFmpeg 可降级） |
| `TOS_ENDPOINT` | TOS 对象存储 Endpoint | 否（TOS 路径时必需） |
| `TOS_ACCESS_KEY` | TOS Access Key | 否（TOS 路径时必需） |
| `TOS_SECRET_KEY` | TOS Secret Key | 否（TOS 路径时必需） |

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

## 评测体系

本 Skill 包含完整的评测框架，支持自动化质量评估：

- **量化评测** (`evaluator.py`) — 基于 tIoU 的片段匹配，计算 Precision/Recall/F1/Hit Rate/MAE
- **LLM Judge** (`llm_judge.py`) — 多维度主观评分（节奏感、完整性、精彩度、指令契合度）
- **评测编排** (`runner.py`) — 自动加载用例 → 运行 Pipeline → 并行评测
- **报告生成** (`report.py`) — 文本报告 + JSON 导出 + 可视化图表

测试用例基于 SumMe 数据集（35 组本地）和自建 URL 用例（10 组远程），覆盖旅行/体育/户外/生活/边界场景。

## 审查标准

执行完成后，Agent 应自检：

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

## Gotchas

- **长视频处理**: 超过 30 分钟的视频建议先分片处理，避免单次多模态调用上下文溢出。多模态路径最多发送 16 帧，长视频会被重度降采样
- **URL 下载**: 使用 yt-dlp 下载，支持主流平台，默认 600 秒超时。部分平台可能有访问限制
- **TOS 下载**: 需要安装 `boto3`（`pip install boto3`），并配置 `TOS_ENDPOINT`/`TOS_ACCESS_KEY`/`TOS_SECRET_KEY` 环境变量
- **LAS 轮询超时**: 默认 600 秒超时，大文件可能需要更长时间
- **FFmpeg 降级**: 降级路径使用本地计算资源，大视频注意磁盘空间。LAS 不可用时自动切换
- **音频提取失败**: 视频无音频流时 `_extract_audio` 返回 None，Pipeline 继续运行但仅使用视觉特征检测高光
- **关键帧采样为空**: 视频无法解码或帧数过少时，多模态检测会降级到规则引擎
- **空文件/无效视频**: 0 字节文件或非视频文件会抛出明确错误（`ValueError` / `RuntimeError`）
- **关键帧采样间隔**: 默认 2 秒，可通过 `DetectorConfig.frame_interval` 调整
- **API Key 安全**: 不要将 API Key 硬编码在代码中，使用环境变量或 `.env` 文件
- **输出路径格式**: LAS 剪辑时 `output_tos_path` 必须是 `tos://` 前缀的目录，服务端自动创建片段文件
- **Region 一致性**: LAS Region 必须与 TOS Bucket 区域一致，否则会导致权限异常或上传失败
