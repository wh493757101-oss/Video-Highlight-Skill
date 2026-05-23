# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

视频高光剪辑 Skill — a video highlight detection and editing pipeline built on Volcano Engine's Ark API (multimodal understanding) and LAS operators (cloud video editing). Input a long video + natural language instructions, output a highlight reel with timestamp annotations.

## Commands

```bash
# Run all tests
cd video-highlight-skill && python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_highlight_detector.py -v

# Run with coverage (requires pytest-cov)
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run evaluation pipeline
python -m evaluation.runner
```

No build step — pure Python project with pyproject.toml. Install dependencies with `pip install -e ".[dev]"` from the `video-highlight-skill/` directory.

## Architecture

The pipeline flows through 5 stages, each with a primary path and a degradation fallback:

```
VideoSource → VideoFetcher → HighlightDetector → VideoEditor → PipelineResult
   (resolve)    (preprocess)    (detect)          (edit)        (format/export)
```

**Core modules (src/):**

- `main.py` — `VideoHighlightPipeline` orchestrates the full flow. Lazy-initializes fetcher/detector/editor. Entry points: `run(source, description, asr_text)`, `run_from_path()`, `run_from_url()`.
- `video_fetcher.py` — `VideoFetcher` handles 3 source types: `LocalFileSource`, `UrlSource` (yt-dlp), `TosSource` (S3-compatible). Preprocessing: format conversion to mp4, audio extraction (16kHz mono wav), keyframe sampling (default 2s interval).
- `highlight_detector.py` — `HighlightDetector` tries multimodal detection (Ark API with image frames + prompt → structured JSON segments) first, falls back to `RuleEngine` (librosa audio analysis + OpenCV visual analysis). Controlled by `DetectorConfig.fallback_enabled`.
- `video_editor.py` — `VideoEditor` tries LAS `las_video_edit` operator first, falls back to FFmpeg (per-segment trim + concat). Controlled by `EditorConfig.fallback_enabled`.
- `ark_client.py` — `ArkClient` wraps Volcano Engine Ark Chat Completion API. Supports text + image multimodal input via `chat_with_images()`. Handles retries with exponential backoff on 429s.
- `las_client.py` — `LasClient` wraps LAS operator API (submit → poll → wait_for_completion). 600s default timeout.
- `rule_engine.py` — Signal-processing fallback: `AudioAnalyzer` (RMS energy + onset strength + zero-crossing rate) and `VisualAnalyzer` (frame diff motion + brightness + contrast). Combined scoring → percentile threshold → segment extraction → merge → top-K.

**Evaluation framework (evaluation/):**

- `evaluator.py` — tIoU-based segment matching: Precision/Recall/F1/Hit Rate/MAE. `TestCaseLoader` reads YAML case manifests from `test_cases/`.
- `llm_judge.py` — LLM-as-Judge for subjective scoring (rhythm, completeness, excitement, instruction fit) on 1-5 scale.
- `runner.py` — `EvalRunner` loads cases, runs pipeline per case, evaluates in parallel (quantitative + LLM judge via ThreadPoolExecutor).
- `report.py` — Text report + JSON export + matplotlib charts.

**Test cases (evaluation/test_cases/):**

- `open_data/` — 35 local test cases from SumMe dataset (travel/sports/outdoor/lifestyle/edge cases). Each case: `video.mp4`, `instruction.json`, `ground_truth.json`, `metadata.yaml`.
- `self-built_data/` — 10 remote URL test cases (sports/gaming/speech/variety/documentary). URLs in `source_url` field.
- Case manifests in `cases.yaml` define id, category, difficulty, description, and instruction per case.

## Key Design Decisions

- **Degradation over failure**: Both detection and editing have automatic fallback paths. The pipeline never crashes on API unavailability — it degrades gracefully.
- **Lazy initialization**: Pipeline components use `@property` with `_x is None` checks, avoiding unnecessary API client construction.
- **Dataclass-driven config**: Each module has a `*Config` dataclass with defaults. Override by passing instances to constructors.
- **Two evaluation dimensions**: Quantitative (tIoU matching against ground truth) and qualitative (LLM Judge). Weighted combination via `compute_weighted_score()`.
- **Local and remote video paths**: `VideoSource` Protocol with `resolve()` → local path. `UrlSource` downloads via yt-dlp, `TosSource` via boto3/S3.

## Environment Variables

| Variable | Required | Used By |
|----------|----------|---------|
| `ARK_API_KEY` | Yes (multimodal path) | `ArkClient` |
| `LAS_API_KEY` | No (FFmpeg fallback) | `LasClient` |
| `TOS_ENDPOINT` | No (TOS path only) | `TosSource` |
| `TOS_ACCESS_KEY` | No (TOS path only) | `TosSource` |
| `TOS_SECRET_KEY` | No (TOS path only) | `TosSource` |

## External Dependencies

- **ffmpeg** — Required on PATH for video conversion, audio extraction, and FFmpeg fallback editing
- **yt-dlp** — For URL video downloads
- **boto3** — Optional, only needed for TOS (S3-compatible) downloads
