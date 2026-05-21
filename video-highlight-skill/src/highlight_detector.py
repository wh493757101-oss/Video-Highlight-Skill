import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ark_client import ArkClient, ArkConfig
from .rule_engine import HighlightSegment, RuleEngine, RuleEngineConfig
from .video_fetcher import VideoFetcher, VideoMetadata

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    frame_interval: float = 2.0
    max_frames_per_batch: int = 16
    ark_model: str = "doubao-seed-2-0-pro"
    ark_temperature: float = 0.3
    ark_max_tokens: int = 4096
    fallback_enabled: bool = True


HIGHLIGHT_PROMPT = """你是一个专业的视频高光检测分析器。请分析以下视频帧序列（按时间顺序排列，每帧间隔约 {interval} 秒），识别出最精彩的高光片段。

视频总时长: {duration} 秒，共 {frame_count} 帧。

请根据以下维度判断高光片段：
1. 画面变化：场景切换、运动强度、视觉冲击力
2. 内容精彩度：关键动作、表情、事件
3. 节奏感：画面变化的频率和幅度

请以 JSON 格式返回结果，包含一个 segments 数组，每个元素包含：
- start_time: 起始时间（秒）
- end_time: 结束时间（秒）
- label: 高光类型标签（如 "精彩动作", "关键场景", "情绪爆发", "转场亮点"）
- score: 精彩度评分（0.0-1.0）
- reason: 简短理由（一句话）

{asr_context}

只返回 JSON，不要包含其他文字。"""


@dataclass
class DetectionResult:
    segments: list[HighlightSegment] = field(default_factory=list)
    source: str = "rule"
    raw_response: dict[str, Any] | None = None


class HighlightDetector:
    def __init__(
        self,
        config: DetectorConfig | None = None,
        ark_client: ArkClient | None = None,
        rule_engine: RuleEngine | None = None,
    ):
        self.config = config or DetectorConfig()
        self._ark_client = ark_client
        self._rule_engine = rule_engine

    @property
    def ark_client(self) -> ArkClient:
        if self._ark_client is None:
            self._ark_client = ArkClient()
        return self._ark_client

    @property
    def rule_engine(self) -> RuleEngine:
        if self._rule_engine is None:
            self._rule_engine = RuleEngine()
        return self._rule_engine

    def detect(
        self,
        metadata: VideoMetadata,
        asr_text: str = "",
    ) -> DetectionResult:
        try:
            return self._detect_multimodal(metadata, asr_text)
        except Exception as e:
            logger.warning("多模态检测失败，降级到规则引擎: %s", e)
            if not self.config.fallback_enabled:
                raise
            return self._detect_rule_based(metadata)

    def _detect_multimodal(
        self, metadata: VideoMetadata, asr_text: str
    ) -> DetectionResult:
        frame_paths = sorted(Path(metadata.frames_dir).glob("*.jpg"))
        if not frame_paths:
            raise ValueError("没有可用的关键帧")

        batch = frame_paths[: self.config.max_frames_per_batch]
        interval = (
            metadata.duration / len(frame_paths) if len(frame_paths) > 1 else self.config.frame_interval
        )

        asr_context = ""
        if asr_text:
            asr_context = f"视频 ASR 文本参考:\n{asr_text[:2000]}\n"

        prompt = HIGHLIGHT_PROMPT.format(
            interval=f"{interval:.1f}",
            duration=f"{metadata.duration:.1f}",
            frame_count=len(batch),
            asr_context=asr_context,
        )

        response = self.ark_client.chat_with_images(
            text=prompt,
            image_paths=[str(p) for p in batch],
            model=self.config.ark_model,
            temperature=self.config.ark_temperature,
            max_tokens=self.config.ark_max_tokens,
        )

        parsed = self.ark_client.extract_json(response)
        segments = self._parse_segments(parsed, metadata.duration)

        return DetectionResult(
            segments=segments,
            source="multimodal",
            raw_response=parsed,
        )

    def _detect_rule_based(self, metadata: VideoMetadata) -> DetectionResult:
        fetcher = VideoFetcher()
        audio = np.array([])
        sr = 22050
        if metadata.audio_path:
            audio, sr = fetcher.load_audio(metadata.audio_path)

        frame_paths = fetcher.load_frames(metadata.frames_dir) if metadata.frames_dir else []

        segments = self.rule_engine.detect(
            audio=audio,
            sr=sr,
            frame_paths=frame_paths,
            fps=metadata.fps,
            duration=metadata.duration,
        )

        return DetectionResult(segments=segments, source="rule")

    def _parse_segments(
        self, parsed: dict[str, Any], duration: float
    ) -> list[HighlightSegment]:
        raw_segments = parsed.get("segments", [])
        if not raw_segments:
            return []

        result: list[HighlightSegment] = []
        for item in raw_segments:
            try:
                seg = HighlightSegment(
                    start_time=float(item.get("start_time", 0)),
                    end_time=float(item.get("end_time", 0)),
                    combined_score=float(item.get("score", 0.5)),
                )
                seg.start_time = max(0.0, min(seg.start_time, duration))
                seg.end_time = max(seg.start_time + 0.5, min(seg.end_time, duration))
                result.append(seg)
            except (ValueError, TypeError):
                continue

        result.sort(key=lambda s: s.combined_score, reverse=True)
        return result
