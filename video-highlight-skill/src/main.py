import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .highlight_detector import DetectorConfig, DetectionResult, HighlightDetector
from .video_editor import EditResult, EditorConfig, VideoEditor
from .video_fetcher import (
    LocalFileSource,
    UrlSource,
    VideoFetcher,
    VideoMetadata,
    VideoSource,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)
    output_dir: str = ""


@dataclass
class PipelineResult:
    metadata: VideoMetadata
    detection: DetectionResult
    edit: EditResult | None = None
    error: str | None = None
    session_dir: str = ""


class VideoHighlightPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._fetcher: VideoFetcher | None = None
        self._detector: HighlightDetector | None = None
        self._editor: VideoEditor | None = None

    @property
    def fetcher(self) -> VideoFetcher:
        if self._fetcher is None:
            self._fetcher = VideoFetcher(output_dir=self.config.output_dir or None)
        return self._fetcher

    @property
    def detector(self) -> HighlightDetector:
        if self._detector is None:
            self._detector = HighlightDetector(self.config.detector)
        return self._detector

    @property
    def editor(self) -> VideoEditor:
        if self._editor is None:
            self._editor = VideoEditor(self.config.editor)
        return self._editor

    def run(
        self,
        source: VideoSource,
        description: str = "",
        asr_text: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        try:
            return self._run_impl(source, description, asr_text, skip_edit)
        except Exception as e:
            logger.error("Pipeline 执行失败: %s", e)
            return PipelineResult(
                metadata=VideoMetadata(path="", duration=0, fps=0, width=0, height=0),
                detection=DetectionResult(source="error"),
                error=f"处理失败，请稍后重试: {e}",
                session_dir="",
            )

    def _make_session_dir(self, video_path: str) -> str:
        base = Path(self.config.output_dir) if self.config.output_dir else Path.cwd() / "output"
        video_name = Path(video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = base / f"{video_name}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    def _run_impl(
        self,
        source: VideoSource,
        description: str = "",
        asr_text: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        metadata = self.fetcher.fetch(source)
        logger.info("视频预处理完成: duration=%.1fs, fps=%.1f", metadata.duration, metadata.fps)

        session_dir = self._make_session_dir(metadata.path)

        detection = self.detector.detect(metadata, asr_text=asr_text)
        logger.info(
            "高光检测完成: source=%s, segments=%d",
            detection.source,
            len(detection.segments),
        )

        if not detection.segments:
            return PipelineResult(
                metadata=metadata, detection=detection,
                error="未检测到高光片段", session_dir=session_dir,
            )

        if skip_edit:
            return PipelineResult(
                metadata=metadata, detection=detection, session_dir=session_dir,
            )

        self.config.editor.output_dir = session_dir
        edit = self.editor.edit(metadata.path, detection.segments, description)
        logger.info("剪辑完成: source=%s, output=%s", edit.source, edit.output_path)

        json_path = Path(session_dir) / "result.json"
        json_path.write_text(self.export_json(
            PipelineResult(metadata=metadata, detection=detection, edit=edit)
        ), encoding="utf-8")

        return PipelineResult(
            metadata=metadata, detection=detection, edit=edit, session_dir=session_dir,
        )

    def run_from_path(
        self,
        video_path: str,
        description: str = "",
        asr_text: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        return self.run(LocalFileSource(video_path), description, asr_text, skip_edit)

    def run_from_url(
        self,
        url: str,
        description: str = "",
        asr_text: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        return self.run(UrlSource(url), description, asr_text, skip_edit)

    def format_result(self, result: PipelineResult) -> str:
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("视频高光剪辑 — 处理结果")
        lines.append("=" * 60)

        if result.session_dir:
            lines.append(f"\n[输出目录] {result.session_dir}")

        lines.append("\n[视频信息]")
        lines.append(f"  文件: {result.metadata.path}")
        lines.append(f"  时长: {result.metadata.duration:.1f}s")
        lines.append(f"  分辨率: {result.metadata.width}x{result.metadata.height}")
        lines.append(f"  帧率: {result.metadata.fps:.1f} fps")

        lines.append("\n[高光检测]")
        lines.append(f"  检测方式: {result.detection.source}")
        lines.append(f"  高光片段数: {len(result.detection.segments)}")

        for i, seg in enumerate(result.detection.segments):
            lines.append(
                f"  #{i + 1}: {seg.start_time:.1f}s - {seg.end_time:.1f}s"
                f" (精彩度: {seg.combined_score:.2f})"
            )

        if result.edit:
            lines.append("\n[剪辑输出]")
            lines.append(f"  剪辑方式: {result.edit.source}")
            lines.append(f"  输出路径: {result.edit.output_path}")

        if result.error:
            lines.append(f"\n[警告] {result.error}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def export_json(self, result: PipelineResult) -> str:
        segments_data = [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "score": seg.combined_score,
            }
            for seg in result.detection.segments
        ]

        output: dict[str, Any] = {
            "video": {
                "path": result.metadata.path,
                "duration": result.metadata.duration,
                "fps": result.metadata.fps,
                "width": result.metadata.width,
                "height": result.metadata.height,
            },
            "detection": {
                "source": result.detection.source,
                "segments": segments_data,
            },
        }

        if result.edit:
            output["edit"] = {
                "source": result.edit.source,
                "output_path": result.edit.output_path,
            }

        if result.error:
            output["error"] = result.error

        if result.session_dir:
            output["session_dir"] = result.session_dir

        return json.dumps(output, ensure_ascii=False, indent=2)
