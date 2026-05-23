import json
from pathlib import Path

import numpy as np
import pytest

from src.highlight_detector import DetectionResult
from src.main import DegradationRecord, PipelineConfig, PipelineResult, VideoHighlightPipeline
from src.rule_engine import HighlightSegment
from src.video_editor import EditResult
from src.video_fetcher import LocalFileSource, UrlSource, VideoMetadata


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.output_dir == ""
        assert cfg.detector.fallback_enabled is True
        assert cfg.editor.fallback_enabled is True

    def test_custom(self):
        cfg = PipelineConfig(output_dir="/tmp/pipeline")
        assert cfg.output_dir == "/tmp/pipeline"


class TestPipelineResult:
    def test_with_edit(self):
        metadata = VideoMetadata(
            path="/tmp/v.mp4", duration=10.0, fps=30.0, width=1920, height=1080
        )
        detection = DetectionResult(source="rule")
        edit = EditResult(output_path="/tmp/out.mp4", source="ffmpeg")
        result = PipelineResult(metadata=metadata, detection=detection, edit=edit)
        assert result.edit is not None
        assert result.edit.output_path == "/tmp/out.mp4"
        assert result.error is None

    def test_with_error(self):
        metadata = VideoMetadata(
            path="/tmp/v.mp4", duration=10.0, fps=30.0, width=1920, height=1080
        )
        detection = DetectionResult(source="rule")
        result = PipelineResult(
            metadata=metadata, detection=detection, error="未检测到高光片段"
        )
        assert result.error == "未检测到高光片段"
        assert result.edit is None


class TestVideoHighlightPipeline:
    def _make_metadata(self, tmp_path, duration=10.0, fps=30.0):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(5):
            (frames_dir / f"frame_{i:06d}.jpg").touch()

        audio_path = str(tmp_path / "audio.wav")
        Path(audio_path).touch()

        return VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=duration,
            fps=fps,
            width=1920,
            height=1080,
            audio_path=audio_path,
            frames_dir=str(frames_dir),
        )

    def test_run_full_pipeline(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "editor",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata
        pipeline.detector.detect.return_value = DetectionResult(
            segments=[
                HighlightSegment(start_time=1.0, end_time=4.0, combined_score=0.9),
                HighlightSegment(start_time=6.0, end_time=9.0, combined_score=0.7),
            ],
            source="multimodal",
        )
        pipeline.editor.edit.return_value = EditResult(
            output_path=str(tmp_path / "highlight_reel.mp4"),
            source="las",
        )

        result = pipeline.run(
            LocalFileSource(str(tmp_path / "video.mp4")),
            description="剪辑精彩片段",
        )

        assert result.detection.source == "multimodal"
        assert len(result.detection.segments) == 2
        assert result.edit is not None
        assert result.edit.source == "las"
        assert result.error is None

    def test_run_no_segments(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata
        pipeline.detector.detect.return_value = DetectionResult(source="rule")

        result = pipeline.run(LocalFileSource(str(tmp_path / "video.mp4")))
        assert result.error == "未检测到高光片段"
        assert result.edit is None

    def test_run_skip_edit(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata
        pipeline.detector.detect.return_value = DetectionResult(
            segments=[HighlightSegment(start_time=1.0, end_time=3.0, combined_score=0.8)],
            source="rule",
        )

        result = pipeline.run(
            LocalFileSource(str(tmp_path / "video.mp4")),
            skip_edit=True,
        )

        assert result.edit is None
        assert result.error is None
        assert len(result.detection.segments) == 1

    def test_run_from_path(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata
        pipeline.detector.detect.return_value = DetectionResult(source="rule")

        result = pipeline.run_from_path("/tmp/video.mp4")
        assert result.detection.source == "rule"

    def test_run_from_url(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata
        pipeline.detector.detect.return_value = DetectionResult(source="rule")

        result = pipeline.run_from_url("https://example.com/video.mp4")
        assert result.detection.source == "rule"

    def test_format_result_basic(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=15.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        detection = DetectionResult(
            segments=[
                HighlightSegment(start_time=2.0, end_time=5.0, combined_score=0.9),
                HighlightSegment(start_time=8.0, end_time=12.0, combined_score=0.7),
            ],
            source="multimodal",
        )
        edit = EditResult(
            output_path=str(tmp_path / "highlight_reel.mp4"),
            source="las",
        )
        result = PipelineResult(metadata=metadata, detection=detection, edit=edit)

        pipeline = VideoHighlightPipeline()
        formatted = pipeline.format_result(result)

        assert "视频高光剪辑" in formatted
        assert "15.0s" in formatted
        assert "1920x1080" in formatted
        assert "multimodal" in formatted
        assert "#1: 2.0s - 5.0s" in formatted
        assert "#2: 8.0s - 12.0s" in formatted
        assert "las" in formatted

    def test_format_result_with_degradations(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=15.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        detection = DetectionResult(
            segments=[HighlightSegment(start_time=2.0, end_time=5.0, combined_score=0.9)],
            source="rule",
            degraded=True,
            degradation_reason="Ark API 超时",
        )
        edit = EditResult(
            output_path=str(tmp_path / "highlight_reel.mp4"),
            source="ffmpeg",
            degraded=True,
            degradation_reason="LAS 不可用",
        )
        result = PipelineResult(
            metadata=metadata,
            detection=detection,
            edit=edit,
            degradations=[
                DegradationRecord(
                    stage="高光检测",
                    from_path="Ark 多模态 API",
                    to_path="规则引擎（librosa + OpenCV）",
                    reason="Ark API 超时",
                ),
                DegradationRecord(
                    stage="视频剪辑",
                    from_path="LAS las_video_edit 云端算子",
                    to_path="FFmpeg 本地剪辑",
                    reason="LAS 不可用",
                ),
            ],
        )

        pipeline = VideoHighlightPipeline()
        formatted = pipeline.format_result(result)

        assert "[降级说明]" in formatted
        assert "Ark 多模态 API" in formatted
        assert "规则引擎" in formatted
        assert "LAS las_video_edit" in formatted
        assert "FFmpeg" in formatted
        assert "Ark API 超时" in formatted
        assert "LAS 不可用" in formatted

    def test_format_result_with_error(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=10.0,
            fps=24.0,
            width=1280,
            height=720,
        )
        detection = DetectionResult(source="rule")
        result = PipelineResult(
            metadata=metadata, detection=detection, error="处理失败"
        )

        pipeline = VideoHighlightPipeline()
        formatted = pipeline.format_result(result)

        assert "[警告] 处理失败" in formatted
        assert "rule" in formatted

    def test_export_json_basic(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=10.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        detection = DetectionResult(
            segments=[
                HighlightSegment(start_time=1.0, end_time=3.0, combined_score=0.9),
            ],
            source="multimodal",
        )
        edit = EditResult(
            output_path=str(tmp_path / "out.mp4"),
            source="las",
        )
        result = PipelineResult(metadata=metadata, detection=detection, edit=edit)

        pipeline = VideoHighlightPipeline()
        exported = pipeline.export_json(result)
        data = json.loads(exported)

        assert data["video"]["duration"] == 10.0
        assert data["video"]["width"] == 1920
        assert data["detection"]["source"] == "multimodal"
        assert len(data["detection"]["segments"]) == 1
        assert data["detection"]["segments"][0]["score"] == 0.9
        assert data["edit"]["source"] == "las"

    def test_export_json_with_degradations(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=10.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        detection = DetectionResult(
            segments=[HighlightSegment(start_time=1.0, end_time=3.0, combined_score=0.9)],
            source="rule",
            degraded=True,
            degradation_reason="Ark API 超时",
        )
        edit = EditResult(
            output_path=str(tmp_path / "out.mp4"),
            source="ffmpeg",
            degraded=True,
            degradation_reason="LAS 不可用",
        )
        result = PipelineResult(
            metadata=metadata,
            detection=detection,
            edit=edit,
            degradations=[
                DegradationRecord(
                    stage="高光检测",
                    from_path="Ark 多模态 API",
                    to_path="规则引擎（librosa + OpenCV）",
                    reason="Ark API 超时",
                ),
                DegradationRecord(
                    stage="视频剪辑",
                    from_path="LAS las_video_edit 云端算子",
                    to_path="FFmpeg 本地剪辑",
                    reason="LAS 不可用",
                ),
            ],
        )

        pipeline = VideoHighlightPipeline()
        exported = pipeline.export_json(result)
        data = json.loads(exported)

        assert "degradations" in data
        assert len(data["degradations"]) == 2
        assert data["degradations"][0]["stage"] == "高光检测"
        assert data["degradations"][0]["reason"] == "Ark API 超时"
        assert data["degradations"][1]["stage"] == "视频剪辑"
        assert data["degradations"][1]["reason"] == "LAS 不可用"

    def test_export_json_with_error(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=5.0,
            fps=25.0,
            width=640,
            height=480,
        )
        detection = DetectionResult(source="rule")
        result = PipelineResult(
            metadata=metadata, detection=detection, error="未检测到高光片段"
        )

        pipeline = VideoHighlightPipeline()
        exported = pipeline.export_json(result)
        data = json.loads(exported)

        assert data["error"] == "未检测到高光片段"
        assert "edit" not in data

    def test_lazy_init_properties(self):
        pipeline = VideoHighlightPipeline()
        assert pipeline._fetcher is None
        assert pipeline._detector is None
        assert pipeline._editor is None

    def test_run_fetch_exception_returns_error(self, mocker):
        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.side_effect = FileNotFoundError("视频文件不存在")

        result = pipeline.run(LocalFileSource("/nonexistent/video.mp4"))
        assert result.error is not None
        assert "视频文件不存在" in result.error

    def test_run_detection_exception_returns_error(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata
        pipeline.detector.detect.side_effect = RuntimeError("模型调用失败")

        result = pipeline.run(LocalFileSource(str(tmp_path / "video.mp4")))
        assert result.error is not None
        assert "模型调用失败" in result.error

    def test_run_editor_exception_returns_error(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "editor",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata
        pipeline.detector.detect.return_value = DetectionResult(
            segments=[HighlightSegment(start_time=1.0, end_time=3.0, combined_score=0.8)],
            source="rule",
        )
        pipeline.editor.edit.side_effect = RuntimeError("剪辑失败")

        result = pipeline.run(LocalFileSource(str(tmp_path / "video.mp4")))
        assert result.error is not None
        assert "剪辑失败" in result.error
