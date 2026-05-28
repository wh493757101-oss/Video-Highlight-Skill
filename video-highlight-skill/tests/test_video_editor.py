from pathlib import Path

import pytest

from src.video_editor import EditResult, EditorConfig, VideoEditor


class TestEditorConfig:
    def test_defaults(self):
        cfg = EditorConfig()
        assert cfg.output_dir == ""
        assert cfg.concat_list_filename == "concat_list.txt"

    def test_custom(self):
        cfg = EditorConfig(
            output_dir="/tmp/out",
            concat_list_filename="my_list.txt",
        )
        assert cfg.output_dir == "/tmp/out"
        assert cfg.concat_list_filename == "my_list.txt"


class TestEditResult:
    def test_defaults(self):
        result = EditResult(output_path="/tmp/out.mp4")
        assert result.output_path == "/tmp/out.mp4"
        assert result.segments == []
        assert result.source == "multimodal"

    def test_with_segments(self):
        seg_info = [{"start_time": 1.0, "end_time": 3.0, "score": 0.9}]
        result = EditResult(
            output_path="/tmp/out.mp4",
            segments=seg_info,
            source="multimodal",
        )
        assert result.source == "multimodal"
        assert len(result.segments) == 1


class TestVideoEditorFFmpeg:
    def test_edit_with_ffmpeg_success(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        editor = VideoEditor(EditorConfig(output_dir=str(tmp_path)))

        segments = [
            {"start_time": 0.0, "end_time": 5.0, "score": 0.9},
            {"start_time": 10.0, "end_time": 15.0, "score": 0.7},
        ]

        result = editor.edit_with_ffmpeg("/tmp/video.mp4", segments)

        assert result.source == "multimodal"
        assert len(result.segments) == 2
        assert result.output_path.startswith(str(tmp_path))
        assert "highlight_reel_" in result.output_path
        assert mock_run.call_count >= 3  # 2 cuts + 1 concat

    def test_edit_with_ffmpeg_empty_segments_raises(self):
        editor = VideoEditor()
        with pytest.raises(ValueError, match="segments 为空"):
            editor.edit_with_ffmpeg("/tmp/video.mp4", [])

    def test_edit_with_ffmpeg_single_segment(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        editor = VideoEditor(EditorConfig(output_dir=str(tmp_path)))

        segments = [{"start_time": 5.0, "end_time": 10.0, "score": 0.8}]

        result = editor.edit_with_ffmpeg("/tmp/video.mp4", segments)

        assert result.source == "multimodal"
        assert len(result.segments) == 1
        assert mock_run.call_count >= 2  # 1 cut + 1 concat

    def test_edit_with_ffmpeg_cleanup_temp_files(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        editor = VideoEditor(EditorConfig(output_dir=str(tmp_path)))

        segments = [{"start_time": 0.0, "end_time": 3.0, "score": 0.9}]

        editor.edit_with_ffmpeg("/tmp/video.mp4", segments)

        # Temp clip files should be cleaned up
        clip_files = list(Path(tmp_path).glob("_clip_*.mp4"))
        assert len(clip_files) == 0
