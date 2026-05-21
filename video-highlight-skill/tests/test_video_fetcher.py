import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video_fetcher import (
    LocalFileSource,
    TosSource,
    UrlSource,
    VideoFetcher,
    VideoMetadata,
)


class TestLocalFileSource:
    def test_resolve_existing_file(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        source = LocalFileSource(str(video))
        assert source.resolve() == str(video.absolute())

    def test_resolve_missing_file_raises(self):
        source = LocalFileSource("/nonexistent/video.mp4")
        with pytest.raises(FileNotFoundError, match="视频文件不存在"):
            source.resolve()


class TestUrlSource:
    def test_resolve_success(self, mocker):
        mock_run = mocker.patch("subprocess.run")
        mocker.patch("tempfile.mkdtemp", return_value="/tmp/video_dl_abc")
        mock_glob = mocker.patch.object(
            Path, "glob", return_value=[Path("/tmp/video_dl_abc/title.mp4")]
        )

        source = UrlSource("https://example.com/video")
        result = source.resolve()

        assert result.replace("\\", "/") == "/tmp/video_dl_abc/title.mp4"
        mock_run.assert_called_once()

    def test_resolve_download_failure(self, mocker):
        mocker.patch("tempfile.mkdtemp", return_value="/tmp/video_dl_abc")
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "yt-dlp", stderr="download error"),
        )

        source = UrlSource("https://example.com/video")
        with pytest.raises(RuntimeError, match="视频下载失败"):
            source.resolve()

    def test_resolve_no_output_file(self, mocker):
        mocker.patch("subprocess.run")
        mocker.patch("tempfile.mkdtemp", return_value="/tmp/video_dl_abc")
        mocker.patch.object(Path, "glob", return_value=[])

        source = UrlSource("https://example.com/video")
        with pytest.raises(RuntimeError, match="未找到输出文件"):
            source.resolve()


class TestTosSource:
    def test_resolve_raises_not_implemented(self):
        source = TosSource("tos://bucket/video.mp4")
        with pytest.raises(NotImplementedError, match="TOS"):
            source.resolve()


class TestVideoFetcher:
    def test_fetch_local_file(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }[prop]
        mock_cap.read.side_effect = (
            [(True, np.zeros((1080, 1920, 3), dtype=np.uint8))] * 5 + [(False, None)]
        )

        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        mocker.patch("cv2.imwrite")
        mock_extract = mocker.patch.object(
            VideoFetcher, "_extract_audio", return_value="/tmp/out/audio/test.wav"
        )

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        meta = fetcher.fetch(LocalFileSource(str(video)))

        assert meta.path == str(video)
        assert meta.duration == 10.0
        assert meta.fps == 30.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.audio_path == "/tmp/out/audio/test.wav"
        assert meta.frames_dir is not None

    def test_fetch_cannot_open_video(self, mocker, tmp_path):
        video = tmp_path / "bad.mp4"
        video.touch()

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = False
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="无法打开视频"):
            fetcher.fetch(LocalFileSource(str(video)))

    def test_convert_to_mp4(self, mocker, tmp_path):
        mocker.patch("subprocess.run")
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        result = fetcher._convert_to_mp4("/tmp/input.mov")
        assert result.endswith(".mp4")

    def test_extract_audio(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        result = fetcher._extract_audio("/tmp/video.mp4")

        assert result.endswith(".wav")
        mock_run.assert_called_once()

    def test_sample_keyframes(self, mocker, tmp_path):
        mock_cap = mocker.MagicMock()
        mock_cap.get.return_value = 10.0
        mock_cap.read.side_effect = (
            [(True, np.zeros((100, 100, 3), dtype=np.uint8))] * 50 + [(False, None)]
        )
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        mocker.patch("cv2.imwrite")

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        result = fetcher._sample_keyframes("/tmp/video.mp4", interval=2.0)

        assert result is not None
        assert "frames" in result

    def test_load_audio(self, mocker, tmp_path):
        fake_audio = np.zeros(16000, dtype=np.float32)
        mocker.patch("librosa.load", return_value=(fake_audio, 16000))

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        y, sr = fetcher.load_audio("/tmp/audio.wav")

        assert sr == 16000
        assert len(y) == 16000

    def test_load_frames(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_000000.jpg").touch()
        (frames_dir / "frame_000001.jpg").touch()
        (frames_dir / "frame_000002.jpg").touch()

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        frames = fetcher.load_frames(str(frames_dir))

        assert len(frames) == 3
        assert all(f.endswith(".jpg") for f in frames)
