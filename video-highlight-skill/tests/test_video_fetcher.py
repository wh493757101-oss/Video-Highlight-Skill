import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import subprocess

from src.video_fetcher import (
    ArkFileSource,
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
        mocker.patch.object(
            Path, "glob", return_value=[Path("/tmp/video_dl_abc/title.mp4")]
        )
        # mock _convert_to_mp4 to avoid ffmpeg call
        mocker.patch(
            "src.video_fetcher._convert_to_mp4",
            return_value="/tmp/video_dl_abc/title.mp4",
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
    def test_resolve_missing_config_raises(self):
        source = TosSource("tos://bucket/video.mp4")
        with pytest.raises(RuntimeError, match="TOS 配置不完整"):
            source.resolve()

    def test_parse_tos_path(self):
        source = TosSource("tos://mybucket/path/to/video.mp4")
        bucket, key = source._parse_tos_path()
        assert bucket == "mybucket"
        assert key == "path/to/video.mp4"

    def test_parse_tos_path_invalid(self):
        source = TosSource("tos://nokey")
        with pytest.raises(ValueError, match="无效的 TOS 路径格式"):
            source._parse_tos_path()

    def test_resolve_success(self, mocker, tmp_path):
        mock_boto = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"boto3": mocker.MagicMock(), "botocore.client": mocker.MagicMock()})
        mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path))
        mocker.patch.object(TosSource, "_parse_tos_path", return_value=("bucket", "key.mp4"))

        source = TosSource("tos://bucket/key.mp4", endpoint="http://tos.example.com", access_key="ak", secret_key="sk")
        result = source.resolve()
        assert result.endswith("key.mp4")


class TestVideoFetcher:
    def test_fetch_local_file(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake mp4 content")

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
        mocker.patch("src.video_fetcher._convert_to_mp4", return_value=str(video))
        mock_extract = mocker.patch.object(
            VideoFetcher, "_extract_audio", return_value="/tmp/out/audio/test.wav"
        )

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        meta = fetcher.fetch(LocalFileSource(str(video)))

        assert meta.duration == 10.0
        assert meta.fps == 30.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.audio_path == "/tmp/out/audio/test.wav"
        assert meta.frames_dir is not None

    def test_fetch_cannot_open_video(self, mocker, tmp_path):
        video = tmp_path / "bad.mp4"
        video.write_bytes(b"fake mp4 content")

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
        # probe returns "Audio:" in stderr to skip the probe guard
        mock_probe = mocker.MagicMock()
        mock_probe.stderr = b"Stream #0:1: Audio: aac"
        mock_run = mocker.patch("subprocess.run", return_value=mock_probe)
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        result = fetcher._extract_audio("/tmp/video.mp4")

        assert result is not None
        assert result.endswith(".wav")

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

    def test_fetch_empty_file_raises(self, tmp_path):
        video = tmp_path / "empty.mp4"
        video.write_text("")

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="视频文件为空"):
            fetcher.fetch(LocalFileSource(str(video)))

    def test_fetch_non_video_file_raises(self, mocker, tmp_path):
        video = tmp_path / "fake.mp4"
        video.write_text("this is not a video file")

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 0.0,
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_FRAME_WIDTH: 0,
            cv2.CAP_PROP_FRAME_HEIGHT: 0,
        }[prop]
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="无法解析视频参数|视频文件中无视频帧"):
            fetcher.fetch(LocalFileSource(str(video)))

    def test_extract_audio_failure_returns_none(self, mocker, tmp_path):
        mocker.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr="No audio stream"))
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        result = fetcher._extract_audio("/tmp/video.mp4")
        assert result is None

    def test_extract_audio_timeout_returns_none(self, mocker, tmp_path):
        mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 120))
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        result = fetcher._extract_audio("/tmp/video.mp4")
        assert result is None

    def test_convert_to_mp4_failure_raises(self, mocker, tmp_path):
        mocker.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr="conversion error"))
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="视频转码失败"):
            fetcher._convert_to_mp4("/tmp/input.mov")

    def test_load_audio_failure_returns_empty(self, mocker, tmp_path):
        mocker.patch("librosa.load", side_effect=RuntimeError("corrupt audio"))
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        y, sr = fetcher.load_audio("/tmp/bad.wav")
        assert len(y) == 0
        assert sr == 22050


class TestArkFileSource:
    def test_resolve_success(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake mp4 content")

        mock_client = mocker.MagicMock()
        mock_client.upload_file.return_value = {"download_url": "https://ark-cn-beijing.volces.com/dl/test_video"}
        mocker.patch("src.ark_client.ArkClient", return_value=mock_client)

        source = ArkFileSource(str(video))
        result = source.resolve()

        assert result == "https://ark-cn-beijing.volces.com/dl/test_video"
        mock_client.upload_file.assert_called_once()

    def test_resolve_file_not_found(self):
        source = ArkFileSource("/nonexistent/video.mp4")
        with pytest.raises(FileNotFoundError, match="视频文件不存在"):
            source.resolve()

    def test_resolve_no_download_url(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake mp4 content")

        mock_client = mocker.MagicMock()
        mock_client.upload_file.return_value = {"id": "file-123"}
        mocker.patch("src.ark_client.ArkClient", return_value=mock_client)

        source = ArkFileSource(str(video))
        with pytest.raises(RuntimeError, match="未返回 download_url"):
            source.resolve()
