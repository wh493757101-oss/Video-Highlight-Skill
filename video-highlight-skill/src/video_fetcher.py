import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    audio_path: str | None = None
    frames_dir: str | None = None


class VideoSource(Protocol):
    def resolve(self) -> str: ...


class LocalFileSource:
    def __init__(self, path: str):
        self.path = Path(path)

    def resolve(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.path}，请检查文件路径后重试")
        return str(self.path.absolute())


class UrlSource:
    def __init__(self, url: str):
        self.url = url

    def resolve(self) -> str:
        output_dir = Path(tempfile.mkdtemp(prefix="video_dl_"))
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", str(output_dir / "%(title)s.%(ext)s"),
            self.url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"视频下载超时，请检查网络后重试: {self.url}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"视频下载失败，请检查链接是否有效后重试: {e.stderr}") from e

        files = list(output_dir.glob("*"))
        if not files:
            raise RuntimeError("下载完成但未找到输出文件，请稍后重试")
        return str(files[0])


class TosSource:
    def __init__(self, tos_path: str, endpoint: str | None = None, access_key: str | None = None, secret_key: str | None = None):
        self.tos_path = tos_path
        self.endpoint = endpoint or os.getenv("TOS_ENDPOINT", "")
        self.access_key = access_key or os.getenv("TOS_ACCESS_KEY", "")
        self.secret_key = secret_key or os.getenv("TOS_SECRET_KEY", "")

    def resolve(self) -> str:
        if not self.endpoint or not self.access_key or not self.secret_key:
            raise RuntimeError(
                "TOS 配置不完整，请设置环境变量 TOS_ENDPOINT / TOS_ACCESS_KEY / TOS_SECRET_KEY "
                "或通过 TosSource 构造参数传入"
            )
        output_dir = Path(tempfile.mkdtemp(prefix="tos_dl_"))
        bucket, key = self._parse_tos_path()
        output_path = output_dir / Path(key).name
        try:
            import boto3
            from botocore.client import Config
            s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(signature_version="s3v4"),
            )
            s3.download_file(bucket, key, str(output_path))
        except ImportError:
            raise RuntimeError("TOS 下载需要 boto3，请执行: pip install boto3")
        except Exception as e:
            raise RuntimeError(f"TOS 下载失败，请检查凭证和路径后重试: {e}") from e
        return str(output_path)

    def _parse_tos_path(self) -> tuple[str, str]:
        path = self.tos_path
        if path.startswith("tos://"):
            path = path[6:]
        elif path.startswith("s3://"):
            path = path[5:]
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"无效的 TOS 路径格式: {self.tos_path}，期望格式: tos://bucket/key")
        return parts[0], parts[1]


class ArkFileSource:
    """通过 Ark Files API 上传本地文件并获取 HTTPS 预签名下载链接。

    只需 ARK_API_KEY，无需 TOS 凭证。返回的 download_url 24 小时有效，
    可直接作为 LAS las_video_edit 算子的输入 URL。
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def resolve(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.path}，请检查文件路径后重试")

        from .ark_client import ArkClient

        client = ArkClient()
        result = client.upload_file(str(self.path.absolute()))
        download_url = result.get("download_url", "")
        if not download_url:
            raise RuntimeError("Files API 未返回 download_url，请稍后重试")
        logger.info("文件已上传到 Ark Files API: %s", download_url[:80])
        return download_url


class VideoFetcher:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="vf_"))

    def fetch(self, source: VideoSource) -> VideoMetadata:
        video_path = source.resolve()
        return self._preprocess(video_path)

    def _preprocess(self, video_path: str) -> VideoMetadata:
        file_path = Path(video_path)
        if not file_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}，请检查文件路径后重试")
        if file_path.stat().st_size == 0:
            raise ValueError(f"视频文件为空（0 字节）: {video_path}，请检查视频源后重试")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}，请检查文件格式后重试")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            if fps <= 0 or duration <= 0:
                raise ValueError(f"无法解析视频参数: fps={fps}, duration={duration}, frames={frame_count}，请检查视频文件后重试")
            if frame_count == 0:
                raise ValueError(f"视频文件中无视频帧: {video_path}，请检查视频源后重试")
        finally:
            cap.release()

        # 统一为 mp4（如果不是则转码）
        src_path = Path(video_path)
        if src_path.suffix.lower() != ".mp4":
            video_path = self._convert_to_mp4(video_path)

        # 提取音频
        audio_path = self._extract_audio(video_path)

        # 关键帧采样
        frames_dir = self._sample_keyframes(video_path)

        return VideoMetadata(
            path=video_path,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            audio_path=audio_path,
            frames_dir=frames_dir,
        )

    def _convert_to_mp4(self, src: str) -> str:
        dst = str(self.output_dir / f"{Path(src).stem}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", src,
            "-c:v", "libx264",
            "-c:a", "aac",
            dst,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"视频转码超时（300s）: {src}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"视频转码失败: {e.stderr}") from e
        return dst

    def _extract_audio(self, video_path: str) -> str | None:
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = str(audio_dir / f"{Path(video_path).stem}.wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            return audio_path
        except subprocess.TimeoutExpired:
            logger.warning("音频提取超时，继续无音频处理")
            return None
        except subprocess.CalledProcessError as e:
            logger.warning("音频提取失败（可能无音频流），继续无音频处理: %s", e.stderr.strip())
            return None

    def _sample_keyframes(self, video_path: str, interval: float = 2.0) -> str:
        frames_dir = self.output_dir / "frames" / Path(video_path).stem
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning("无法获取视频帧率，使用默认 30fps")
            fps = 30.0
        frame_interval = max(1, int(fps * interval))

        idx = 0
        saved = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_interval == 0:
                    cv2.imwrite(str(frames_dir / f"frame_{saved:06d}.jpg"), frame)
                    saved += 1
                idx += 1
        finally:
            cap.release()

        if saved == 0:
            logger.warning("关键帧采样结果为空: %s", video_path)
        return str(frames_dir)

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        try:
            y, sr = librosa.load(audio_path, sr=None)
            return y, sr
        except Exception as e:
            logger.warning("音频加载失败: %s", e)
            return np.array([]), 22050

    def load_frames(self, frames_dir: str) -> list[str]:
        frame_files = sorted(Path(frames_dir).glob("*.jpg"))
        return [str(f) for f in frame_files]
