import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import cv2
import librosa
import numpy as np
import soundfile as sf


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
            raise FileNotFoundError(f"视频文件不存在: {self.path}")
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
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"视频下载失败: {e.stderr}") from e

        files = list(output_dir.glob("*"))
        if not files:
            raise RuntimeError("下载完成但未找到输出文件")
        return str(files[0])


class TosSource:
    def __init__(self, tos_path: str, endpoint: str | None = None):
        self.tos_path = tos_path

    def resolve(self) -> str:
        raise NotImplementedError("TOS 下载需要 tos-sdk，暂未实现")


class VideoFetcher:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="vf_"))

    def fetch(self, source: VideoSource) -> VideoMetadata:
        video_path = source.resolve()
        return self._preprocess(video_path)

    def _preprocess(self, video_path: str) -> VideoMetadata:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
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
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        return dst

    def _extract_audio(self, video_path: str) -> str:
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
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return audio_path

    def _sample_keyframes(self, video_path: str, interval: float = 2.0) -> str:
        frames_dir = self.output_dir / "frames" / Path(video_path).stem
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)

        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                cv2.imwrite(str(frames_dir / f"frame_{saved:06d}.jpg"), frame)
                saved += 1
            idx += 1

        cap.release()
        return str(frames_dir)

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        y, sr = librosa.load(audio_path, sr=None)
        return y, sr

    def load_frames(self, frames_dir: str) -> list[str]:
        frame_files = sorted(Path(frames_dir).glob("*.jpg"))
        return [str(f) for f in frame_files]
