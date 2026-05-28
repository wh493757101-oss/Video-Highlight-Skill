import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EditorConfig:
    output_dir: str = ""
    concat_list_filename: str = "concat_list.txt"


@dataclass
class EditTiming:
    """剪辑各阶段耗时（秒）。"""
    detection: float = 0.0
    ffmpeg_concat: float = 0.0


@dataclass
class EditResult:
    output_path: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    source: str = "multimodal"
    timing: EditTiming = field(default_factory=EditTiming)


class VideoEditor:
    def __init__(self, config: EditorConfig | None = None):
        self.config = config or EditorConfig()

    def edit_with_ffmpeg(
        self,
        video_path: str,
        segments: list[dict[str, Any]],
    ) -> EditResult:
        """使用 FFmpeg stream-copy 模式拼接高光片段。

        两阶段：先逐片段裁剪，再 concat demuxer 拼接。使用 -c copy 避免重新编码。
        """
        if not segments:
            raise ValueError("segments 为空，无法进行剪辑")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"highlight_reel_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

        t_start = time.time()
        clip_paths: list[str] = []

        try:
            for i, seg in enumerate(segments):
                start = seg["start_time"]
                end = seg["end_time"]
                duration = end - start
                clip_path = str(output_dir / f"_clip_{i:03d}.mp4")
                cmd = [
                    _get_ffmpeg(), "-y",
                    "-ss", str(start),
                    "-i", video_path,
                    "-t", str(duration),
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    clip_path,
                ]
                subprocess.run(cmd, check=True, capture_output=True, timeout=120)
                clip_paths.append(clip_path)

            concat_list_path = output_dir / self.config.concat_list_filename
            with open(concat_list_path, "w") as f:
                for cp in clip_paths:
                    f.write(f"file '{cp}'\n")

            concat_cmd = [
                _get_ffmpeg(), "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-c", "copy",
                output_path,
            ]
            subprocess.run(concat_cmd, check=True, capture_output=True, timeout=120)
        finally:
            for cp in clip_paths:
                try:
                    Path(cp).unlink(missing_ok=True)
                except OSError:
                    pass

        t_concat = time.time() - t_start

        return EditResult(
            output_path=output_path,
            segments=segments,
            source="multimodal",
            timing=EditTiming(ffmpeg_concat=t_concat),
        )


def _get_ffmpeg() -> str:
    import shutil
    for candidate in [
        shutil.which("ffmpeg") or shutil.which("ffmpeg.exe") or "",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    try:
        import imageio_ffmpeg
        exe: str | None = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except ImportError:
        pass
    return "ffmpeg"
