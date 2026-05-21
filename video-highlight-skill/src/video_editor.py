import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .las_client import LasClient, LasConfig
from .rule_engine import HighlightSegment

logger = logging.getLogger(__name__)


@dataclass
class EditorConfig:
    output_dir: str = ""
    las_operator_id: str = "las_video_edit"
    transition_duration: float = 0.5
    fallback_enabled: bool = True


@dataclass
class EditResult:
    output_path: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    source: str = "ffmpeg"


class VideoEditor:
    def __init__(
        self,
        config: EditorConfig | None = None,
        las_client: LasClient | None = None,
    ):
        self.config = config or EditorConfig()
        self._las_client = las_client

    @property
    def las_client(self) -> LasClient:
        if self._las_client is None:
            self._las_client = LasClient()
        return self._las_client

    def edit(
        self,
        video_path: str,
        segments: list[HighlightSegment],
        description: str = "",
    ) -> EditResult:
        try:
            return self._edit_with_las(video_path, segments, description)
        except Exception as e:
            logger.warning("LAS 剪辑失败，降级到 FFmpeg: %s", e)
            if not self.config.fallback_enabled:
                raise
            return self._edit_with_ffmpeg(video_path, segments)

    def _edit_with_las(
        self,
        video_path: str,
        segments: list[HighlightSegment],
        description: str,
    ) -> EditResult:
        task_description = self._build_las_description(segments, description)

        task_input: dict[str, Any] = {
            "video_url": video_path,
            "task_description": task_description,
            "output_format": "mp4",
        }

        result = self.las_client.submit(self.config.las_operator_id, task_input)
        task_id = result.get("task_id", "")
        if not task_id:
            raise RuntimeError("LAS 未返回 task_id")

        final = self.las_client.wait_for_completion(task_id)
        output = final.get("output", {})
        output_url = output.get("url", output.get("video_url", ""))

        seg_info = [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "score": s.combined_score,
            }
            for s in segments
        ]

        return EditResult(
            output_path=output_url,
            segments=seg_info,
            source="las",
        )

    def _edit_with_ffmpeg(
        self,
        video_path: str,
        segments: list[HighlightSegment],
    ) -> EditResult:
        output_dir = Path(self.config.output_dir) if self.config.output_dir else Path(tempfile.mkdtemp(prefix="ve_"))
        output_dir.mkdir(parents=True, exist_ok=True)

        clip_paths: list[str] = []
        for i, seg in enumerate(segments):
            clip_path = str(output_dir / f"clip_{i:03d}.mp4")
            duration = seg.end_time - seg.start_time
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg.start_time),
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                clip_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            clip_paths.append(clip_path)

        concat_file = output_dir / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{p}'" for p in clip_paths)
        )

        output_path = str(output_dir / "highlight_reel.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)

        seg_info = [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "score": s.combined_score,
            }
            for s in segments
        ]

        return EditResult(
            output_path=output_path,
            segments=seg_info,
            source="ffmpeg",
        )

    def _build_las_description(
        self,
        segments: list[HighlightSegment],
        user_description: str,
    ) -> str:
        parts: list[str] = []
        if user_description:
            parts.append(f"用户需求: {user_description}")

        parts.append("请从以下时间段剪辑高光集锦:")
        for i, seg in enumerate(segments):
            parts.append(
                f"片段{i + 1}: {seg.start_time:.1f}s - {seg.end_time:.1f}s"
                f"（精彩度: {seg.combined_score:.2f}）"
            )

        parts.append(
            f"要求: 片段之间添加 {self.config.transition_duration}s 转场，"
            "输出为 mp4 格式。"
        )

        return "\n".join(parts)
