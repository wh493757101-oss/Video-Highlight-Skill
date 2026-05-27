import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .las_client import LasClient, LasConfig
from .rule_engine import HighlightSegment
from .video_fetcher import _get_ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class EditorConfig:
    output_dir: str = ""
    las_operator_id: str = field(default_factory=lambda: os.environ.get("LAS_OPERATOR_ID", "las_video_edit"))
    las_operator_version: str = "v1"
    output_tos_path: str = ""
    transition_duration: float = 0.5
    fallback_enabled: bool = True


@dataclass
class EditResult:
    output_path: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    source: str = "ffmpeg"
    degraded: bool = False
    degradation_reason: str = ""


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
            try:
                result = self._edit_with_ffmpeg(video_path, segments)
                result.degraded = True
                result.degradation_reason = f"LAS las_video_edit 云端算子不可用: {e}"
                return result
            except Exception as e2:
                logger.error("FFmpeg 降级也失败: %s", e2)
                raise RuntimeError(
                    "视频剪辑失败（LAS 和 FFmpeg 均不可用），请稍后重试"
                ) from e2

    def _edit_with_las(
        self,
        video_path: str,
        segments: list[HighlightSegment],
        description: str,
    ) -> EditResult:
        task_description = self._build_las_description(segments, description)

        video_url = self._resolve_video_url(video_path)

        output_tos_path = self.config.output_tos_path or os.environ.get("TOS_OUTPUT_PATH", "")
        task_input: dict[str, Any] = {
            "video_url": video_url,
            "task_description": task_description,
            "output_tos_path": output_tos_path,
            "mode": "detail",
        }

        logger.info("LAS submit: operator=%s, input keys=%s", self.config.las_operator_id, list(task_input.keys()))

        result = self.las_client.submit(
            self.config.las_operator_id,
            task_input,
            operator_version=self.config.las_operator_version,
        )
        task_id = result.get("metadata", {}).get("task_id", result.get("task_id", ""))
        if not task_id:
            raise RuntimeError("LAS 未返回 task_id")

        final = self.las_client.wait_for_completion(task_id)
        data = final.get("data", {})
        output_url = final.get("output", {}).get("url", "")

        if not output_url:
            clips = data.get("clips", [])
            valid_clips = [
                c for c in clips
                if c.get("clip_url") and c.get("file_size", 0) > 1024
            ]
            if valid_clips:
                output_url = valid_clips[0]["clip_url"]

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

    def _resolve_video_url(self, video_path: str) -> str:
        """将本地视频路径转换为可公网访问的 URL。

        如果已是 http/https/tos URL 则直接返回；否则上传到 TOS 并返回 tos:// URL。
        每个视频存放在独立的文件夹中: input/{filename}/video.mp4
        """
        if video_path.startswith(("http://", "https://", "tos://")):
            return video_path

        import os as _os
        import tos as _tos
        from pathlib import Path as _Path

        _ak = _os.environ.get("TOS_ACCESS_KEY", "")
        _sk = _os.environ.get("TOS_SECRET_KEY", "")
        _bucket = "arkclaw-tos-2124145136-cn-guangzhou"
        _base_prefix = "arkclaw-tos-ci-yemqjzxa0w9t6r1y3a0v-lk0rj/video-highlight-bucket"

        _client = _tos.TosClientV2(_ak, _sk, "tos-cn-guangzhou.volces.com", "cn-guangzhou")
        _filename = _Path(video_path).name
        _folder = _Path(video_path).stem
        _tos_key = f"{_base_prefix}/input/{_folder}/{_filename}"
        _client.put_object_from_file(_bucket, _tos_key, video_path)
        logger.info("视频已上传到 TOS: tos://%s/%s", _bucket, _tos_key)
        return f"tos://{_bucket}/{_tos_key}"

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
                _get_ffmpeg(), "-y",
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
            _get_ffmpeg(), "-y",
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
