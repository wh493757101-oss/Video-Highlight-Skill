import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """你是一个专业的视频剪辑质量评审员。请根据以下信息对高光集锦视频的剪辑质量进行多维度评分。

## 原始视频信息
- 视频类型: {category}
- 剪辑目标: {target}
- 风格要求: {style}

## 高光片段列表
{segments}

## 评分维度（每项 1-5 分）

1. **节奏感** — 片段衔接是否流畅，节奏是否符合风格要求
2. **内容完整性** — 每个高光片段是否完整表达了关键内容，是否有截断感
3. **精彩程度** — 选中的片段是否是真正的高光时刻
4. **指令契合度** — 剪辑结果是否符合用户的剪辑目标和要求

请以 JSON 格式返回评分结果：
{{
  "节奏感": <1-5>,
  "内容完整性": <1-5>,
  "精彩程度": <1-5>,
  "指令契合度": <1-5>,
  "总体评价": "<一句话评价>"
}}

只返回 JSON，不要包含其他文字。"""


@dataclass
class JudgeScore:
    rhythm: float = 0.0
    completeness: float = 0.0
    excitement: float = 0.0
    instruction_fit: float = 0.0
    overall_comment: str = ""
    error: str | None = None

    @property
    def average(self) -> float:
        scores = [self.rhythm, self.completeness, self.excitement, self.instruction_fit]
        return sum(scores) / len(scores)


@dataclass
class JudgeReport:
    scores: list[JudgeScore] = field(default_factory=list)
    overall_rhythm: float = 0.0
    overall_completeness: float = 0.0
    overall_excitement: float = 0.0
    overall_instruction_fit: float = 0.0
    overall_average: float = 0.0
    degraded: bool = False


class LLMJudge:
    def __init__(self, ark_client=None):
        self._ark_client = ark_client

    @property
    def ark_client(self):
        if self._ark_client is None:
            from src.ark_client import ArkClient
            self._ark_client = ArkClient()
        return self._ark_client

    def build_prompt(
        self,
        category: str,
        target: str,
        style: str,
        segments: list[dict[str, Any]],
    ) -> str:
        segment_lines: list[str] = []
        for i, seg in enumerate(segments):
            label = seg.get("label", "")
            label_str = f" ({label})" if label else ""
            segment_lines.append(
                f"  #{i + 1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s"
                f" (精彩度: {seg.get('score', 0):.2f}){label_str}"
            )

        return JUDGE_PROMPT.format(
            category=category,
            target=target or "精彩集锦",
            style=style or "无特定要求",
            segments="\n".join(segment_lines) if segment_lines else "无",
        )

    def judge(
        self,
        category: str,
        target: str,
        style: str,
        segments: list[dict[str, Any]],
        max_retries: int = 3,
    ) -> JudgeScore:
        prompt = self.build_prompt(category, target, style, segments)

        last_error: str = ""
        for attempt in range(max_retries):
            try:
                response = self.ark_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                )
                parsed = self.ark_client.extract_json(response)

                return JudgeScore(
                    rhythm=float(parsed.get("节奏感", 0)),
                    completeness=float(parsed.get("内容完整性", 0)),
                    excitement=float(parsed.get("精彩程度", 0)),
                    instruction_fit=float(parsed.get("指令契合度", 0)),
                    overall_comment=str(parsed.get("总体评价", "")),
                )
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    logger.warning("LLM Judge 评分失败（第 %d/%d 次）: %s，重试中...", attempt + 1, max_retries, e)
                    time.sleep(1)
                    continue
                logger.warning("LLM Judge 评分失败（已重试 %d 次）: %s", max_retries, last_error)
                return JudgeScore(error=last_error)

    def judge_all(self, cases: list[dict[str, Any]], max_retries: int = 3) -> JudgeReport:
        report = JudgeReport()
        if not cases:
            return report

        total_rhythm = 0.0
        total_completeness = 0.0
        total_excitement = 0.0
        total_fit = 0.0
        valid_count = 0

        for case in cases:
            score = self.judge(
                category=case.get("category", ""),
                target=case.get("target", ""),
                style=case.get("style", ""),
                segments=case.get("segments", []),
                max_retries=max_retries,
            )
            report.scores.append(score)

            if score.error:
                continue

            total_rhythm += score.rhythm
            total_completeness += score.completeness
            total_excitement += score.excitement
            total_fit += score.instruction_fit
            valid_count += 1

        if valid_count > 0:
            report.overall_rhythm = total_rhythm / valid_count
            report.overall_completeness = total_completeness / valid_count
            report.overall_excitement = total_excitement / valid_count
            report.overall_instruction_fit = total_fit / valid_count
            report.overall_average = (
                report.overall_rhythm
                + report.overall_completeness
                + report.overall_excitement
                + report.overall_instruction_fit
            ) / 4
        else:
            report.degraded = True

        return report


def format_judge_report(report: JudgeReport) -> str:
    lines = [
        "=" * 50,
        "LLM Judge 评分报告",
        "=" * 50,
        "",
        f"节奏感:       {report.overall_rhythm:.2f} / 5.0",
        f"内容完整性:   {report.overall_completeness:.2f} / 5.0",
        f"精彩程度:     {report.overall_excitement:.2f} / 5.0",
        f"指令契合度:   {report.overall_instruction_fit:.2f} / 5.0",
        f"综合均分:     {report.overall_average:.2f} / 5.0",
        "",
        "各用例评价:",
    ]

    for i, score in enumerate(report.scores):
        if score.error:
            lines.append(f"  #{i + 1}: [ERROR] {score.error}")
        else:
            lines.append(
                f"  #{i + 1}: 均分 {score.average:.1f} — {score.overall_comment}"
            )

    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)
