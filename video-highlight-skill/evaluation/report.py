import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    output_dir: str = ""
    save_charts: bool = True
    save_json: bool = True


class ReportGenerator:
    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()

    def generate(
        self,
        eval_report: Any,
        judge_report: Any,
    ) -> str:
        lines: list[str] = []

        lines.append("=" * 70)
        lines.append("视频高光剪辑 — 评测报告")
        lines.append("=" * 70)

        lines.append("")
        lines.append("## 一、时间戳 IoU 评测")
        lines.append("")
        lines.append(f"  整体 IoU:    {eval_report.overall_iou:.3f}")
        lines.append(f"  整体 Precision: {eval_report.overall_precision:.3f}")
        lines.append(f"  整体 Recall:    {eval_report.overall_recall:.3f}")
        lines.append(f"  整体 F1:        {eval_report.overall_f1:.3f}")

        if eval_report.by_category:
            lines.append("")
            lines.append("  按视频类型:")
            for cat, stats in eval_report.by_category.items():
                lines.append(f"    {cat}: F1={stats['f1']:.3f} (n={stats['count']})")

        if eval_report.by_difficulty:
            lines.append("")
            lines.append("  按难度:")
            for dif, stats in eval_report.by_difficulty.items():
                lines.append(f"    {dif}: F1={stats['f1']:.3f} (n={stats['count']})")

        if eval_report.by_source:
            lines.append("")
            lines.append("  按来源:")
            for src, stats in eval_report.by_source.items():
                lines.append(f"    {src}: F1={stats['f1']:.3f} (n={stats['count']})")

        lines.append("")
        lines.append("  各用例详情:")
        lines.append(f"  {'ID':<12} {'类型':<8} {'难度':<8} {'来源':<6} {'Prec':<8} {'Recall':<8} {'F1':<8}")
        lines.append("  " + "-" * 60)
        for score in eval_report.scores:
            if score.error:
                lines.append(f"  {score.case_id:<12} {'-':<8} {'-':<8} {'-':<6} [SKIP] {score.error}")
            else:
                lines.append(
                    f"  {score.case_id:<12} {score.category:<8} {score.difficulty:<8} "
                    f"{score.source_type:<6} {score.precision:<8.3f} {score.recall:<8.3f} {score.f1:<8.3f}"
                )

        lines.append("")
        lines.append("## 二、LLM Judge 主观评测")
        lines.append("")

        if hasattr(judge_report, 'overall_average'):
            lines.append(f"  节奏感:       {judge_report.overall_rhythm:.2f} / 5.0")
            lines.append(f"  内容完整性:   {judge_report.overall_completeness:.2f} / 5.0")
            lines.append(f"  精彩程度:     {judge_report.overall_excitement:.2f} / 5.0")
            lines.append(f"  指令契合度:   {judge_report.overall_instruction_fit:.2f} / 5.0")
            lines.append(f"  综合均分:     {judge_report.overall_average:.2f} / 5.0")

            lines.append("")
            lines.append("  各用例 LLM 评价:")
            for i, score in enumerate(judge_report.scores):
                if score.error:
                    lines.append(f"    #{i + 1}: [ERROR] {score.error}")
                else:
                    lines.append(
                        f"    #{i + 1}: {score.average:.1f}/5.0 — {score.overall_comment}"
                    )

        lines.append("")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        if self.config.output_dir:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if self.config.save_json:
                json_path = out_dir / "report.json"
                json_path.write_text(
                    self._build_json(eval_report, judge_report),
                    encoding="utf-8",
                )
                logger.info("JSON 报告已保存: %s", json_path)

            text_path = out_dir / "report.txt"
            text_path.write_text(report_text, encoding="utf-8")
            logger.info("文本报告已保存: %s", text_path)

            if self.config.save_charts:
                self._save_charts(eval_report, judge_report, out_dir)

        return report_text

    def _build_json(self, eval_report: Any, judge_report: Any) -> str:
        data: dict[str, Any] = {
            "iou_eval": {
                "overall_iou": eval_report.overall_iou,
                "overall_precision": eval_report.overall_precision,
                "overall_recall": eval_report.overall_recall,
                "overall_f1": eval_report.overall_f1,
                "by_category": {
                    k: {"f1": round(v["f1"], 3), "count": v["count"]}
                    for k, v in eval_report.by_category.items()
                },
                "by_difficulty": {
                    k: {"f1": round(v["f1"], 3), "count": v["count"]}
                    for k, v in eval_report.by_difficulty.items()
                },
                "by_source": {
                    k: {"f1": round(v["f1"], 3), "count": v["count"]}
                    for k, v in eval_report.by_source.items()
                },
                "cases": [
                    {
                        "case_id": s.case_id,
                        "category": s.category,
                        "difficulty": s.difficulty,
                        "source_type": s.source_type,
                        "precision": round(s.precision, 3),
                        "recall": round(s.recall, 3),
                        "f1": round(s.f1, 3),
                        "error": s.error,
                    }
                    for s in eval_report.scores
                ],
            },
        }

        if hasattr(judge_report, 'overall_average'):
            data["llm_judge"] = {
                "overall_rhythm": round(judge_report.overall_rhythm, 2),
                "overall_completeness": round(judge_report.overall_completeness, 2),
                "overall_excitement": round(judge_report.overall_excitement, 2),
                "overall_instruction_fit": round(judge_report.overall_instruction_fit, 2),
                "overall_average": round(judge_report.overall_average, 2),
                "cases": [
                    {
                        "rhythm": s.rhythm,
                        "completeness": s.completeness,
                        "excitement": s.excitement,
                        "instruction_fit": s.instruction_fit,
                        "average": round(s.average, 1),
                        "comment": s.overall_comment,
                        "error": s.error,
                    }
                    for s in judge_report.scores
                ],
            }

        return json.dumps(data, ensure_ascii=False, indent=2)

    def _save_charts(
        self,
        eval_report: Any,
        judge_report: Any,
        out_dir: Path,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib 未安装，跳过图表生成")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. F1 by category
        if eval_report.by_category:
            ax = axes[0][0]
            cats = list(eval_report.by_category.keys())
            f1s = [eval_report.by_category[c]["f1"] for c in cats]
            colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))
            ax.bar(cats, f1s, color=colors)
            ax.set_title("F1 by Category")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            for i, v in enumerate(f1s):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        # 2. F1 by difficulty
        if eval_report.by_difficulty:
            ax = axes[0][1]
            difs = list(eval_report.by_difficulty.keys())
            f1s = [eval_report.by_difficulty[d]["f1"] for d in difs]
            colors = ["#4CAF50" if d == "easy" else "#FF9800" if d == "medium" else "#F44336" for d in difs]
            ax.bar(difs, f1s, color=colors)
            ax.set_title("F1 by Difficulty")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            for i, v in enumerate(f1s):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        # 3. F1 by source
        if eval_report.by_source:
            ax = axes[1][0]
            srcs = list(eval_report.by_source.keys())
            f1s = [eval_report.by_source[s]["f1"] for s in srcs]
            ax.bar(srcs, f1s, color=["#2196F3", "#FF5722"])
            ax.set_title("F1 by Source")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            for i, v in enumerate(f1s):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        # 4. LLM Judge radar
        if hasattr(judge_report, 'overall_average') and judge_report.overall_average > 0:
            ax = axes[1][1]
            ax.set_title("LLM Judge")
            dims = ["Rhythm", "Completeness", "Excitement", "Fit"]
            values = [
                judge_report.overall_rhythm,
                judge_report.overall_completeness,
                judge_report.overall_excitement,
                judge_report.overall_instruction_fit,
            ]
            x = np.arange(len(dims))
            ax.bar(x, values, color=plt.cm.Set3(np.linspace(0, 1, 4)))
            ax.set_xticks(x)
            ax.set_xticklabels(dims)
            ax.set_ylabel("Score")
            ax.set_ylim(0, 5)
            for i, v in enumerate(values):
                ax.text(i, v + 0.1, f"{v:.1f}", ha="center")

        plt.tight_layout()
        chart_path = out_dir / "charts.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("图表已保存: %s", chart_path)
