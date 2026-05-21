import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SegmentMatch:
    predicted_start: float
    predicted_end: float
    gt_start: float
    gt_end: float
    iou: float
    hit: bool


@dataclass
class CaseScore:
    case_id: str
    category: str
    difficulty: str
    source_type: str
    iou_scores: list[float] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    matched_pairs: list[SegmentMatch] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalReport:
    scores: list[CaseScore] = field(default_factory=list)
    overall_iou: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)


class HighlightEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def compute_iou(
        self, pred_start: float, pred_end: float, gt_start: float, gt_end: float
    ) -> float:
        intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def match_segments(
        self,
        predicted: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
    ) -> list[SegmentMatch]:
        matches: list[SegmentMatch] = []
        gt_used: set[int] = set()

        for pred in predicted:
            best_iou = 0.0
            best_gt_idx = -1
            best_gt: dict[str, Any] | None = None

            for j, gt in enumerate(ground_truth):
                if j in gt_used:
                    continue
                iou = self.compute_iou(
                    pred["start_time"], pred["end_time"],
                    gt["start_time"], gt["end_time"],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    best_gt = gt

            hit = best_iou >= self.iou_threshold
            if hit and best_gt is not None:
                gt_used.add(best_gt_idx)

            matches.append(SegmentMatch(
                predicted_start=pred["start_time"],
                predicted_end=pred["end_time"],
                gt_start=best_gt["start_time"] if best_gt else 0.0,
                gt_end=best_gt["end_time"] if best_gt else 0.0,
                iou=best_iou,
                hit=hit,
            ))

        return matches

    def score_case(
        self,
        case_id: str,
        predicted: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        category: str = "",
        difficulty: str = "",
        source_type: str = "local",
    ) -> CaseScore:
        if not ground_truth:
            return CaseScore(
                case_id=case_id,
                category=category,
                difficulty=difficulty,
                source_type=source_type,
                error="ground_truth 为空",
            )

        if not predicted:
            return CaseScore(
                case_id=case_id,
                category=category,
                difficulty=difficulty,
                source_type=source_type,
                precision=0.0,
                recall=0.0,
                f1=0.0,
            )

        matches = self.match_segments(predicted, ground_truth)
        hit_count = sum(1 for m in matches if m.hit)
        iou_scores = [m.iou for m in matches]

        precision = hit_count / len(predicted) if predicted else 0.0
        recall = hit_count / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return CaseScore(
            case_id=case_id,
            category=category,
            difficulty=difficulty,
            source_type=source_type,
            iou_scores=iou_scores,
            precision=precision,
            recall=recall,
            f1=f1,
            matched_pairs=matches,
        )

    def evaluate_all(self, results: list[dict[str, Any]]) -> EvalReport:
        report = EvalReport()
        if not results:
            return report

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_iou = 0.0
        iou_count = 0

        cat_scores: dict[str, list[float]] = {}
        dif_scores: dict[str, list[float]] = {}
        src_scores: dict[str, list[float]] = {}

        for r in results:
            score = self.score_case(
                case_id=r.get("case_id", ""),
                predicted=r.get("predicted", []),
                ground_truth=r.get("ground_truth", []),
                category=r.get("category", ""),
                difficulty=r.get("difficulty", ""),
                source_type=r.get("source_type", "local"),
            )
            report.scores.append(score)

            if score.error:
                continue

            total_precision += score.precision
            total_recall += score.recall
            total_f1 += score.f1
            for iou in score.iou_scores:
                total_iou += iou
                iou_count += 1

            cat_scores.setdefault(score.category, []).append(score.f1)
            dif_scores.setdefault(score.difficulty, []).append(score.f1)
            src_scores.setdefault(score.source_type, []).append(score.f1)

        n = len([s for s in report.scores if not s.error]) or 1
        report.overall_precision = total_precision / n
        report.overall_recall = total_recall / n
        report.overall_f1 = total_f1 / n
        report.overall_iou = total_iou / iou_count if iou_count > 0 else 0.0

        report.by_category = {
            k: {"f1": sum(v) / len(v), "count": len(v)}
            for k, v in cat_scores.items()
        }
        report.by_difficulty = {
            k: {"f1": sum(v) / len(v), "count": len(v)}
            for k, v in dif_scores.items()
        }
        report.by_source = {
            k: {"f1": sum(v) / len(v), "count": len(v)}
            for k, v in src_scores.items()
        }

        return report


class TestCaseLoader:
    def __init__(self, test_cases_root: str):
        self.root = Path(test_cases_root)

    def load_local_cases(self) -> list[dict[str, Any]]:
        cases_yaml = self.root / "local" / "cases.yaml"
        if not cases_yaml.exists():
            return []

        with open(cases_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cases: list[dict[str, Any]] = []
        for c in data.get("cases", []):
            case_dir = self.root / "local" / c["id"]
            instruction_path = case_dir / "instruction.json"
            gt_path = case_dir / "ground_truth.json"

            instruction = {}
            if instruction_path.exists():
                instruction = json.loads(instruction_path.read_text(encoding="utf-8"))

            ground_truth = []
            if gt_path.exists():
                gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
                ground_truth = gt_data.get("highlights", [])

            cases.append({
                "case_id": c["id"],
                "category": c["category"],
                "difficulty": c["difficulty"],
                "source_type": "local",
                "video_path": str(case_dir / c.get("video_file", "video.mp4")),
                "instruction": instruction,
                "ground_truth": ground_truth,
            })

        return cases

    def load_remote_cases(self) -> list[dict[str, Any]]:
        cases_yaml = self.root / "remote" / "cases.yaml"
        if not cases_yaml.exists():
            return []

        with open(cases_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cases: list[dict[str, Any]] = []
        for c in data.get("cases", []):
            case_dir = self.root / "remote" / c["id"]
            instruction_path = case_dir / "instruction.json"
            gt_path = case_dir / "ground_truth.json"

            instruction = {}
            if instruction_path.exists():
                instruction = json.loads(instruction_path.read_text(encoding="utf-8"))

            ground_truth = []
            if gt_path.exists():
                gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
                ground_truth = gt_data.get("highlights", [])

            cases.append({
                "case_id": c["id"],
                "category": c["category"],
                "difficulty": c["difficulty"],
                "source_type": "remote",
                "source_url": c.get("source_url", ""),
                "instruction": instruction,
                "ground_truth": ground_truth,
            })

        return cases

    def load_all(self) -> list[dict[str, Any]]:
        return self.load_local_cases() + self.load_remote_cases()
