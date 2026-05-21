import json
from pathlib import Path

import pytest

from evaluation.evaluator import (
    CaseScore,
    EvalReport,
    HighlightEvaluator,
    SegmentMatch,
    TestCaseLoader,
)
from evaluation.llm_judge import (
    JudgeReport,
    JudgeScore,
    LLMJudge,
    format_judge_report,
)
from evaluation.report import ReportConfig, ReportGenerator


class TestHighlightEvaluator:
    def test_compute_iou_perfect_match(self):
        evaluator = HighlightEvaluator()
        iou = evaluator.compute_iou(0.0, 5.0, 0.0, 5.0)
        assert iou == 1.0

    def test_compute_iou_no_overlap(self):
        evaluator = HighlightEvaluator()
        iou = evaluator.compute_iou(0.0, 2.0, 3.0, 5.0)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self):
        evaluator = HighlightEvaluator()
        iou = evaluator.compute_iou(0.0, 4.0, 2.0, 6.0)
        assert 0.0 < iou < 1.0

    def test_compute_iou_zero_union(self):
        evaluator = HighlightEvaluator()
        iou = evaluator.compute_iou(1.0, 1.0, 1.0, 1.0)
        assert iou == 0.0

    def test_match_segments_perfect(self):
        evaluator = HighlightEvaluator(iou_threshold=0.5)
        predicted = [{"start_time": 0.0, "end_time": 5.0}]
        ground_truth = [{"start_time": 0.0, "end_time": 5.0}]

        matches = evaluator.match_segments(predicted, ground_truth)
        assert len(matches) == 1
        assert matches[0].hit is True
        assert matches[0].iou == 1.0

    def test_match_segments_no_hit(self):
        evaluator = HighlightEvaluator(iou_threshold=0.5)
        predicted = [{"start_time": 0.0, "end_time": 2.0}]
        ground_truth = [{"start_time": 5.0, "end_time": 7.0}]

        matches = evaluator.match_segments(predicted, ground_truth)
        assert len(matches) == 1
        assert matches[0].hit is False

    def test_match_segments_one_to_one_greedy(self):
        evaluator = HighlightEvaluator(iou_threshold=0.5)
        predicted = [
            {"start_time": 0.0, "end_time": 4.0},
            {"start_time": 6.0, "end_time": 10.0},
        ]
        ground_truth = [
            {"start_time": 0.5, "end_time": 3.5},
            {"start_time": 6.5, "end_time": 9.5},
        ]

        matches = evaluator.match_segments(predicted, ground_truth)
        assert len(matches) == 2
        assert all(m.hit for m in matches)

    def test_score_case_basic(self):
        evaluator = HighlightEvaluator()
        predicted = [{"start_time": 0.0, "end_time": 5.0}]
        ground_truth = [{"start_time": 0.0, "end_time": 5.0}]

        score = evaluator.score_case(
            "case_001", predicted, ground_truth,
            category="体育", difficulty="easy", source_type="local",
        )

        assert score.case_id == "case_001"
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.f1 == 1.0
        assert score.error is None

    def test_score_case_empty_gt(self):
        evaluator = HighlightEvaluator()
        score = evaluator.score_case("case_001", [], [])
        assert score.error == "ground_truth 为空"

    def test_score_case_empty_pred(self):
        evaluator = HighlightEvaluator()
        score = evaluator.score_case(
            "case_001", [],
            [{"start_time": 0.0, "end_time": 5.0}],
        )
        assert score.precision == 0.0
        assert score.recall == 0.0

    def test_evaluate_all(self):
        evaluator = HighlightEvaluator()
        results = [
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
            {
                "case_id": "case_002",
                "category": "游戏",
                "difficulty": "medium",
                "source_type": "remote",
                "predicted": [{"start_time": 2.0, "end_time": 6.0}],
                "ground_truth": [{"start_time": 2.0, "end_time": 6.0}],
            },
        ]

        report = evaluator.evaluate_all(results)
        assert report.overall_f1 == 1.0
        assert len(report.scores) == 2
        assert "体育" in report.by_category
        assert "游戏" in report.by_category
        assert "easy" in report.by_difficulty
        assert "medium" in report.by_difficulty
        assert "local" in report.by_source
        assert "remote" in report.by_source

    def test_evaluate_all_empty(self):
        evaluator = HighlightEvaluator()
        report = evaluator.evaluate_all([])
        assert report.overall_f1 == 0.0


class TestLLMJudge:
    def test_build_prompt(self):
        judge = LLMJudge()
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "score": 0.9, "label": "进球"},
        ]
        prompt = judge.build_prompt("体育", "进球集锦", "快节奏", segments)

        assert "体育" in prompt
        assert "进球集锦" in prompt
        assert "快节奏" in prompt
        assert "0.0s - 5.0s" in prompt
        assert "进球" in prompt

    def test_build_prompt_no_target_style(self):
        judge = LLMJudge()
        segments: list = []
        prompt = judge.build_prompt("体育", "", "", segments)
        assert "精彩集锦" in prompt
        assert "无特定要求" in prompt

    def test_judge_score_average(self):
        score = JudgeScore(rhythm=4.0, completeness=3.0, excitement=5.0, instruction_fit=4.0)
        assert score.average == 4.0

    def test_judge_score_error(self):
        score = JudgeScore(error="API 不可用")
        assert score.error == "API 不可用"
        assert score.average == 0.0

    def test_judge_all_empty(self):
        judge = LLMJudge()
        report = judge.judge_all([])
        assert report.overall_average == 0.0

    def test_format_judge_report(self):
        report = JudgeReport(
            scores=[
                JudgeScore(rhythm=4.0, completeness=4.0, excitement=5.0, instruction_fit=4.0, overall_comment="剪辑质量优秀"),
            ],
            overall_rhythm=4.0,
            overall_completeness=4.0,
            overall_excitement=5.0,
            overall_instruction_fit=4.0,
            overall_average=4.25,
        )
        formatted = format_judge_report(report)
        assert "4.00 / 5.0" in formatted
        assert "剪辑质量优秀" in formatted

    def test_format_judge_report_with_error(self):
        report = JudgeReport(
            scores=[JudgeScore(error="API 不可用")],
        )
        formatted = format_judge_report(report)
        assert "[ERROR]" in formatted
        assert "API 不可用" in formatted


class TestReportGenerator:
    def test_generate_text_report(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])

        judge_report = JudgeReport(
            scores=[
                JudgeScore(rhythm=4.0, completeness=4.0, excitement=5.0, instruction_fit=4.0, overall_comment="不错"),
            ],
            overall_rhythm=4.0,
            overall_completeness=4.0,
            overall_excitement=5.0,
            overall_instruction_fit=4.0,
            overall_average=4.25,
        )

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=False))
        text = gen.generate(eval_report, judge_report)

        assert "视频高光剪辑" in text
        assert "case_001" in text
        assert "4.25" in text
        assert "不错" in text

    def test_generate_json_report(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])

        judge_report = JudgeReport(
            scores=[JudgeScore(rhythm=4.0, completeness=4.0, excitement=5.0, instruction_fit=4.0, overall_comment="不错")],
            overall_rhythm=4.0,
            overall_completeness=4.0,
            overall_excitement=5.0,
            overall_instruction_fit=4.0,
            overall_average=4.25,
        )

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=False))
        gen.generate(eval_report, judge_report)

        json_path = tmp_path / "report.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["iou_eval"]["overall_f1"] == 1.0
        assert data["llm_judge"]["overall_average"] == 4.25

    def test_generate_charts(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])

        judge_report = JudgeReport(
            scores=[JudgeScore(rhythm=4.0, completeness=4.0, excitement=5.0, instruction_fit=4.0, overall_comment="不错")],
            overall_rhythm=4.0,
            overall_completeness=4.0,
            overall_excitement=5.0,
            overall_instruction_fit=4.0,
            overall_average=4.25,
        )

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=True))
        gen.generate(eval_report, judge_report)

        chart_path = tmp_path / "charts.png"
        assert chart_path.exists()

    def test_report_config_defaults(self):
        cfg = ReportConfig()
        assert cfg.save_charts is True
        assert cfg.save_json is True


class TestTestCaseLoader:
    def test_load_local_cases(self, tmp_path):
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        cases_yaml = local_dir / "cases.yaml"
        cases_yaml.write_text("""
cases:
  - id: "case_001"
    category: 体育
    difficulty: easy
    description: "测试用例"
    video_file: "video.mp4"
    instruction:
      target: "进球集锦"
      duration_limit: "60秒以内"
      style: "快节奏"
""", encoding="utf-8")

        case_dir = local_dir / "case_001"
        case_dir.mkdir()
        (case_dir / "instruction.json").write_text(
            '{"target": "进球集锦", "duration_limit": "60秒以内", "style": "快节奏"}',
            encoding="utf-8",
        )
        (case_dir / "ground_truth.json").write_text(
            '{"highlights": [{"start_time": 0.0, "end_time": 5.0, "label": "进球", "score": 0.95}]}',
            encoding="utf-8",
        )

        loader = TestCaseLoader(str(tmp_path))
        cases = loader.load_local_cases()

        assert len(cases) == 1
        assert cases[0]["case_id"] == "case_001"
        assert cases[0]["category"] == "体育"
        assert cases[0]["source_type"] == "local"
        assert len(cases[0]["ground_truth"]) == 1

    def test_load_remote_cases(self, tmp_path):
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        cases_yaml = remote_dir / "cases.yaml"
        cases_yaml.write_text("""
cases:
  - id: "case_021"
    category: 体育
    difficulty: medium
    description: "远程测试"
    source_url: "https://example.com/video.mp4"
    instruction:
      target: "精彩回合"
      duration_limit: "60秒以内"
      style: ""
""", encoding="utf-8")

        case_dir = remote_dir / "case_021"
        case_dir.mkdir()
        (case_dir / "instruction.json").write_text(
            '{"target": "精彩回合", "duration_limit": "60秒以内", "style": ""}',
            encoding="utf-8",
        )
        (case_dir / "ground_truth.json").write_text(
            '{"highlights": []}',
            encoding="utf-8",
        )

        loader = TestCaseLoader(str(tmp_path))
        cases = loader.load_remote_cases()

        assert len(cases) == 1
        assert cases[0]["case_id"] == "case_021"
        assert cases[0]["source_type"] == "remote"
        assert cases[0]["source_url"] == "https://example.com/video.mp4"

    def test_load_all(self, tmp_path):
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        (local_dir / "cases.yaml").write_text("cases: []", encoding="utf-8")

        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        (remote_dir / "cases.yaml").write_text("cases: []", encoding="utf-8")

        loader = TestCaseLoader(str(tmp_path))
        cases = loader.load_all()
        assert cases == []

    def test_load_missing_yaml(self, tmp_path):
        loader = TestCaseLoader(str(tmp_path))
        cases = loader.load_local_cases()
        assert cases == []
