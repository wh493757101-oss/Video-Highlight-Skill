"""双路径验证 — 本地文件 + URL 下载两条路径分别验证"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("dual_verify")


def verify_local(video_path: str, description: str):
    from src.main import VideoHighlightPipeline, PipelineConfig
    from src.video_fetcher import LocalFileSource

    logger.info("=" * 60)
    logger.info("Path 1: local file")
    logger.info("  video: %s", video_path)
    logger.info("  prompt: %s", description)
    logger.info("=" * 60)

    pipeline = VideoHighlightPipeline(PipelineConfig(output_dir="./output"))
    result = pipeline.run(
        LocalFileSource(video_path),
        description=description,
        skip_edit=True,
    )

    if result.error:
        logger.error("FAIL: %s", result.error)
        return False

    logger.info("OK: source=%s, segments=%d, duration=%.1fs, degraded=%s",
                result.detection.source, len(result.detection.segments),
                result.metadata.duration, bool(result.degradations))
    for d in result.degradations:
        logger.info("  degrade: %s -> %s (%s)", d.from_path, d.to_path, d.reason)
    for i, seg in enumerate(result.detection.segments[:5]):
        logger.info("  #%d: %.1fs - %.1fs (%.2f)", i + 1, seg.start_time, seg.end_time, seg.combined_score)
    return True


def verify_url(url: str, description: str):
    from src.main import VideoHighlightPipeline, PipelineConfig
    from src.video_fetcher import UrlSource

    logger.info("=" * 60)
    logger.info("Path 2: URL download")
    logger.info("  url: %s", url)
    logger.info("  prompt: %s", description)
    logger.info("=" * 60)

    pipeline = VideoHighlightPipeline(PipelineConfig(output_dir="./output"))
    try:
        result = pipeline.run(
            UrlSource(url),
            description=description,
            skip_edit=True,
        )
    except Exception as e:
        logger.error("FAIL: %s", e)
        return False

    if result.error:
        logger.error("FAIL: %s", result.error)
        return False

    logger.info("OK: source=%s, segments=%d, duration=%.1fs, degraded=%s",
                result.detection.source, len(result.detection.segments),
                result.metadata.duration, bool(result.degradations))
    for d in result.degradations:
        logger.info("  degrade: %s -> %s (%s)", d.from_path, d.to_path, d.reason)
    for i, seg in enumerate(result.detection.segments[:5]):
        logger.info("  #%d: %.1fs - %.1fs (%.2f)", i + 1, seg.start_time, seg.end_time, seg.combined_score)
    return True


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import yaml

    cases_yaml = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "video-highlight-bucket", "test_cases", "self-built_data", "cases.yaml"
    )

    if not os.path.exists(cases_yaml):
        logger.error("cases.yaml not found: %s", cases_yaml)
        sys.exit(1)

    with open(cases_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cases_dir = os.path.dirname(cases_yaml)
    results = {"local": [], "url": []}

    for case in data.get("cases", []):
        case_id = case["id"]
        video_path = os.path.join(cases_dir, case_id, case.get("video_file", "video.mp4"))
        source_url = case.get("source_url", "")
        prompt = case.get("instruction", {}).get("prompt", "") or "帮我把精彩片段剪成60秒集锦，节奏要快"

        logger.info("\n=== Case: %s ===", case_id)

        if os.path.exists(video_path):
            ok = verify_local(video_path, prompt)
            results["local"].append((case_id, ok))
        else:
            logger.warning("Skip local: file not found %s", video_path)

        if source_url:
            ok = verify_url(source_url, prompt)
            results["url"].append((case_id, ok))
        else:
            logger.warning("Skip url: no source_url")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("local:", ", ".join(f"{cid}={ok}" for cid, ok in results["local"]))
    print("url:", ", ".join(f"{cid}={ok}" for cid, ok in results["url"]))


if __name__ == "__main__":
    main()
