"""Command-line entry point: ``python -m app.evaluation``.

Spends real provider quota. Each topic costs one Tavily search, a full
workflow run, one baseline call and four evaluator calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import BACKEND_DIR, get_settings
from app.core.exceptions import AgentFlowError
from app.core.logging import configure_logging
from app.evaluation.report import render_markdown
from app.evaluation.runner import RateLimitPolicy, evaluate
from app.models.enums import Platform, Tone
from app.schemas.workflow import MAX_POSTS, MIN_POSTS
from app.services.llm_service import build_llm_service
from app.services.pricing import resolve_pricing
from app.services.research_service import build_research_service
from app.services.retrieval_service import build_optional_retrieval_service

DEFAULT_TOPICS = BACKEND_DIR / "evals" / "topics.json"
DEFAULT_OUT = BACKEND_DIR / "evals" / "results"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation",
        description="Compare the multi-agent workflow against a single-agent baseline.",
    )
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS, help="JSON list of topics.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N topics.")
    parser.add_argument("--posts", type=int, default=1, help="Posts per topic (default 1).")
    parser.add_argument("--tone", choices=[t.value for t in Tone], default=Tone.PROFESSIONAL.value)
    parser.add_argument(
        "--pause",
        type=float,
        default=60.0,
        help="Seconds to wait between topics so per-minute quotas can refill (default 60).",
    )
    parser.add_argument(
        "--rate-limit-wait",
        type=float,
        default=60.0,
        help="Seconds to wait after a rate limit before retrying a call (default 60).",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=2,
        help="Retries per call after a rate limit (default 2).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Directory for results.")
    args = parser.parse_args(argv)
    if not MIN_POSTS <= args.posts <= MAX_POSTS:
        parser.error(f"--posts must be between {MIN_POSTS} and {MAX_POSTS}")
    return args


def _load_topics(path: Path, limit: int | None) -> list[str]:
    topics = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(topics, list) or not all(isinstance(t, str) and t.strip() for t in topics):
        raise ValueError(f"{path} must be a JSON list of non-empty strings.")
    return topics[:limit] if limit else topics


def _progress(index: int, total: int, topic: str) -> None:
    print(f"[{index}/{total}] {topic}", flush=True)


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging("WARNING", json_output=False)

    try:
        llm = build_llm_service(settings)
        research = build_research_service(settings)
    except AgentFlowError as exc:
        print(f"Configuration error: {exc.message}", file=sys.stderr)
        return 2

    topics = _load_topics(args.topics, args.limit)
    report = await evaluate(
        llm,
        research,
        topics=topics,
        platform=Platform.LINKEDIN,
        tone=Tone(args.tone),
        number_of_posts=args.posts,
        settings=settings,
        retrieval=build_optional_retrieval_service(settings),
        pricing=resolve_pricing(settings),
        pause_seconds=args.pause,
        rate_limits=RateLimitPolicy(
            wait_seconds=args.rate_limit_wait, retries=max(0, args.rate_limit_retries)
        ),
        on_progress=_progress,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    json_path = args.out / f"evaluation-{stamp}.json"
    md_path = args.out / f"evaluation-{stamp}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"\nWrote {md_path}\nWrote {json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
