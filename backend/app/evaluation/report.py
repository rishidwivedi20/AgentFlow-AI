"""Renders an evaluation report as Markdown for reading and committing."""

from __future__ import annotations

from app.evaluation.models import EvaluationReport, SystemName, SystemOutcome, SystemSummary

_LABELS = {SystemName.MULTI_AGENT: "Multi-agent", SystemName.SINGLE_AGENT: "Single agent"}


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _secs(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}s"


def _usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def _summary_table(summaries: list[SystemSummary]) -> list[str]:
    multi, single = (
        next(s for s in summaries if s.system is name)
        for name in (SystemName.MULTI_AGENT, SystemName.SINGLE_AGENT)
    )
    rows = [
        ("Topics completed", str(multi.topics_completed), str(single.topics_completed)),
        ("Quality wins", str(multi.quality_wins), str(single.quality_wins)),
        ("Mean quality (0-10)", _num(multi.mean_quality), _num(single.mean_quality)),
        (
            "Claims verified",
            _pct(multi.mean_supported_rate),
            _pct(single.mean_supported_rate),
        ),
        (
            "Unsupported claims / topic",
            _num(multi.mean_unsupported_per_topic),
            _num(single.mean_unsupported_per_topic),
        ),
        ("Mean tokens", _num(multi.mean_tokens, 0), _num(single.mean_tokens, 0)),
        (
            "Mean generation latency",
            _secs(multi.mean_latency_seconds),
            _secs(single.mean_latency_seconds),
        ),
        ("Mean LLM calls", _num(multi.mean_llm_calls, 1), _num(single.mean_llm_calls, 1)),
        ("Mean cost (list price)", _usd(multi.mean_cost_usd), _usd(single.mean_cost_usd)),
    ]
    lines = ["| Metric | Multi-agent | Single agent |", "|---|---|---|"]
    lines += [f"| {label} | {a} | {b} |" for label, a, b in rows]
    return lines


def _dimension_table(summaries: list[SystemSummary]) -> list[str]:
    names = sorted({name for s in summaries for name in s.mean_dimensions})
    if not names:
        return []
    by_system = {s.system: s for s in summaries}
    lines = ["| Dimension | Multi-agent | Single agent |", "|---|---|---|"]
    for name in names:
        lines.append(
            f"| {name.replace('_', ' ')} "
            f"| {_num(by_system[SystemName.MULTI_AGENT].mean_dimensions.get(name))} "
            f"| {_num(by_system[SystemName.SINGLE_AGENT].mean_dimensions.get(name))} |"
        )
    return lines


def _outcome_cell(outcome: SystemOutcome) -> str:
    if not outcome.succeeded:
        return f"failed: {outcome.error}"
    quality = _num(outcome.quality.overall, 1) if outcome.quality else "—"
    grounding = outcome.grounding
    claims = (
        f"{grounding.verified}/{grounding.claims} verified, {grounding.unsupported} unsupported"
        if grounding
        else "not checked"
    )
    return f"{quality} · {claims} · {outcome.total_tokens:,} tok · {outcome.latency_seconds:.1f}s"


def render_markdown(report: EvaluationReport) -> str:
    """The whole report, headline first."""
    summaries = report.summaries
    pricing = (
        f"{report.pricing.model}: ${report.pricing.input_per_mtok}/M input, "
        f"${report.pricing.output_per_mtok}/M output ({report.pricing.source})"
        if report.pricing
        else "model not in the price table; costs not estimated"
    )

    lines = [
        "# Single-agent vs multi-agent evaluation",
        "",
        f"- **Model:** {report.provider} / {report.model}",
        f"- **Topics:** {len(report.results)} · **posts per topic:** {report.number_of_posts}"
        f" · **revision cap:** {report.max_revisions}"
        f" · **retrieval:** {'on' if report.retrieval_enabled else 'off'}",
        f"- **Pricing:** {pricing}",
        f"- **Run:** {report.started_at:%Y-%m-%d %H:%M} to {report.finished_at:%H:%M} UTC",
        "",
        "## Summary",
        "",
        *_summary_table(summaries),
        "",
        f"Ties on quality: {report.ties}.",
        "",
    ]

    dimensions = _dimension_table(summaries)
    if dimensions:
        lines += ["## Quality by dimension", "", *dimensions, ""]

    lines += [
        "## Per topic",
        "",
        "Each cell: quality · grounding · tokens · generation latency.",
        "",
        "| Topic | Multi-agent | Single agent | Winner |",
        "|---|---|---|---|",
    ]
    for result in report.results:
        winner = _LABELS[result.quality_winner] if result.quality_winner else "tie / n/a"
        topic = result.topic.replace("|", "\\|")
        lines.append(
            f"| {topic} | {_outcome_cell(result.multi_agent)} "
            f"| {_outcome_cell(result.single_agent)} | {winner} |"
        )

    problems = [
        f"- {result.topic}: {error}"
        for result in report.results
        for error in result.evaluator_errors
    ]
    if problems:
        lines += [
            "",
            "## Evaluator errors",
            "",
            "Missing scores are excluded, never counted as zero.",
            "",
        ]
        lines += problems

    lines += [
        "",
        "## Method",
        "",
        "- Each topic is searched once. The multi-agent graph runs unchanged; the single "
        "agent then writes from the same sources and retrieved excerpts in one call.",
        "- Quality is scored by a neutral rubric judge that is not told which system "
        "wrote the posts. It is not the workflow's Editor, which the multi-agent output "
        "was revised to satisfy.",
        "- Grounding re-checks both outputs with the Fact Checker against the raw search "
        "snippets and excerpts, not against either system's own research notes.",
        "- Latency, tokens and cost exclude the shared search and retrieval, and the "
        "evaluators' own calls.",
        "- The judge is the same model as the systems under test, which can favour its "
        "own style. Small topic sets are noisy: treat differences under a point as "
        "inconclusive.",
        "",
    ]
    return "\n".join(lines)
