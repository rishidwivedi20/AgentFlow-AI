"""Phase 6: single-agent baseline versus the multi-agent workflow.

Run from the ``backend/`` directory:

    python -m app.evaluation --topics evals/topics.json

Each topic is searched once. The multi-agent graph runs as it does in
production, and the single-agent baseline then writes from the same sources
and evidence, so neither system benefits from a luckier search. Both outputs
are scored by the same evaluators, neither of which is told which system wrote
what.
"""
