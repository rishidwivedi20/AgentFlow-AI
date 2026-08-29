"""Agent implementations.

Each agent is a plain async function that receives what it needs, calls the LLM
service and returns a validated result. Metrics, error capture and event
emission are added by the wrapper in ``app.services.metrics_service`` - never
inline here. No agent contains orchestration logic.
"""

from app.agents.editor import run_editor
from app.agents.writer import run_writer

__all__ = ["run_editor", "run_writer"]
