"""Service layer: LLM abstraction, metrics, run registry.

Agents depend on these services rather than on provider SDKs, so swapping a
provider touches one adapter instead of every agent.
"""
