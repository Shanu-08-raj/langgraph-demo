"""State definition for the LangGraph workflow.

This module defines the shared state schema used across all agent nodes
in the research, writing, and review pipeline.
"""

from typing import TypedDict


class WorkflowState(TypedDict):
    """Shared state for the fixed sequential LangGraph workflow.

    In LangGraph, the shared state acts as a central data blackboard.
    Each node in the sequence (Researcher -> Writer -> Reviewer) reads
    the state populated by earlier nodes and returns a dictionary containing
    only the specific fields it modifies or produces. LangGraph updates the
    shared state with these returned values.
    """

    topic: str
    evidence: str
    draft: str
    feedback: str
    score: int
    revision_count: int
    status: str
