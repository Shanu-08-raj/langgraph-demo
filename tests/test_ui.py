"""Unit tests for the Streamlit UI integration and helper functions."""

from unittest.mock import MagicMock, patch
import pytest

from app.ui import init_workflow_state, execute_workflow
from app.state import WorkflowState


def test_init_workflow_state():
    """Verify init_workflow_state initializes all 7 fields with correct defaults."""
    state = init_workflow_state("  Neural Networks  ")
    assert state["topic"] == "Neural Networks"
    assert state["evidence"] == ""
    assert state["draft"] == ""
    assert state["feedback"] == ""
    assert state["score"] == 0
    assert state["revision_count"] == 0
    assert state["status"] == "initialized"
    assert len(state) == 7


def test_execute_workflow_mocked_streaming():
    """Verify execute_workflow streams updates and returns complete final state."""
    mock_steps = [
        {"researcher": {"evidence": "Neural networks learn representations.", "status": "research_completed"}},
        {"writer": {"draft": "Introduction: Networks\nExplanation: Layers\nConclusion: Powerful", "status": "writing_completed"}},
        {"reviewer": {"feedback": "Clear explanation.", "score": 9, "revision_count": 0, "status": "completed"}},
    ]

    mock_graph = MagicMock()
    mock_graph.stream.return_value = mock_steps
    mock_placeholder = MagicMock()

    with patch("app.ui.workflow_graph", mock_graph):
        final_state = execute_workflow("Neural Networks", mock_placeholder)

    assert final_state is not None
    assert final_state["topic"] == "Neural Networks"
    assert final_state["evidence"] == "Neural networks learn representations."
    assert "Introduction" in final_state["draft"]
    assert final_state["feedback"] == "Clear explanation."
    assert final_state["score"] == 9
    assert final_state["revision_count"] == 0
    assert final_state["status"] == "completed"
    assert mock_placeholder.markdown.call_count == 3


def test_execute_workflow_exception_handling():
    """Verify execute_workflow raises cleanly when graph streaming fails."""
    mock_graph = MagicMock()
    mock_graph.stream.side_effect = RuntimeError("Streaming connection failed")

    with patch("app.ui.workflow_graph", mock_graph):
        with pytest.raises(RuntimeError):
            execute_workflow("Faulty Topic")
