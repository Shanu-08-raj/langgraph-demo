"""Unit and integration tests for the LangGraph State and Workflow Designer.

All unit tests use mock LLM responses to ensure reliable, fast, and
reproducible test runs without requiring a live Ollama server or network access.
"""

from io import StringIO
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import START, END

from app.state import WorkflowState
from app.nodes import (
    researcher_node,
    writer_node,
    reviewer_node,
    _extract_score_and_feedback,
)
from app.graph import create_workflow_graph, workflow_graph, get_mermaid_diagram
from app.main import run_workflow, print_final_state


# ============================================================================
# Phase 2: Node Unit Tests (with Mocked LLM)
# ============================================================================

def test_researcher_node_updates_expected_fields():
    """Verify researcher_node returns only 'evidence' and 'status'."""
    initial_state: WorkflowState = {
        "topic": "Quantum Computing",
        "evidence": "",
        "draft": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "initialized",
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Quantum computing uses qubits and superposition to solve complex problems."
    )

    with patch("app.nodes.get_llm", return_value=mock_llm):
        result = researcher_node(initial_state)

    # Must return only the fields it updates
    assert set(result.keys()) == {"evidence", "status"}
    assert result["status"] == "research_completed"
    assert "qubits" in result["evidence"]
    mock_llm.invoke.assert_called_once()


def test_researcher_node_handles_empty_topic():
    """Verify researcher_node handles an empty topic gracefully."""
    initial_state: WorkflowState = {
        "topic": "   ",
        "evidence": "",
        "draft": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "initialized",
    }

    result = researcher_node(initial_state)
    assert result["status"] == "research_failed"
    assert "Error" in result["evidence"]


def test_writer_node_updates_expected_fields():
    """Verify writer_node returns only 'draft' and 'status'."""
    state: WorkflowState = {
        "topic": "Artificial Intelligence in Healthcare",
        "evidence": "AI aids early disease diagnosis through image analysis.",
        "draft": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "research_completed",
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Introduction: AI is revolutionizing healthcare.\n\n"
        "Explanation: Computer vision identifies anomalies in scans.\n\n"
        "Conclusion: AI enhances clinician efficiency."
    )

    with patch("app.nodes.get_llm", return_value=mock_llm):
        result = writer_node(state)

    # Must return only the fields it updates
    assert set(result.keys()) == {"draft", "status"}
    assert result["status"] == "writing_completed"
    assert "Introduction" in result["draft"]
    mock_llm.invoke.assert_called_once()


def test_reviewer_node_updates_expected_fields():
    """Verify reviewer_node returns only 'feedback', 'score', 'revision_count', and 'status'."""
    state: WorkflowState = {
        "topic": "Photosynthesis",
        "evidence": "Plants convert light energy into chemical energy.",
        "draft": "Introduction: Plants produce oxygen.\nExplanation: Light reactions.\nConclusion: Essential for life.",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "writing_completed",
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="FEEDBACK:\nThe draft is clear, well-structured, and consistent with the evidence.\n\nSCORE: 9"
    )

    with patch("app.nodes.get_llm", return_value=mock_llm):
        result = reviewer_node(state)

    # Must return only the fields it updates
    assert set(result.keys()) == {"feedback", "score", "revision_count", "status"}
    assert result["status"] == "completed"
    assert result["revision_count"] == 0
    assert isinstance(result["score"], int)
    assert result["score"] == 9
    assert "clear" in result["feedback"]


@pytest.mark.parametrize(
    "raw_text,expected_score,expected_feedback_snippet",
    [
        ("FEEDBACK: Great work.\nSCORE: 10", 10, "Great work"),
        ("FEEDBACK: Mediocre.\nSCORE: 5", 5, "Mediocre"),
        ("FEEDBACK: Off the charts.\nSCORE: 15", 10, "Off the charts"),  # Clamped to 10
        ("FEEDBACK: Terrible.\nSCORE: 0", 1, "Terrible"),  # Clamped to 1
        ("Draft was okay but unformatted.", 7, "Draft was okay"),  # Default fallback
        ("Score: 8\nFeedback: Very thorough.", 8, "Very thorough"),
        (
            "FEEDBACK:\nBinary Search Algorithm: A Comprehensive Overview\n\nSCORE: 9\n\nDetailed Evaluation:\nThe draft is clear, complete, and accurate.",
            9,
            "The draft is clear, complete, and accurate",
        ),
    ],
)
def test_reviewer_score_and_feedback_parsing(raw_text, expected_score, expected_feedback_snippet):
    """Test score extraction, validation (1-10 range clamping), and fallback behavior."""
    feedback, score = _extract_score_and_feedback(raw_text)
    assert isinstance(score, int)
    assert 1 <= score <= 10
    assert score == expected_score
    assert expected_feedback_snippet.lower() in feedback.lower()


def test_node_llm_exception_handling():
    """Verify nodes propagate exceptions cleanly when LLM fails."""
    state: WorkflowState = {
        "topic": "Failing Topic",
        "evidence": "Some evidence",
        "draft": "Some draft",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "initialized",
    }

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Ollama connection timed out")

    with patch("app.nodes.get_llm", return_value=mock_llm):
        with pytest.raises(RuntimeError):
            researcher_node(state)

        with pytest.raises(RuntimeError):
            writer_node(state)

        with pytest.raises(RuntimeError):
            reviewer_node(state)


# ============================================================================
# Phase 3: Graph Construction and Topology Tests
# ============================================================================

def test_graph_compiles_successfully():
    """Verify the workflow graph compiles and is ready for execution."""
    compiled = create_workflow_graph()
    assert compiled is not None
    assert workflow_graph is not None


def test_graph_contains_exact_three_agent_nodes():
    """Verify the graph registers exactly researcher, writer, and reviewer nodes."""
    graph_repr = workflow_graph.get_graph()
    nodes = set(graph_repr.nodes.keys())
    expected_agents = {"researcher", "writer", "reviewer"}
    assert expected_agents.issubset(nodes)
    agent_nodes = {n for n in nodes if not n.startswith("__")}
    assert agent_nodes == expected_agents
    assert len(agent_nodes) == 3


def test_graph_has_required_sequential_edges():
    """Verify the sequential edge topology: START -> researcher -> writer -> reviewer -> END."""
    graph_repr = workflow_graph.get_graph()
    edges = [(edge.source, edge.target) for edge in graph_repr.edges]

    assert ("__start__", "researcher") in edges
    assert ("researcher", "writer") in edges
    assert ("writer", "reviewer") in edges
    assert ("reviewer", "__end__") in edges


def test_mermaid_diagram_generation():
    """Verify Mermaid diagram generation returns valid mermaid content."""
    diagram = get_mermaid_diagram()
    assert isinstance(diagram, str)
    assert len(diagram) > 0
    assert "researcher" in diagram.lower()
    assert "writer" in diagram.lower()
    assert "reviewer" in diagram.lower()


# ============================================================================
# Phase 4: Main Application & Integration Tests (Mocked LLM)
# ============================================================================

def test_full_graph_invocation_reaches_final_state():
    """Verify complete end-to-end graph invocation updates all fields properly."""
    initial_state: WorkflowState = {
        "topic": "Graph Neural Networks",
        "evidence": "",
        "draft": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "initialized",
    }

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="GNNs operate on non-Euclidean graph-structured data using message passing."),
        AIMessage(content="Introduction: GNNs process relational graphs.\n\n"
                          "Explanation: Message passing aggregates node neighbor features.\n\n"
                          "Conclusion: GNNs excel in molecular modeling and social network analysis."),
        AIMessage(content="FEEDBACK:\nExcellent, comprehensive explanation with clear structure.\n\nSCORE: 9"),
    ]

    with patch("app.nodes.get_llm", return_value=mock_llm):
        final_state = workflow_graph.invoke(initial_state)

    # All seven required state fields are present
    required_fields = {"topic", "evidence", "draft", "feedback", "score", "revision_count", "status"}
    assert set(final_state.keys()) == required_fields
    assert final_state["topic"] == "Graph Neural Networks"
    assert "message passing" in final_state["evidence"]
    assert "Introduction" in final_state["draft"]
    assert "comprehensive explanation" in final_state["feedback"]
    assert isinstance(final_state["score"], int)
    assert 1 <= final_state["score"] <= 10
    assert final_state["score"] == 9
    assert final_state["revision_count"] == 0
    assert final_state["status"] == "completed"
    assert mock_llm.invoke.call_count == 3


def test_run_workflow_empty_topic_handled():
    """Verify run_workflow rejects empty topic."""
    result = run_workflow("   ")
    assert result is None


def test_run_workflow_ollama_unavailable():
    """Verify run_workflow exits gracefully with error message when Ollama is offline."""
    with patch("app.main.check_ollama_availability", return_value=False):
        result = run_workflow("Valid Topic")
        assert result is None


def test_run_workflow_success_mocked():
    """Verify run_workflow runs and returns completed state when Ollama is online."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="Evidence about Cloud Computing."),
        AIMessage(content="Draft about Cloud Computing."),
        AIMessage(content="FEEDBACK: Clear.\nSCORE: 8"),
    ]

    with patch("app.main.check_ollama_availability", return_value=True), \
         patch("app.nodes.get_llm", return_value=mock_llm):
        final_state = run_workflow("Cloud Computing")

    assert final_state is not None
    assert final_state["status"] == "completed"
    assert final_state["score"] == 8
    assert final_state["revision_count"] == 0


def test_print_final_state_output(capsys):
    """Verify print_final_state formats all required fields without crashing."""
    sample_state: WorkflowState = {
        "topic": "Microservices",
        "evidence": "Microservices decompose apps into small services.",
        "draft": "Intro, explanation, conclusion.",
        "feedback": "Well done.",
        "score": 9,
        "revision_count": 0,
        "status": "completed",
    }
    print_final_state(sample_state)
    captured = capsys.readouterr().out
    assert "Microservices" in captured
    assert "Researcher -> Writer -> Reviewer" in captured
    assert "9 / 10" in captured
    assert "completed" in captured


# ============================================================================
# Optional Real Integration Test (Requires Running Ollama)
# ============================================================================

@pytest.mark.integration
def test_real_ollama_integration():
    """Optional end-to-end integration test against a live local Ollama service.

    This test is clearly separated from standard unit tests using the
    @pytest.mark.integration marker. If Ollama is not running or unreachable,
    it automatically skips so the standard test suite never fails.
    """
    from app.llm import check_ollama_availability

    if not check_ollama_availability():
        pytest.skip("Local Ollama service is not reachable at configured endpoint.")

    initial_state: WorkflowState = {
        "topic": "Recursion in Computer Science",
        "evidence": "",
        "draft": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "initialized",
    }

    final_state = workflow_graph.invoke(initial_state)

    assert final_state["status"] == "completed"
    assert len(final_state["evidence"]) > 0
    assert len(final_state["draft"]) > 0
    assert len(final_state["feedback"]) > 0
    assert isinstance(final_state["score"], int)
    assert 1 <= final_state["score"] <= 10
    assert final_state["revision_count"] == 0

