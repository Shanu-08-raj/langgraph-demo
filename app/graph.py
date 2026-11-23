"""LangGraph workflow definition for the Research-Writer-Reviewer pipeline.

Constructs a fixed, sequential StateGraph:
    START -> researcher -> writer -> reviewer -> END
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from app.state import WorkflowState
from app.nodes import researcher_node, writer_node, reviewer_node

logger = logging.getLogger(__name__)


def create_workflow_graph() -> CompiledStateGraph:
    """Build and compile the sequential LangGraph agent workflow.

    Workflow topology:
        START -> researcher -> writer -> reviewer -> END

    Returns:
        CompiledStateGraph: Executable LangGraph instance ready for invocation.
    """
    logger.info("Initializing StateGraph with WorkflowState schema")
    workflow = StateGraph(WorkflowState)

    # Register the three agent nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # Define the fixed sequential edges
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", END)

    logger.info("Compiling workflow graph")
    compiled_graph = workflow.compile()
    return compiled_graph


# Module-level compiled graph instance for direct import
workflow_graph: CompiledStateGraph = create_workflow_graph()


def get_mermaid_diagram() -> str:
    """Generate the Mermaid representation from the compiled graph.

    Uses LangGraph's built-in get_graph().draw_mermaid() if available,
    falling back to a static Mermaid flowchart.

    Returns:
        str: Mermaid diagram specification.
    """
    try:
        return workflow_graph.get_graph().draw_mermaid()
    except Exception as exc:
        logger.warning("Could not generate dynamic Mermaid from graph: %s", exc)
        return (
            "graph TD\n"
            "    START([START]) --> researcher[Researcher]\n"
            "    researcher --> writer[Writer]\n"
            "    writer --> reviewer[Reviewer]\n"
            "    reviewer --> END([END])\n"
        )
