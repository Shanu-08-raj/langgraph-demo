"""Streamlit browser-based UI for the LangGraph State and Workflow Designer.

Provides a modern, interactive dashboard to execute and visualize the
three-agent sequential pipeline: Researcher -> Writer -> Reviewer.
"""

import logging
from typing import Any, Dict, Optional
import streamlit as st

from app import setup_logging
from app.graph import get_mermaid_diagram, workflow_graph
from app.llm import check_ollama_availability, get_base_url, get_model_name
from app.state import WorkflowState

logger = logging.getLogger("app.ui")
setup_logging()


def init_workflow_state(topic: str) -> WorkflowState:
    """Create the initial 7-field WorkflowState dictionary.

    Args:
        topic: User-provided research topic.

    Returns:
        WorkflowState: Initial state dictionary.
    """
    return {
        "topic": topic.strip(),
        "evidence": "",
        "draft": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "initialized",
    }


def execute_workflow(topic: str, status_placeholder: Optional[Any] = None) -> Optional[WorkflowState]:
    """Execute the compiled LangGraph workflow with streaming step updates.

    Args:
        topic: Research topic to evaluate.
        status_placeholder: Optional Streamlit status container for live feedback.

    Returns:
        Optional[WorkflowState]: Complete final state or None on error.
    """
    current_state: WorkflowState = init_workflow_state(topic)

    try:
        # Use streaming to track each sequential node execution in real-time
        for step in workflow_graph.stream(current_state):
            for node_name, state_update in step.items():
                current_state.update(state_update)
                if status_placeholder:
                    if node_name == "researcher":
                        status_placeholder.markdown("✅ **Researcher Agent**: Evidence collected successfully.")
                    elif node_name == "writer":
                        status_placeholder.markdown("✅ **Writer Agent**: Structured draft generated.")
                    elif node_name == "reviewer":
                        status_placeholder.markdown("✅ **Reviewer Agent**: Evaluation and scoring completed.")

        return current_state
    except Exception as exc:
        logger.exception("UI Workflow execution failed: %s", exc)
        raise exc


def render_pipeline_visualization() -> None:
    """Render a visual representation of the sequential workflow."""
    st.markdown(
        """
        <style>
        .pipeline-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 14px 20px;
            margin-bottom: 20px;
        }
        .pipeline-step {
            text-align: center;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid #ced4da;
            color: #212529;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .pipeline-arrow {
            font-size: 1.3rem;
            color: #6c757d;
            font-weight: bold;
        }
        .step-start-end {
            background: #e2e3e5;
            color: #383d41;
            border-color: #d6d8db;
        }
        .step-active {
            border-color: #0d6efd;
            color: #0d6efd;
        }
        </style>
        <div class="pipeline-container">
            <span class="pipeline-step step-start-end">START</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step step-active">🔬 Researcher Agent</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step step-active">✍️ Writer Agent</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step step-active">🧐 Reviewer Agent</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step step-start-end">END</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render sidebar configuration and connection status."""
    with st.sidebar:
        st.header("⚙️ Environment & LLM")
        model = get_model_name()
        base_url = get_base_url()
        st.markdown(f"**Model:** `{model}`")
        st.markdown(f"**Base URL:** `{base_url}`")

        is_available = check_ollama_availability()
        if is_available:
            st.success("🟢 Ollama Server Online")
        else:
            st.error("🔴 Ollama Server Offline")
            st.caption("Ensure Ollama is running (`ollama serve`).")

        st.markdown("---")
        st.markdown("### 🧩 LangGraph Shared State")
        st.caption(
            "All 3 agent nodes read and update a single centralized `WorkflowState` "
            "schema containing exactly 7 typed fields."
        )

        with st.expander("View Mermaid Topology"):
            st.code(get_mermaid_diagram(), language="mermaid")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="LangGraph State and Workflow Designer",
        page_icon="🤖",
        layout="wide",
    )

    render_sidebar()

    st.title("LangGraph State and Workflow Designer")
    st.markdown(
        "A multi-agent sequential pipeline built with **LangGraph**, **LangChain**, and local **Ollama** LLMs."
    )

    render_pipeline_visualization()

    # User input section
    topic = st.text_input(
        "Enter a Research Topic:",
        placeholder="e.g., Binary Search Algorithm, Quantum Computing, Photosynthesis...",
        help="Supply the topic to initiate the Researcher -> Writer -> Reviewer workflow.",
    )

    run_button = st.button("🚀 Run Workflow", type="primary", use_container_width=True)

    if run_button:
        cleaned_topic = topic.strip()
        if not cleaned_topic:
            st.warning("⚠️ Please enter a research topic before starting the workflow.")
            return

        if not check_ollama_availability():
            st.error(
                f"❌ Cannot connect to Ollama server at `{get_base_url()}`. "
                "Please verify that Ollama is installed, running (`ollama serve`), and the model "
                f"`{get_model_name()}` is downloaded."
            )
            return

        progress_bar = st.progress(0, text="Initializing workflow...")
        status_box = st.status("Executing LangGraph agents...", expanded=True)

        try:
            status_box.write("🔬 **Researcher Node**: Gathering factual research evidence...")
            progress_bar.progress(20, text="Step 1/3: Researcher agent working...")

            # Run execution
            final_state = execute_workflow(cleaned_topic, status_box)

            progress_bar.progress(100, text="Workflow completed successfully!")
            status_box.update(label="Workflow completed successfully!", state="complete", expanded=False)

            # Store in session state for persistent inspection
            st.session_state["workflow_state"] = final_state
            st.success(f"Execution complete for topic: **{cleaned_topic}**")

        except Exception as exc:
            progress_bar.empty()
            status_box.update(label="Workflow execution failed!", state="error", expanded=True)
            st.error(f"Execution error: {exc}")
            return

    # Display results dashboard if state exists
    if "workflow_state" in st.session_state:
        state: WorkflowState = st.session_state["workflow_state"]

        st.markdown("---")
        st.subheader("📊 Execution Results Dashboard")

        # Top summary KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Topic", state.get("topic", ""))
        with col2:
            st.metric("Status", state.get("status", ""))
        with col3:
            st.metric("Review Score", f"{state.get('score', 0)} / 10")
        with col4:
            st.metric("Revisions", state.get("revision_count", 0))

        st.markdown("### Agent Deliverables")

        # 1. Researcher Output
        with st.expander("🔬 [1] Researcher Output: Factual Evidence", expanded=True):
            st.markdown(state.get("evidence", "No evidence generated."))

        # 2. Writer Output
        with st.expander("✍️ [2] Writer Output: Structured Draft", expanded=True):
            st.markdown(state.get("draft", "No draft generated."))

        # 3. Reviewer Output
        with st.expander("🧐 [3] Reviewer Output: Evaluation & Score", expanded=True):
            st.markdown(f"**Review Score:** `{state.get('score', 0)} / 10`")
            st.markdown("---")
            st.markdown(state.get("feedback", "No feedback recorded."))

        # 4. Final Shared State Inspection
        with st.expander("📋 [4] Final Workflow State (Raw Shared Blackboard)", expanded=False):
            st.caption("Inspect the exact 7-field dictionary maintained in LangGraph memory:")
            st.json(state)


if __name__ == "__main__":
    main()
