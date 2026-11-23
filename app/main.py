"""Main CLI entrypoint for the LangGraph State and Workflow Designer application.

Provides an interactive and command-line interface to execute the sequential
workflow: Researcher -> Writer -> Reviewer.
"""

import argparse
import logging
import sys
from typing import Optional

from app import setup_logging
from app.graph import workflow_graph
from app.llm import check_ollama_availability, get_base_url, get_model_name
from app.state import WorkflowState

logger = logging.getLogger("app.main")


def print_banner() -> None:
    """Print the application banner and pipeline description."""
    banner = """
================================================================================
           LANGGRAPH STATE AND WORKFLOW DESIGNER
           Execution Flow: START -> Researcher -> Writer -> Reviewer -> END
================================================================================
"""
    print(banner)


def print_final_state(state: WorkflowState) -> None:
    """Format and display the final workflow state fields cleanly.

    Args:
        state: Final WorkflowState dictionary containing all 7 fields.
    """
    separator = "=" * 80
    sub_sep = "-" * 80

    print(f"\n{separator}")
    print("                     WORKFLOW EXECUTION RESULTS")
    print(f"{separator}")
    print(f"Execution Order : Researcher -> Writer -> Reviewer")
    print(f"Topic           : {state.get('topic')}")
    print(f"Workflow Status : {state.get('status')}")
    print(f"Review Score    : {state.get('score')} / 10")
    print(f"Revision Count  : {state.get('revision_count')}")
    print(f"{sub_sep}")

    print("\n--- [1] RESEARCH EVIDENCE ---")
    print(state.get("evidence", "No evidence recorded."))

    print("\n--- [2] GENERATED DRAFT ---")
    print(state.get("draft", "No draft generated."))

    print("\n--- [3] REVIEWER FEEDBACK ---")
    print(f"Score: {state.get('score')} / 10\n")
    print(state.get("feedback", "No feedback recorded."))

    print(f"\n{separator}\n")


def run_workflow(topic: str) -> Optional[WorkflowState]:
    """Execute the LangGraph workflow for a given topic.

    Args:
        topic: User-supplied research topic.

    Returns:
        Optional[WorkflowState]: Final state dictionary or None on failure.
    """
    cleaned_topic = topic.strip()
    if not cleaned_topic:
        print("\n[Error] The research topic cannot be empty. Please enter a valid topic.")
        logger.error("Attempted to run workflow with empty topic.")
        return None

    # Verify Ollama server connectivity
    if not check_ollama_availability():
        base_url = get_base_url()
        model_name = get_model_name()
        error_msg = f"""
[ERROR] Unable to connect to Ollama at {base_url}.

To resolve this issue:
1. Make sure Ollama is installed and running:
   - On Windows, start Ollama from the Start Menu or run 'ollama serve' in PowerShell.
2. Verify that the configured model is installed:
   - Run: 'ollama pull {model_name}'
3. Verify running models with:
   - Run: 'ollama list'
"""
        print(error_msg, file=sys.stderr)
        logger.error("Ollama availability check failed for %s", base_url)
        return None

    # Initialize complete 7-field state
    initial_state: WorkflowState = {
        "topic": cleaned_topic,
        "evidence": "",
        "draft": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "status": "initialized",
    }

    logger.info("Starting workflow execution for topic: '%s'", cleaned_topic)
    print(f"\nStarting workflow for topic: '{cleaned_topic}'")
    print("Execution pipeline: Researcher -> Writer -> Reviewer")

    try:
        final_state: WorkflowState = workflow_graph.invoke(initial_state)
        logger.info(
            "Workflow finished successfully with status '%s' (Score: %d/10)",
            final_state.get("status"),
            final_state.get("score", 0),
        )
        return final_state
    except Exception as exc:
        logger.exception("Workflow execution failed: %s", exc)
        print(f"\n[ERROR] An error occurred during workflow execution:\n{exc}", file=sys.stderr)
        return None


def main() -> int:
    """CLI application entrypoint."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="LangGraph State and Workflow Designer: Researcher -> Writer -> Reviewer",
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        help="Research topic (if omitted, an interactive prompt will be displayed)",
    )
    args = parser.parse_args()

    print_banner()

    topic = args.topic
    if not topic:
        try:
            topic = input("Enter a research topic: ")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return 1

    final_state = run_workflow(topic)
    if final_state is None:
        return 1

    print_final_state(final_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
