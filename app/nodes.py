"""Agent node definitions for the LangGraph workflow.

Implements the three sequential agent nodes:
1. researcher_node: Gathers factual evidence for a topic.
2. writer_node: Generates a structured draft from the evidence.
3. reviewer_node: Reviews the draft, provides feedback, and scores it.
"""

import logging
import re
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.state import WorkflowState

logger = logging.getLogger(__name__)

# Prompt templates
RESEARCHER_SYSTEM_PROMPT = (
    "You are an expert research agent. Given a research topic, gather clear, factual, "
    "and concise evidence. Summarize the key concepts, background, and verified details. "
    "Strict rule: Do NOT fabricate or invent statistics, citations, or imaginary sources. "
    "State only well-established factual knowledge objectively."
)

WRITER_SYSTEM_PROMPT = (
    "You are a professional academic and technical writer. Using ONLY the provided research "
    "evidence, write a coherent, well-structured, and comprehensive answer for the topic.\n\n"
    "Your draft must follow this structure:\n"
    "1. Introduction: Define the topic, scope, and background.\n"
    "2. Explanation: Detailed, logically organized discussion based on the evidence.\n"
    "3. Conclusion: Summary of key insights and takeaways.\n\n"
    "Ensure clear transitions, academic tone, and accurate representation of the evidence."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a senior academic reviewer. Evaluate the provided draft based on the topic "
    "and supplied research evidence.\n\n"
    "Assess the draft on:\n"
    "- Relevance: Does it directly address the topic?\n"
    "- Clarity: Is it well-written, easy to follow, and grammatically sound?\n"
    "- Factual Consistency: Is it faithful to the supplied evidence without unverified claims?\n"
    "- Completeness: Does it cover key aspects adequately?\n"
    "- Organization: Are the introduction, explanation, and conclusion clearly distinguished?\n\n"
    "Provide comprehensive, constructive feedback explaining the strengths, weaknesses, "
    "and justification for your rating, then assign an integer score from 1 to 10 "
    "(1 = poor, 10 = excellent).\n\n"
    "You MUST format your response as follows:\n"
    "FEEDBACK:\n"
    "<Write your detailed review and evaluation comments here>\n\n"
    "SCORE: <integer from 1 to 10>"
)


def _extract_score_and_feedback(review_text: str) -> tuple[str, int]:
    """Parse the reviewer LLM response into feedback text and a validated score (1-10).

    Args:
        review_text: Raw LLM response string.

    Returns:
        tuple[str, int]: Parsed (feedback, validated_score).
    """
    default_score = 7
    score = None

    # Search for common score patterns in the response
    patterns = [
        r"(?:SCORE|Score|score)\s*[:=]\s*(\d+)",
        r"(?:SCORE|Score|score)\s+(?:is\s+|of\s+)?(\d+)(?:\s*/\s*10)?",
        r"(\d+)\s*/\s*10",
        r"(?:RATING|Rating|rating)\s*[:=]\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, review_text)
        if match:
            try:
                candidate = int(match.group(1))
                if 1 <= candidate <= 10:
                    score = candidate
                    break
                elif candidate > 10:
                    logger.warning("Parsed score %d > 10; clamping to 10", candidate)
                    score = 10
                    break
                else:
                    logger.warning("Parsed score %d < 1; clamping to 1", candidate)
                    score = 1
                    break
            except ValueError:
                continue

    if score is None:
        logger.warning(
            "Could not parse score from reviewer response; defaulting to %d",
            default_score,
        )
        score = default_score

    # Remove score line(s) to avoid discarding feedback that appears after the score
    cleaned_feedback = re.sub(
        r"(?im)^\s*(?:\*{0,2})(?:SCORE|Score|score|RATING|Rating)\s*[:=]?\s*\d+(?:\s*/\s*10)?(?:\*{0,2})\s*$",
        "",
        review_text,
    )
    # Also remove inline score declarations
    cleaned_feedback = re.sub(
        r"(?i)(?:\*{0,2})(?:SCORE|Score|score|RATING|Rating)\s*[:=]?\s*\d+(?:\s*/\s*10)?(?:\*{0,2})",
        "",
        cleaned_feedback,
    )

    # Strip leading FEEDBACK: / Feedback: header label if present
    cleaned_feedback = re.sub(r"(?i)^\s*(?:FEEDBACK|Feedback)\s*[:=]?\s*", "", cleaned_feedback).strip()

    feedback = cleaned_feedback if cleaned_feedback else review_text.strip()
    if not feedback:
        feedback = "Evaluation completed. The draft meets basic requirements."

    return feedback, score


def researcher_node(state: WorkflowState) -> Dict[str, Any]:
    """Researcher agent node.

    Reads the topic from shared state, gathers factual evidence using the LLM,
    and returns only the updated evidence and status fields.

    Args:
        state: Shared workflow state containing at least 'topic'.

    Returns:
        Dict[str, Any]: State update containing 'evidence' and 'status'.
    """
    topic = state.get("topic", "").strip()
    logger.info("Executing Researcher node for topic: '%s'", topic)

    if not topic:
        logger.error("Researcher node received an empty topic.")
        return {
            "evidence": "Error: No topic provided for research.",
            "status": "research_failed",
        }

    llm = get_llm()
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"Research topic: {topic}"),
    ]

    try:
        response = llm.invoke(messages)
        evidence = str(response.content).strip()
        logger.info("Researcher node completed successfully (%d chars generated)", len(evidence))
        return {
            "evidence": evidence,
            "status": "research_completed",
        }
    except Exception as exc:
        logger.exception("Researcher node failed during LLM invocation: %s", exc)
        raise


def writer_node(state: WorkflowState) -> Dict[str, Any]:
    """Writer agent node.

    Reads topic and evidence from shared state, generates a structured draft
    (introduction, explanation, conclusion), and returns only the draft and status fields.

    Args:
        state: Shared workflow state containing 'topic' and 'evidence'.

    Returns:
        Dict[str, Any]: State update containing 'draft' and 'status'.
    """
    topic = state.get("topic", "").strip()
    evidence = state.get("evidence", "").strip()
    logger.info("Executing Writer node for topic: '%s'", topic)

    llm = get_llm()
    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Topic: {topic}\n\nResearch Evidence:\n{evidence}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        draft = str(response.content).strip()
        logger.info("Writer node completed successfully (%d chars generated)", len(draft))
        return {
            "draft": draft,
            "status": "writing_completed",
        }
    except Exception as exc:
        logger.exception("Writer node failed during LLM invocation: %s", exc)
        raise


def reviewer_node(state: WorkflowState) -> Dict[str, Any]:
    """Reviewer agent node.

    Reads topic, evidence, and draft from shared state, evaluates the draft,
    assigns an integer score from 1 to 10, and returns only feedback, score,
    revision_count, and status fields.

    Args:
        state: Shared workflow state containing 'topic', 'evidence', and 'draft'.

    Returns:
        Dict[str, Any]: State update containing 'feedback', 'score', 'revision_count', and 'status'.
    """
    topic = state.get("topic", "").strip()
    evidence = state.get("evidence", "").strip()
    draft = state.get("draft", "").strip()
    logger.info("Executing Reviewer node for topic: '%s'", topic)

    llm = get_llm()
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Topic: {topic}\n\n"
                f"Research Evidence:\n{evidence}\n\n"
                f"Draft Answer:\n{draft}\n\n"
                "Please provide a thorough review evaluating relevance, clarity, consistency, completeness, "
                "and structure, explaining your justification for the score.\n\n"
                "Format:\n"
                "FEEDBACK:\n"
                "<Detailed evaluation comments and justification>\n\n"
                "SCORE: <integer from 1 to 10>"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        review_raw = str(response.content).strip()
        feedback, score = _extract_score_and_feedback(review_raw)
        logger.info("Reviewer node completed successfully (Score: %d/10)", score)
        return {
            "feedback": feedback,
            "score": score,
            "revision_count": 0,
            "status": "completed",
        }
    except Exception as exc:
        logger.exception("Reviewer node failed during LLM invocation: %s", exc)
        raise
