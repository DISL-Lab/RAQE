"""Prompt shared by RAQE distillation, GRPO, and inference."""

from typing import Dict, List

SYSTEM_PROMPT = (
    "You are a helpful assistant for query expansion. "
    "Return only the requested pseudo-passage; do not include commentary."
)


def build_messages_for_task(task: str, query: str, pseudo_ref: str = "", **_: object) -> List[Dict[str, str]]:
    """Build the camera-ready RAQE pseudo-passage supervision example."""
    if task != "pseudo":
        raise ValueError("RAQE trains only the pseudo-passage generation task.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Instruction: Write a single cohesive paragraph (at least three sentences) "
                "that directly addresses the query. Include concrete details when possible, "
                "avoid meta commentary, and return only the passage text.\n"
                f"Query: {query}"
            ),
        },
        {"role": "assistant", "content": pseudo_ref},
    ]


def build_user_only_messages(task: str, query: str, **_: object) -> List[Dict[str, str]]:
    """Build the inference prompt used for RAQE rollouts and evaluation."""
    return build_messages_for_task(task=task, query=query, pseudo_ref="")[:-1]
