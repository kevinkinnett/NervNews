"""Prompt template primitives for LLM powered enrichment."""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class JsonPromptTemplate:
    """Simple representation for JSON-constrained prompts."""

    name: str
    system_prompt: str
    user_template: str
    response_schema: Dict[str, Any]

    def render_user_prompt(self, **variables: Any) -> str:
        """Render the user template with the provided ``variables``."""

        return dedent(self.user_template).format(**{k: v or "" for k, v in variables.items()})

    @property
    def required_fields(self) -> Iterable[str]:
        return tuple(self.response_schema.get("required", ()))

    @property
    def response_format(self) -> Dict[str, Any]:
        """Return the schema formatted for Ollama JSON mode/function calling."""

        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.response_schema,
            },
        }

    def validate(self, payload: Dict[str, Any]) -> None:
        missing = [field for field in self.required_fields if field not in payload]
        if missing:
            raise ValueError(f"Missing required fields in response: {', '.join(missing)}")

        empty = [
            field
            for field in self.required_fields
            if isinstance(payload.get(field), str) and not payload[field].strip()
        ]
        if empty:
            raise ValueError(f"Required fields contained empty strings: {', '.join(empty)}")


__all__ = ["JsonPromptTemplate"]
