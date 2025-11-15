"""LLM client wrapper for invoking local Ollama compatible models."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

from src.config.settings import LLMSettings
from src.llm.prompts import JsonPromptTemplate

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the LLM backend fails to produce a valid response."""


@dataclass(frozen=True)
class _RuntimeConfig:
    max_tokens: int
    temperature: float
    top_p: float
    repeat_penalty: float


class LLMClient:
    """Simple wrapper that talks to a local Ollama runtime."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._runtime = _RuntimeConfig(
            max_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            repeat_penalty=settings.repeat_penalty,
        )
        self._client: Optional[Any] = None
        self._client_error: Optional[Type[Exception]] = None

    @property
    def settings(self) -> LLMSettings:
        return self._settings

    def update_settings(self, settings: LLMSettings) -> None:
        """Update runtime settings and reset the client if required."""

        if settings == self._settings:
            return

        logger.info(
            "Updating LLM client configuration (provider=%s -> %s, model=%s -> %s)",
            self._settings.provider,
            settings.provider,
            self._settings.model,
            settings.model,
        )
        self._settings = settings
        self._runtime = _RuntimeConfig(
            max_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            repeat_penalty=settings.repeat_penalty,
        )
        # Reset cached client so it reloads with the new parameters lazily.
        self._client = None
        self._client_error = None

    def _get_client(self) -> Tuple[Any, Type[Exception]]:  # pragma: no cover - heavy dependency
        if self._client is not None and self._client_error is not None:
            return self._client, self._client_error

        try:
            from ollama import Client, ResponseError
        except Exception as exc:  # pragma: no cover - import time failure
            raise LLMClientError("ollama Python client is not available") from exc

        logger.info(
            "Connecting to Ollama runtime at %s (provider=%s, model=%s)",
            self._settings.base_url,
            self._settings.provider,
            self._settings.model,
        )

        try:
            self._client = Client(host=self._settings.base_url)
            self._client_error = ResponseError
        except Exception as exc:  # pragma: no cover - backend errors
            raise LLMClientError(f"Failed to initialise Ollama client: {exc}") from exc

        return self._client, self._client_error

    def generate_structured(
        self,
        template: JsonPromptTemplate,
        variables: Dict[str, Any],
        *,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a JSON object that complies with ``template``."""

        client, error_cls = self._get_client()
        attempts = max_retries or self._settings.max_retries
        temperature = self._runtime.temperature

        for attempt in range(1, attempts + 1):
            try:
                payload = self._invoke(
                    client=client,
                    error_cls=error_cls,
                    template=template,
                    variables=variables,
                    temperature=temperature if attempt == 1 else 0.0,
                )
                template.validate(payload)
                return payload
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt,
                    attempts,
                    template.name,
                    exc,
                )
                if attempt >= attempts:
                    raise LLMClientError(
                        f"Failed to generate structured response for {template.name}"
                    ) from exc

        raise LLMClientError(f"Failed to invoke LLM for {template.name}")

    def _invoke(
        self,
        *,
        client,
        error_cls: Type[Exception],
        template: JsonPromptTemplate,
        variables: Dict[str, Any],
        temperature: float,
    ) -> Dict[str, Any]:
        """Invoke the chat completion endpoint and parse JSON."""

        user_prompt = template.render_user_prompt(**variables)
        messages = [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug(
            "Invoking LLM for %s with temperature %.2f and %d tokens",
            template.name,
            temperature,
            self._runtime.max_tokens,
        )

        try:
            result = client.chat(
                model=self._settings.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "top_p": self._runtime.top_p,
                    "repeat_penalty": self._runtime.repeat_penalty,
                    "num_predict": self._runtime.max_tokens,
                    "num_ctx": self._settings.context_window,
                },
            )
        except error_cls as exc:
            raise ValueError(f"Ollama returned an error: {exc}") from exc
        except Exception as exc:  # pragma: no cover - backend errors
            raise ValueError(f"Unexpected Ollama failure: {exc}") from exc

        content = self._extract_content(result)
        if not content:
            raise ValueError("Empty response from LLM")

        payload = self._parse_json(content)
        if not isinstance(payload, dict):
            raise ValueError("Response was not a JSON object")

        return payload

    @staticmethod
    def _extract_content(result: Dict[str, Any]) -> str:
        message = result.get("message") or {}
        return (message.get("content") or "").strip()

    @staticmethod
    def _parse_json(raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def ping(self, prompt: str = "Hello, are you there?") -> str:
        """Perform a lightweight connectivity check against the LLM backend."""

        client, error_cls = self._get_client()

        try:
            result = client.chat(
                model=self._settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a status probe for NervNews. Reply concisely.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.0,
                    "top_p": self._runtime.top_p,
                    "repeat_penalty": self._runtime.repeat_penalty,
                    "num_predict": min(64, self._runtime.max_tokens),
                    "num_ctx": self._settings.context_window,
                },
            )
        except error_cls as exc:  # pragma: no cover - backend dependency
            raise LLMClientError(f"Ollama returned an error during connectivity test: {exc}") from exc
        except Exception as exc:  # pragma: no cover - backend dependency
            raise LLMClientError(f"Unexpected error during LLM connectivity test: {exc}") from exc

        content = self._extract_content(result)
        if not content:
            raise LLMClientError("LLM returned an empty response")

        return content


__all__ = ["LLMClient", "LLMClientError"]
