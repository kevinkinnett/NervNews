"""LLM client wrapper for invoking local llama.cpp compatible models."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
    n_threads: Optional[int]


class LLMClient:
    """Simple wrapper that talks to a local ``llama.cpp`` compatible model."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._runtime = _RuntimeConfig(
            max_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            repeat_penalty=settings.repeat_penalty,
            n_threads=settings.threads,
        )
        self._model: Any = None

    @property
    def settings(self) -> LLMSettings:
        return self._settings

    def update_settings(self, settings: LLMSettings) -> None:
        """Update runtime settings and reset the loaded model if required."""

        if settings == self._settings:
            return

        logger.info(
            "Updating LLM client configuration (provider=%s -> %s, model=%s -> %s)",
            self._settings.provider,
            settings.provider,
            self._settings.model_path,
            settings.model_path,
        )
        self._settings = settings
        self._runtime = _RuntimeConfig(
            max_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            repeat_penalty=settings.repeat_penalty,
            n_threads=settings.threads,
        )
        # Reset cached model so it reloads with the new parameters lazily.
        self._model = None

    def _load_model(self) -> Any:  # pragma: no cover - heavy dependency
        if self._model is not None:
            return self._model

        model_path = Path(self._settings.model_path)
        if not model_path.exists():
            raise LLMClientError(f"Model path {model_path} does not exist")

        try:
            from llama_cpp import Llama
        except Exception as exc:  # pragma: no cover - import time failure
            raise LLMClientError("llama-cpp-python is not available") from exc

        logger.info(
            "Loading LLM model from %s (provider=%s, quantization=%s)",
            model_path,
            self._settings.provider,
            self._settings.quantization or "auto",
        )
        try:
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self._settings.context_window,
                n_threads=self._runtime.n_threads,
                logits_all=False,
            )
        except Exception as exc:  # pragma: no cover - backend errors
            raise LLMClientError(f"Failed to initialise model: {exc}") from exc

        return self._model

    def generate_structured(
        self,
        template: JsonPromptTemplate,
        variables: Dict[str, Any],
        *,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a JSON object that complies with ``template``."""

        model = self._load_model()
        attempts = max_retries or self._settings.max_retries
        temperature = self._runtime.temperature

        for attempt in range(1, attempts + 1):
            try:
                payload = self._invoke(
                    model=model,
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
        model,
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

        result = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=self._runtime.max_tokens,
            top_p=self._runtime.top_p,
            repeat_penalty=self._runtime.repeat_penalty,
        )
        content = self._extract_content(result)
        if not content:
            raise ValueError("Empty response from LLM")

        payload = self._parse_json(content)
        if not isinstance(payload, dict):
            raise ValueError("Response was not a JSON object")

        return payload

    @staticmethod
    def _extract_content(result: Dict[str, Any]) -> str:
        choices = result.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
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


__all__ = ["LLMClient", "LLMClientError"]
