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

    def __init__(self, message: str, debug_info: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.debug_info = debug_info


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
            debug_info: Optional[Dict[str, Any]] = None
            debug_logged = False
            try:
                payload, debug_info = self._invoke(
                    client=client,
                    error_cls=error_cls,
                    template=template,
                    variables=variables,
                    temperature=temperature if attempt == 1 else 0.0,
                )
                debug_logged = self._log_debug_payload(template.name, debug_info)
                template.validate(payload)
                return payload
            except Exception as exc:
                if isinstance(exc, LLMClientError):
                    failure = exc
                    debug_info = exc.debug_info
                else:
                    failure = LLMClientError(str(exc), debug_info)
                if not debug_logged:
                    debug_logged = self._log_debug_payload(template.name, debug_info)

                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt,
                    attempts,
                    template.name,
                    failure,
                )
                if attempt >= attempts:
                    raise LLMClientError(
                        f"Failed to generate structured response for {template.name}",
                        failure.debug_info,
                    ) from failure

        raise LLMClientError(f"Failed to invoke LLM for {template.name}")

    def _invoke(
        self,
        *,
        client,
        error_cls: Type[Exception],
        template: JsonPromptTemplate,
        variables: Dict[str, Any],
        temperature: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

        response_format = template.response_format if template.response_schema else None

        payload = {
            "model": self._settings.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "top_p": self._runtime.top_p,
                "repeat_penalty": self._runtime.repeat_penalty,
                "num_predict": self._runtime.max_tokens,
                "num_ctx": self._settings.context_window,
            },
        }

        if response_format:
            payload["format"] = response_format

        debug_info: Dict[str, Any] = {
            "method": "POST",
            "url": f"{self._settings.base_url}/api/chat",
            "headers": self._build_debug_headers(),
            "payload": payload,
        }

        try:
            result = self._chat_with_optional_format(
                client=client,
                error_cls=error_cls,
                messages=messages,
                response_format=response_format,
                debug_info=debug_info,
            )
        except error_cls as exc:
            debug_info["error"] = str(exc)
            raise LLMClientError(f"Ollama returned an error: {exc}", debug_info) from exc
        except Exception as exc:  # pragma: no cover - backend errors
            debug_info["error"] = str(exc)
            raise LLMClientError(f"Unexpected Ollama failure: {exc}", debug_info) from exc

        content = self._extract_content(result)
        if not content:
            debug_info["response_body"] = ""
            debug_info["error"] = "Empty response from LLM"
            raise LLMClientError("Empty response from LLM", debug_info)

        debug_info["response_body"] = content

        if template.response_schema and not content.lstrip().startswith("{"):
            debug_info["error"] = "LLM response was not JSON"
            raise LLMClientError("LLM response was not JSON", debug_info)

        try:
            parsed = self._parse_json(content)
        except Exception as exc:
            debug_info["error"] = f"Failed to parse JSON response: {exc}"
            raise LLMClientError("Failed to parse JSON response", debug_info) from exc
        if not isinstance(parsed, dict):
            debug_info["error"] = "Response was not a JSON object"
            raise LLMClientError("Response was not a JSON object", debug_info)

        return parsed, debug_info

    def _build_debug_headers(self) -> Dict[str, str]:
        """Return the headers used for debug logging."""

        return {"Content-Type": "application/json"}

    def _chat_with_optional_format(
        self,
        *,
        client,
        error_cls: Type[Exception],
        messages,
        response_format: Optional[Dict[str, Any]],
        debug_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        kwargs = {
            "model": self._settings.model,
            "messages": messages,
            "options": debug_info["payload"]["options"],
        }

        if not response_format:
            return client.chat(**kwargs)

        try:
            return client.chat(**kwargs, format=response_format)
        except TypeError as exc:
            debug_info.setdefault("warnings", []).append(
                f"Falling back to plain mode because client rejected format parameter: {exc}"
            )
        except error_cls as exc:
            if not self._format_rejected(exc):
                raise
            debug_info.setdefault("warnings", []).append(
                f"Falling back to plain mode because backend rejected format parameter: {exc}"
            )
        # Fallback path removes the format to match the actual request body in debug output.
        debug_info["payload"].pop("format", None)
        return client.chat(**kwargs)

    @staticmethod
    def _format_rejected(error: Exception) -> bool:
        message = str(error).lower()
        if "format" not in message:
            return False
        return any(token in message for token in ("unsupported", "unknown", "invalid", "unexpected"))

    def _log_debug_payload(
        self, template_name: str, debug_info: Optional[Dict[str, Any]]
    ) -> bool:
        """Emit redacted debug information when enabled."""

        if not self._settings.debug_payloads or not debug_info:
            return False

        redacted = self._redact_sensitive(debug_info)
        try:
            serialised = json.dumps(redacted, ensure_ascii=False, sort_keys=True)
        except TypeError:
            serialised = str(redacted)

        logger.debug("LLM debug payload for %s: %s", template_name, serialised)
        return True

    @classmethod
    def _redact_sensitive(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ("***" if cls._is_sensitive_key(key) else cls._redact_sensitive(val))
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [cls._redact_sensitive(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._redact_sensitive(item) for item in value)
        return value

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        lowered = key.lower()
        sensitive_tokens = (
            "authorization",
            "api_key",
            "apikey",
            "token",
            "secret",
            "password",
        )
        return any(token in lowered for token in sensitive_tokens)

    @staticmethod
    def _extract_content(result: Dict[str, Any]) -> str:
        message = result.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            # Some models put the response in "thinking" field
            content = (message.get("thinking") or "").strip()
        return content

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

        import requests

        messages = [
            {
                "role": "system",
                "content": "You are a status probe for NervNews. Reply concisely.",
            },
            {"role": "user", "content": prompt},
        ]
        options = {
            "temperature": 0.0,
            "top_p": self._runtime.top_p,
            "repeat_penalty": self._runtime.repeat_penalty,
            "num_predict": min(64, self._runtime.max_tokens),
            "num_ctx": self._settings.context_window,
        }

        url = f"{self._settings.base_url}/api/chat"
        body = {
            "model": self._settings.model,
            "messages": messages,
            "options": options,
            "stream": False,
        }

        debug_info = {
            "method": "POST",
            "url": url,
            "headers": {"Content-Type": "application/json"},
            "body": body,
        }

        try:
            response = requests.post(url, json=body, timeout=30)
            debug_info["status_code"] = response.status_code
            debug_info["response_headers"] = dict(response.headers)
            debug_info["response_body"] = response.text
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            debug_info["error"] = str(exc)
            raise LLMClientError(f"HTTP error during LLM connectivity test: {exc}", debug_info) from exc
        except Exception as exc:
            debug_info["error"] = str(exc)
            raise LLMClientError(f"Unexpected error during LLM connectivity test: {exc}", debug_info) from exc

        content = self._extract_content(result)
        if not content:
            debug_info["error"] = "Empty response from LLM"
            raise LLMClientError("LLM returned an empty response", debug_info)

        return content


__all__ = ["LLMClient", "LLMClientError"]
