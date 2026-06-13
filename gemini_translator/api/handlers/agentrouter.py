# gemini_translator/api/handlers/agentrouter.py
"""
AgentRouter local proxy handler.

Proxies requests to a local server.js running on http://127.0.0.1:3000
which forwards to agentrouter.org with the stored API key and required
Kilo-Code identity headers.

OpenAI-compatible /v1/chat/completions endpoint, supports SSE streaming.
"""

import aiohttp
import asyncio
import json

from ..base import BaseApiHandler
from ..errors import (
    ContentFilterError,
    NetworkError,
    LocationBlockedError,
    RateLimitExceededError,
    ModelNotFoundError,
    ValidationFailedError,
    TemporaryRateLimitError,
    PartialGenerationError,
)


class AgentRouterApiHandler(BaseApiHandler):
    """
    Хендлер для локального прокси AgentRouter (server.js на порту 3000).

    API-ключ в приложении не нужен — прокси берёт его из своего config.json
    и сам подставляет нужные upstream-заголовки (User-Agent, HTTP-Referer и т.д.).

    Формат запросов: OpenAI /v1/chat/completions.
    Поддерживает SSE-стриминг и обычный JSON-режим.
    """

    def setup_client(self, client_override=None, proxy_settings=None):
        super().setup_client(client_override, proxy_settings)

        self.worker.model_id = self.worker.model_config.get(
            "id", "claude-sonnet-4-5"
        )
        self.base_url = self.worker.provider_config.get(
            "base_url", "http://127.0.0.1:3000/v1/chat/completions"
        )
        # Прокси локальный — прокидывать proxy_settings через него не нужно,
        # но session инициализируем как обычно.
        self._proactive_session_init()
        return True

    async def call_api(
        self,
        prompt,
        log_prefix,
        allow_incomplete=False,
        use_stream=True,
        debug=False,
        max_output_tokens=None,
    ):
        session = await self._get_or_create_session_internal()

        # Прокси не требует Bearer-токена от клиента
        headers = {
            "Content-Type": "application/json",
        }

        messages = (
            [
                {
                    "role": "system",
                    "content": self.worker.prompt_builder.system_instruction,
                }
            ]
            if self.worker.prompt_builder.system_instruction
            else []
        ) + [{"role": "user", "content": prompt}]

        payload = {
            "model": self.worker.model_id,
            "messages": messages,
            "stream": use_stream,
        }

        temperature = self._temperature_payload_value()
        if temperature is not None:
            payload["temperature"] = temperature

        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        elif allow_incomplete:
            payload["max_tokens"] = int(
                self.worker.model_config.get("max_output_tokens", 8192) * 0.98
            )

        self._debug_record_request(
            {
                "method": "POST",
                "url": self.base_url,
                "headers": headers,
                "payload": payload,
            },
            extra={"use_stream": use_stream, "allow_incomplete": allow_incomplete},
        )

        try:
            async with session.post(
                self.base_url, headers=headers, json=payload
            ) as response:

                # ── Обработка HTTP-ошибок ─────────────────────────────────
                if response.status != 200:
                    error_text = await response.text()
                    self._debug_record_response(
                        error_text,
                        status=f"http_{response.status}",
                        extra={"http_status": response.status, "mode": "error"},
                    )

                    if response.status in [401, 403]:
                        raise RateLimitExceededError(
                            f"Ошибка доступа ({response.status}): {error_text[:150]}"
                        )
                    if response.status == 429:
                        raise TemporaryRateLimitError(
                            "Лимит запросов AgentRouter (429).", delay_seconds=20
                        )
                    if response.status == 404:
                        raise ModelNotFoundError(
                            f"Модель {self.worker.model_id} не найдена (404)."
                        )
                    if response.status in [500, 502, 503]:
                        raise NetworkError(
                            f"Прокси или upstream недоступен ({response.status}): "
                            f"{error_text[:150]}"
                        )
                    raise NetworkError(
                        f"Ошибка ({response.status}): {error_text[:150]}"
                    )

                # ── Ветка А: SSE-стриминг ─────────────────────────────────
                if use_stream:
                    collected_text = ""
                    finish_reason = None
                    raw_lines = [] if (self._has_debug_trace() or debug) else None

                    try:
                        async for raw_line in response.content:
                            line_str = raw_line.decode("utf-8").strip()
                            if raw_lines is not None:
                                raw_lines.append(line_str)
                            if not line_str or not line_str.startswith("data: "):
                                continue
                            data_str = line_str[len("data: "):]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            collected_text += delta.get("content") or ""
                            reason = choices[0].get("finish_reason")
                            if reason:
                                finish_reason = reason

                    except asyncio.CancelledError:
                        raise
                    except Exception as stream_exc:
                        if collected_text:
                            raise PartialGenerationError(
                                f"Обрыв стрима AgentRouter: {stream_exc}",
                                partial_text=collected_text,
                                reason="NETWORK_ERROR",
                            )
                        raise NetworkError(
                            f"Ошибка SSE-стрима: {stream_exc}"
                        ) from stream_exc

                    if raw_lines is not None:
                        self._debug_record_response(
                            "\n".join(raw_lines),
                            status=finish_reason or "stream",
                            extra={"mode": "stream", "http_status": response.status},
                        )

                    if not collected_text:
                        raise ValidationFailedError(
                            "AgentRouter вернул пустой ответ."
                        )
                    if finish_reason == "length" and not allow_incomplete:
                        raise PartialGenerationError(
                            "Превышен лимит токенов (finish_reason=length)",
                            partial_text=collected_text,
                            reason="LENGTH",
                        )

                    return collected_text

                # ── Ветка Б: обычный JSON ─────────────────────────────────
                else:
                    result = await response.json()
                    self._debug_record_response(
                        result,
                        status="http_200",
                        extra={"mode": "full", "http_status": response.status},
                    )
                    content = (
                        (result.get("choices") or [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if not content:
                        raise ValidationFailedError(
                            "AgentRouter вернул пустой ответ."
                        )
                    return content

        except (
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
            asyncio.TimeoutError,
        ) as e:
            raise NetworkError(
                f"Нет соединения с AgentRouter proxy (127.0.0.1:3000). "
                f"Убедитесь, что server.js запущен: node ~/projects/server.js\n"
                f"Ошибка: {e}"
            ) from e
        except (
            RateLimitExceededError,
            TemporaryRateLimitError,
            ModelNotFoundError,
            ValidationFailedError,
            PartialGenerationError,
            NetworkError,
            ContentFilterError,
            LocationBlockedError,
        ):
            raise
        except Exception as e:
            raise NetworkError(
                f"Неожиданная ошибка AgentRouter: {e}"
            ) from e
