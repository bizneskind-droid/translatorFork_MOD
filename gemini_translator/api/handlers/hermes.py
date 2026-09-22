# gemini_translator/api/handlers/hermes.py
"""
Hermes Agent handler (Nous Research).

Локальный шлюз Hermes отдаёт протокол Anthropic Messages:
    POST <base_url>            (полный URL до /v1/messages)
    заголовки: x-api-key, anthropic-version: 2023-06-01, content-type: application/json
    system-инструкция — ОТДЕЛЬНОЕ поле "system", не роль в messages.

SSE-формат Anthropic:
    event: content_block_delta
    data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
    event: message_delta
    data: {"type":"message_delta","delta":{"stop_reason":"end_turn|max_tokens|...}}

Non-stream:
    {"content":[{"type":"text","text":...}], "stop_reason":"end_turn"}

Ключ ОБЯЗАТЕЛЕН (в отличие от прокси-хендлеров вроде agentrouter) —
Hermes-шлюз требует валидный x-api-key.
"""

import aiohttp
import asyncio
import json
import os

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

# stop_reason'ы Anthropic, означающие обрыв по лимиту токенов
_LENGTH_STOP_REASONS = ("max_tokens", "length")

ANTHROPIC_VERSION = "2023-06-01"


class HermesApiHandler(BaseApiHandler):
    """
    Хендлер для Hermes Agent (Nous Research), протокол Anthropic Messages.

    Формат запросов: POST <base_url> (например http://127.0.0.1:8790/v1/messages).
    Поддерживает SSE-стриминг и обычный JSON-режим.
    """

    def setup_client(self, client_override=None, proxy_settings=None):
        super().setup_client(client_override, proxy_settings)

        if not client_override:
            return False

        self.worker.api_key = client_override.api_key
        self.worker.model_id = self.worker.model_config.get("id", "claude-opus-4-8")
        self.base_url = self.worker.provider_config.get(
            "base_url", "http://127.0.0.1:8790/v1/messages"
        )
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

        headers = {
            "x-api-key": self.worker.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.worker.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": use_stream,
        }

        # system-инструкция — отдельное поле Anthropic, а не роль в messages
        if self.worker.prompt_builder.system_instruction:
            payload["system"] = self.worker.prompt_builder.system_instruction

        # Extended thinking. Причина управления бюджетом: justwoker сидит за
        # Cloudflare с origin-timeout ~100 с. С thinking Opus 4.8 на крупном входе
        # (глава + промпт + глоссарий) молчит на TTFB и жжёт минуты reasoning →
        # Cloudflare рубит соединение HTTP 524. На переводе модель реально думает
        # ~87 знаков вне зависимости от бюджета, т.е. thinking не улучшает перевод,
        # только добавляет риск 524 (замер гл.100, решение юзера 2026-09-22).
        # ПОЭТОМУ ПО УМОЛЧАНИЮ thinking ВЫКЛЮЧЕН. Включить: HERMES_THINKING_BUDGET=<токены>.
        try:
            _budget = int(os.environ.get("HERMES_THINKING_BUDGET", "0"))
        except (TypeError, ValueError):
            _budget = 0
        if _budget > 0:
            payload["thinking"] = {"type": "enabled", "budget_tokens": _budget}
        else:
            payload["thinking"] = {"type": "disabled"}

        temperature = self._temperature_payload_value()
        if temperature is not None:
            payload["temperature"] = temperature

        # max_tokens в Anthropic обязателен — всегда проставляем значение
        if max_output_tokens is not None:
            payload["max_tokens"] = int(max_output_tokens)
        else:
            configured_limit = self.worker.model_config.get("max_output_tokens")
            if configured_limit is not None:
                payload["max_tokens"] = int(int(configured_limit) * 0.98)
            else:
                payload["max_tokens"] = 8192

        self._debug_record_request(
            {
                "method": "POST",
                "url": self.base_url,
                "headers": {k: v for k, v in headers.items() if k != "x-api-key"},
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

                    if response.status == 400:
                        # Anthropic-фильтр контента и «invalid_request» приходят как 400.
                        low = error_text.lower()
                        if "content" in low and (
                            "block" in low or "filter" in low or "policy" in low
                        ):
                            raise ContentFilterError(
                                "Hermes-шлюз заблокировал запрос (content filter). "
                                "Глава пропущена."
                            )
                        raise NetworkError(f"Неверный запрос (400): {error_text[:200]}")

                    if response.status in (401, 403):
                        raise RateLimitExceededError(
                            f"Ошибка доступа ({response.status}), проверьте x-api-key "
                            f"(…{str(self.worker.api_key)[-4:]}): {error_text[:150]}"
                        )
                    if response.status == 429:
                        raise TemporaryRateLimitError(
                            "Лимит запросов Hermes (429).", delay_seconds=20
                        )
                    if response.status == 404:
                        raise ModelNotFoundError(
                            f"Модель {self.worker.model_id} не найдена (404)."
                        )
                    if response.status == 504:
                        raise NetworkError(
                            "Upstream таймаут (504). Ретрай сработает.",
                            delay_seconds=5,
                        )
                    if response.status in (500, 502, 503):
                        raise NetworkError(
                            f"Шлюз или upstream недоступен ({response.status}): "
                            f"{error_text[:150]}"
                        )
                    raise NetworkError(
                        f"Ошибка ({response.status}): {error_text[:150]}"
                    )

                # ── Ветка А: SSE-стриминг (события Anthropic) ─────────────
                if use_stream:
                    collected_text = ""
                    stop_reason = None
                    raw_lines = [] if (self._has_debug_trace() or debug) else None

                    try:
                        async for raw_line in response.content:
                            line_str = raw_line.decode("utf-8").strip()
                            if raw_lines is not None:
                                raw_lines.append(line_str)
                            # Игнорируем строки event: и пустые; данные — в "data: "
                            if not line_str or not line_str.startswith("data: "):
                                continue
                            data_str = line_str[len("data: "):]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(chunk, dict):
                                continue

                            ev_type = chunk.get("type")

                            if ev_type == "content_block_delta":
                                delta = chunk.get("delta") or {}
                                # text_delta несёт обычный текст; input_json_delta и
                                # thinking_delta нам не нужны для перевода
                                if delta.get("type") == "text_delta":
                                    collected_text += delta.get("text") or ""

                            elif ev_type == "message_delta":
                                delta = chunk.get("delta") or {}
                                sr = delta.get("stop_reason")
                                if sr:
                                    stop_reason = sr

                            elif ev_type == "message_start":
                                msg = chunk.get("message") or {}
                                sr = msg.get("stop_reason")
                                if sr:
                                    stop_reason = sr

                            elif ev_type == "error":
                                err = chunk.get("error") or {}
                                err_type = str(err.get("type", "")).lower()
                                err_msg = err.get("message", data_str)
                                if "overloaded" in err_type or "rate" in err_type:
                                    raise TemporaryRateLimitError(
                                        f"Hermes SSE error: {err_msg}", delay_seconds=15
                                    )
                                raise NetworkError(f"Hermes SSE error: {err_msg}")

                    except asyncio.CancelledError:
                        raise
                    except (
                        TemporaryRateLimitError,
                        NetworkError,
                        ContentFilterError,
                    ):
                        raise
                    except Exception as stream_exc:
                        if collected_text:
                            raise PartialGenerationError(
                                f"Обрыв стрима Hermes: {stream_exc}",
                                partial_text=collected_text,
                                reason="NETWORK_ERROR",
                            )
                        raise NetworkError(
                            f"Ошибка SSE-стрима: {stream_exc}"
                        ) from stream_exc

                    if raw_lines is not None:
                        self._debug_record_response(
                            "\n".join(raw_lines),
                            status=stop_reason or "stream",
                            extra={"mode": "stream", "http_status": response.status},
                        )

                    if not collected_text:
                        raise ValidationFailedError(
                            f"Hermes вернул пустой ответ (stop_reason={stop_reason})."
                        )
                    if stop_reason in _LENGTH_STOP_REASONS and not allow_incomplete:
                        raise PartialGenerationError(
                            f"Превышен лимит токенов (stop_reason={stop_reason})",
                            partial_text=collected_text,
                            reason="LENGTH",
                        )

                    return collected_text

                # ── Ветка Б: обычный JSON ─────────────────────────────────
                else:
                    raw_text = await response.text()
                    try:
                        result = json.loads(raw_text)
                    except json.JSONDecodeError:
                        raise NetworkError(
                            f"Hermes вернул не-JSON (200): {raw_text[:300]}"
                        )
                    self._debug_record_response(
                        result,
                        status="http_200",
                        extra={"mode": "full", "http_status": response.status},
                    )

                    # content — массив блоков, собираем только текстовые
                    content = ""
                    for block in result.get("content") or []:
                        if isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text") or ""

                    if not content:
                        raise ValidationFailedError(
                            "Hermes вернул пустой ответ."
                        )
                    stop_reason = result.get("stop_reason")
                    if stop_reason in _LENGTH_STOP_REASONS and not allow_incomplete:
                        raise PartialGenerationError(
                            f"Превышен лимит токенов (stop_reason={stop_reason})",
                            partial_text=content,
                            reason="LENGTH",
                        )
                    return content

        except (
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
            asyncio.TimeoutError,
        ) as e:
            raise NetworkError(
                f"Нет соединения с Hermes-шлюзом ({self.base_url}). "
                f"Убедитесь, что шлюз запущен.\nОшибка: {e}"
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
                f"Неожиданная ошибка Hermes: {e}"
            ) from e
