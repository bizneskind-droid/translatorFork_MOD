# gemini_translator/api/handlers/gumloop.py
"""
Gumloop chat/completions handler.

Особенности Gumloop, из-за которых нельзя переиспользовать openrouter.py:

1. Эндпоинт живёт на ws.gumloop.com, а НЕ на api.gumloop.com
   (на api.gumloop.com /chat/completions отдаёт 404).
2. Кроме Bearer-ключа обязателен user_id. Передаётся заголовком
   `x-auth-key` (в теле запроса не принимается — 400 "user_id is required").
3. Мультиплексор поверх Anthropic / OpenAI / Gemini / OpenRouter,
   формат — OpenAI chat/completions, поддерживает SSE.

Ключ и user_id берутся из ~/.hermes/.env (GUMLOOP_API_KEY / GUMLOOP_USER_ID),
если не заданы в настройках приложения.
"""

import asyncio
import json
import os
from pathlib import Path

import aiohttp

from ..base import BaseApiHandler
from ..errors import (
    ContentFilterError,
    LocationBlockedError,
    ModelNotFoundError,
    NetworkError,
    PartialGenerationError,
    RateLimitExceededError,
    TemporaryRateLimitError,
    ValidationFailedError,
)

_ENV_FILE = Path.home() / ".hermes" / ".env"


def _load_env_creds():
    """Читает GUMLOOP_API_KEY / GUMLOOP_USER_ID из окружения или ~/.hermes/.env."""
    key = os.environ.get("GUMLOOP_API_KEY")
    uid = os.environ.get("GUMLOOP_USER_ID")
    if key and uid:
        return key, uid
    try:
        with open(_ENV_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k == "GUMLOOP_API_KEY" and not key:
                    key = v
                elif k == "GUMLOOP_USER_ID" and not uid:
                    uid = v
    except OSError:
        pass
    return key, uid


class GumloopApiHandler(BaseApiHandler):
    """
    Хендлер для Gumloop (ws.gumloop.com/api/v1/chat/completions).

    ВАЖНО: у Gumloop сломан multi-turn tool-calling (400 на втором ходу),
    но для перевода это неважно — перевод это один запрос без инструментов.
    """

    def setup_client(self, client_override=None, proxy_settings=None):
        super().setup_client(client_override, proxy_settings)

        self.worker.model_id = self.worker.model_config.get("id", "claude-opus-5")
        self.base_url = self.worker.provider_config.get(
            "base_url", "https://ws.gumloop.com/api/v1/chat/completions"
        )

        env_key, env_uid = _load_env_creds()
        # Ключ из настроек приложения имеет приоритет, иначе — из .env.
        # DUMMY/пустышки игнорируем: для прокси-провайдеров (agentrouter) принято
        # передавать --api-key DUMMY, и такой ключ нельзя отправлять в Gumloop.
        gui_key = getattr(self.worker, "api_key", None)
        if gui_key and gui_key.strip().upper() in ("DUMMY", "NONE", "X", "-"):
            gui_key = None
        self._api_key = gui_key or env_key
        self._user_id = self.worker.provider_config.get("user_id") or env_uid

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
        if not self._api_key:
            raise RateLimitExceededError(
                "Не задан GUMLOOP_API_KEY (ни в настройках, ни в ~/.hermes/.env)."
            )
        if not self._user_id:
            raise RateLimitExceededError(
                "Не задан GUMLOOP_USER_ID — Gumloop требует user_id "
                "заголовком x-auth-key (см. ~/.hermes/.env)."
            )

        session = await self._get_or_create_session_internal()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            # user_id принимается ТОЛЬКО заголовком или query-параметром,
            # в теле запроса Gumloop его игнорирует (400 user_id is required)
            "x-auth-key": self._user_id,
        }

        messages = []
        if self.worker.prompt_builder.system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": self.worker.prompt_builder.system_instruction,
                }
            )
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.worker.model_id,
            "messages": messages,
            "stream": use_stream,
        }

        temperature = self._temperature_payload_value()
        if temperature is not None:
            payload["temperature"] = temperature

        # Gumloop использует max_completion_tokens (max_tokens объявлен deprecated)
        if max_output_tokens is not None:
            payload["max_completion_tokens"] = max_output_tokens
        else:
            configured_limit = self.worker.model_config.get("max_output_tokens")
            if configured_limit is not None:
                payload["max_completion_tokens"] = int(int(configured_limit) * 0.98)
            elif allow_incomplete:
                payload["max_completion_tokens"] = int(8192 * 0.98)

        safe_headers = dict(headers)
        safe_headers["Authorization"] = "Bearer ***"
        safe_headers["x-auth-key"] = "***"
        self._debug_record_request(
            {
                "method": "POST",
                "url": self.base_url,
                "headers": safe_headers,
                "payload": payload,
            },
            extra={"use_stream": use_stream, "allow_incomplete": allow_incomplete},
        )

        try:
            async with session.post(
                self.base_url, headers=headers, json=payload
            ) as response:

                # ── HTTP-ошибки ───────────────────────────────────────────
                if response.status != 200:
                    error_text = await response.text()
                    self._debug_record_response(
                        error_text,
                        status=f"http_{response.status}",
                        extra={"http_status": response.status, "mode": "error"},
                    )

                    if response.status == 400:
                        low = error_text.lower()
                        if "user_id" in low:
                            raise RateLimitExceededError(
                                "Gumloop не принял user_id — проверьте "
                                "GUMLOOP_USER_ID в ~/.hermes/.env."
                            )
                        if "content" in low and "filter" in low:
                            raise ContentFilterError(
                                f"Gumloop заблокировал запрос: {error_text[:200]}"
                            )
                        raise NetworkError(
                            f"Неверный запрос (400): {error_text[:200]}"
                        )

                    if response.status in (401, 403):
                        raise RateLimitExceededError(
                            f"Ошибка доступа ({response.status}) — неверный ключ "
                            f"или план ниже Pro: {error_text[:150]}"
                        )
                    if response.status == 402:
                        raise RateLimitExceededError(
                            f"Кончились кредиты Gumloop (402): {error_text[:150]}"
                        )
                    if response.status == 429:
                        raise TemporaryRateLimitError(
                            "Лимит запросов Gumloop (429).", delay_seconds=20
                        )
                    if response.status == 404:
                        raise ModelNotFoundError(
                            f"Модель {self.worker.model_id} не найдена (404). "
                            f"Список: gumloop.py models"
                        )
                    if response.status in (500, 502, 503, 504):
                        raise NetworkError(
                            f"Gumloop или upstream недоступен ({response.status}): "
                            f"{error_text[:150]}",
                            delay_seconds=5,
                        )
                    raise NetworkError(
                        f"Ошибка ({response.status}): {error_text[:150]}"
                    )

                # ── Ветка А: SSE-стриминг ─────────────────────────────────
                if use_stream:
                    collected_text = ""
                    finish_reason = None
                    stream_error = None
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
                            if not isinstance(chunk, dict):
                                continue

                            # Gumloop отдаёт ошибку ВНУТРИ 200-стрима отдельным
                            # чанком: {"error": {"code": 400, "message": ...}}
                            err = chunk.get("error")
                            if isinstance(err, dict) and err.get("message"):
                                stream_error = err.get("message")

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            if choice is None:
                                continue
                            delta = choice.get("delta") or {}
                            collected_text += delta.get("content") or ""
                            reason = choice.get("finish_reason")
                            if reason:
                                finish_reason = reason

                    except asyncio.CancelledError:
                        raise
                    except Exception as stream_exc:
                        if collected_text:
                            raise PartialGenerationError(
                                f"Обрыв стрима Gumloop: {stream_exc}",
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

                    if stream_error and not collected_text:
                        # finish_reason == "error" приходит с HTTP 200
                        raise NetworkError(f"Gumloop вернул ошибку: {stream_error}")

                    if not collected_text:
                        raise ValidationFailedError(
                            f"Gumloop вернул пустой ответ "
                            f"(finish_reason={finish_reason})."
                        )
                    if finish_reason in ("length", "max_tokens") and not allow_incomplete:
                        raise PartialGenerationError(
                            f"Превышен лимит токенов (finish_reason={finish_reason})",
                            partial_text=collected_text,
                            reason="LENGTH",
                        )

                    return collected_text

                # ── Ветка Б: обычный JSON ─────────────────────────────────
                raw_text = await response.text()
                try:
                    result = json.loads(raw_text)
                except json.JSONDecodeError:
                    raise NetworkError(
                        f"Gumloop вернул не-JSON (200): {raw_text[:300]}"
                    )
                self._debug_record_response(
                    result,
                    status="http_200",
                    extra={"mode": "full", "http_status": response.status},
                )

                err = result.get("error")
                if isinstance(err, dict) and err.get("message"):
                    raise NetworkError(f"Gumloop вернул ошибку: {err['message']}")

                first_choice = (result.get("choices") or [{}])[0]
                content = first_choice.get("message", {}).get("content", "")
                finish_reason = first_choice.get("finish_reason")

                if not content:
                    # У reasoning-моделей весь бюджет может уйти в reasoning_details
                    raise ValidationFailedError(
                        f"Gumloop вернул пустой content "
                        f"(finish_reason={finish_reason}). Поднимите max_output_tokens."
                    )
                if finish_reason in ("length", "max_tokens") and not allow_incomplete:
                    raise PartialGenerationError(
                        f"Превышен лимит токенов (finish_reason={finish_reason})",
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
                f"Нет соединения с ws.gumloop.com: {e}"
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
            raise NetworkError(f"Неожиданная ошибка Gumloop: {e}") from e
