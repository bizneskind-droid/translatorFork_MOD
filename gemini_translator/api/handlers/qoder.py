# gemini_translator/api/handlers/qoder.py
"""
Qoder CLI handler.

Вызывает локально установленный `qodercli` как подпроцесс в неинтерактивном
режиме (-p / --print). Авторизация берётся из ~/.qoder/.auth/ (PAT-логин),
поэтому API-ключ в приложении не нужен.

Промпт передаётся через stdin (обходит лимит длины argv для больших глав),
системная инструкция — через --system-prompt. Ответ читается как единый JSON
(--output-format json) либо как поток событий (--output-format stream-json).

Инструменты (Bash/Edit/Write/Read/...) принудительно отключены, а режим
разрешений выставлен в dont_ask, чтобы CLI не зависал в ожидании ввода и не
трогал файловую систему во время перевода.
"""

import asyncio
import json
import logging
import os
import shutil

from ..base import BaseApiHandler
from ..errors import (
    ContentFilterError,
    NetworkError,
    RateLimitExceededError,
    ModelNotFoundError,
    ValidationFailedError,
    TemporaryRateLimitError,
    PartialGenerationError,
)


class QoderApiHandler(BaseApiHandler):
    """
    Хендлер для локального Qoder CLI (qodercli).

    Не использует HTTP/aiohttp — общается с CLI через подпроцесс.
    Формат вывода: Anthropic-подобный JSON, где итоговый текст лежит в поле
    "result", а состояние завершения — в "stop_reason".
    """

    _logger = logging.getLogger("qoder.handler")

    # Инструменты, которые CLI не должен использовать во время перевода.
    _DISALLOWED_TOOLS = [
        "Bash", "Edit", "Write", "Read", "Glob", "Grep",
        "WebFetch", "WebSearch", "NotebookEdit", "Agent",
        "TodoWrite", "Workflow", "Skill", "ImageGen", "ImageSearch",
    ]

    # Жёсткий запрет инструментов на уровне промпта.
    #
    # Флаг --disallowed-tools блокирует ИСПОЛНЕНИЕ инструментов, но модель с
    # включённым thinking (high/max) всё равно ПЫТАЕТСЯ вызвать Write, чтобы
    # "сохранить" перевод в файл (stop_reason=tool_use). Вызов отбивается
    # ошибкой, модель уходит на второй ход разбираться с ней — и там нередко
    # ловит внутренний idle-timeout qodercli (120с молчания в потоке) → обрыв,
    # причём кредиты за сгенерированный ответ уже списаны.
    #
    # Поэтому дополнительно запрещаем инструменты прямо в системной инструкции,
    # чтобы модель сразу отдавала перевод текстом, а не лезла в Write.
    _NO_TOOLS_DIRECTIVE = (
        "\n\n## TOOL POLICY (STRICT)\n"
        "You have NO tools available. Do NOT call any tool "
        "(no Write, Edit, Bash, Read, or any function call).\n"
        "Return the full translation DIRECTLY as your text answer only. "
        "Do not attempt to save, write, or store the result anywhere — "
        "output it inline as plain text/HTML as instructed above."
    )

    def setup_client(self, client_override=None, proxy_settings=None):
        # HTTP-сессия не нужна, но базовый setup сохраняет proxy_settings и пр.
        super().setup_client(client_override, proxy_settings)

        self.worker.model_id = self.worker.model_config.get("id", "Auto")

        provider_cfg = self.worker.provider_config
        self.binary = (
            provider_cfg.get("binary")
            or shutil.which("qodercli")
            or os.path.expanduser("~/.local/bin/qodercli")
        )
        return True

    # Уровни thinking, которые понимает qodercli.
    # Полный набор; конкретный список допустимых уровней задаётся per-model
    # в конфиге (thinkingLevel). qodercli молча сбрасывает неподдерживаемый
    # моделью уровень в "none" (clearUnsupportedReasoningEffort).
    _VALID_EFFORT = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

    def _resolve_reasoning_effort(self):
        """
        Определяет уровень --reasoning-effort.

        Приоритет:
          1. UI: галка Thinking выключена → "none" (thinking отключён явно).
          2. UI: галка включена + выбран уровень → этот уровень.
          3. Конфиг модели: reasoning_effort (запасной вариант).
          4. Иначе None (не передаём флаг, дефолт модели).
        """
        model_config = self.worker.model_config if isinstance(
            self.worker.model_config, dict
        ) else {}

        # Модель вообще поддерживает выбор уровней?
        supports_levels = model_config.get("thinkingLevel") is not None

        if supports_levels:
            thinking_enabled = bool(
                getattr(self.worker, "thinking_enabled", False)
            )
            if not thinking_enabled:
                return "none"
            level = getattr(self.worker, "thinking_level", None)
            if level:
                level = str(level).strip().lower()
                if level in self._VALID_EFFORT:
                    return level

        # Запасной вариант — значение из конфига модели.
        fallback = model_config.get("reasoning_effort")
        if fallback:
            fallback = str(fallback).strip().lower()
            if fallback in self._VALID_EFFORT:
                return fallback

        return None

    def _build_command(self, max_output_tokens):
        cmd = [
            self.binary,
            "-p",
            "--output-format", self.worker.provider_config.get("output_format", "json"),
            "--permission-mode", "dont_ask",
            "-m", self.worker.model_id,
        ]

        # Отключаем инструменты по одному (пустой --tools "" подвешивает CLI).
        cmd.append("--disallowed-tools")
        cmd.extend(self._DISALLOWED_TOOLS)

        system_instruction = getattr(
            self.worker.prompt_builder, "system_instruction", None
        )
        # Всегда добавляем запрет инструментов, даже если системной инструкции
        # нет (тогда директива идёт как самостоятельный system-prompt).
        system_instruction = (system_instruction or "") + self._NO_TOOLS_DIRECTIVE
        cmd.extend(["--system-prompt", system_instruction])

        # Управление "мышлением" модели (--reasoning-effort).
        # Приоритет: UI-настройка (галка Thinking + выпадающий уровень) →
        # затем reasoning_effort из конфига модели как запасной вариант.
        # Допустимые уровни qodercli: none / minimal / low / medium / high.
        reasoning = self._resolve_reasoning_effort()
        if reasoning:
            cmd.extend(["--reasoning-effort", reasoning])

        if max_output_tokens is not None:
            cmd.extend(["--max-output-tokens", str(int(max_output_tokens))])
        else:
            configured_limit = self.worker.model_config.get("max_output_tokens")
            if configured_limit is not None:
                cmd.extend([
                    "--max-output-tokens",
                    str(int(int(configured_limit) * 0.98)),
                ])

        context_window = self.worker.model_config.get("context_length")
        if context_window:
            cmd.extend(["--context-window", str(int(context_window))])

        return cmd

    def _build_subprocess_env(self):
        """
        Окружение для подпроцесса qodercli.

        Ключевой момент: QODER_HTTPDNS=0 отключает встроенный HTTPDNS-резолвер
        qodercli. По умолчанию CLI резолвит свои домены в захардкоженные IP
        Alibaba Cloud (8.211.x.x / 47.77.x.x), которые из некоторых сетей (РФ)
        то доступны, то нет — в обход системного DNS и VPN. Это давало
        интермиттентные зависания на 90-400с и обрывы "Unable to connect".

        С QODER_HTTPDNS=0 CLI переходит в networkMode=direct и использует
        системный DNS → трафик идёт через VPN/прокси штатно.

        Значение можно переопределить в конфиге провайдера ключом
        "httpdns" (true → не трогаем поведение CLI).
        """
        env = dict(os.environ)

        httpdns_enabled = self.worker.provider_config.get("httpdns", False)
        if not httpdns_enabled:
            env["QODER_HTTPDNS"] = "0"

        # Дополнительные env-переменные из конфига (например прокси).
        extra_env = self.worker.provider_config.get("env")
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                if value is not None:
                    env[str(key)] = str(value)

        return env

    async def call_api(
        self,
        prompt,
        log_prefix,
        allow_incomplete=False,
        use_stream=True,
        debug=False,
        max_output_tokens=None,
    ):
        if not self.binary or not os.path.exists(self.binary):
            raise NetworkError(
                "qodercli не найден. Установите Qoder CLI и авторизуйтесь "
                "через PAT (qodercli login), либо укажите путь в поле "
                "'binary' конфигурации провайдера."
            )

        cmd = self._build_command(max_output_tokens)

        system_instruction = getattr(
            self.worker.prompt_builder, "system_instruction", None
        )
        self._debug_record_request(
            {
                "method": "SUBPROCESS",
                "binary": self.binary,
                "args": cmd[1:],
                "system_prompt": system_instruction or "",
                "stdin": prompt,
                "stdin_chars": len(prompt),
            },
            extra={"use_stream": use_stream, "allow_incomplete": allow_incomplete},
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_subprocess_env(),
            )
        except FileNotFoundError as e:
            raise NetworkError(f"Не удалось запустить qodercli: {e}") from e

        try:
            stdout_data, stderr_data = await process.communicate(
                input=prompt.encode("utf-8")
            )
        except asyncio.CancelledError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            raise

        stdout_text = stdout_data.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_data.decode("utf-8", errors="replace").strip()

        if process.returncode != 0:
            self._debug_record_response(
                stderr_text or stdout_text,
                status=f"exit_{process.returncode}",
                extra={"returncode": process.returncode, "mode": "error"},
            )
            self._raise_for_cli_error(process.returncode, stderr_text, stdout_text)

        if use_stream and self.worker.provider_config.get(
            "output_format", "json"
        ) == "stream-json":
            return self._parse_stream_json(
                stdout_text, allow_incomplete
            )
        return self._parse_result_json(stdout_text, allow_incomplete)

    # ── Парсинг единого JSON (--output-format json) ──────────────────────
    def _parse_result_json(self, stdout_text, allow_incomplete):
        if not stdout_text:
            raise ValidationFailedError("qodercli вернул пустой вывод.")

        try:
            result = json.loads(stdout_text)
        except json.JSONDecodeError:
            # Иногда перед JSON может быть служебный вывод — берём последнюю
            # непустую строку, похожую на JSON-объект.
            candidate = None
            for line in reversed(stdout_text.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    candidate = line
                    break
            if candidate is None:
                raise NetworkError(
                    f"qodercli вернул не-JSON: {stdout_text[:300]}"
                )
            result = json.loads(candidate)

        self._debug_record_response(
            result,
            status=result.get("stop_reason") or result.get("subtype") or "ok",
            extra={"mode": "full", "is_error": result.get("is_error")},
        )

        if result.get("is_error"):
            self._raise_for_result_error(result)

        text = result.get("result") or ""
        if not text:
            raise ValidationFailedError("qodercli вернул пустой ответ (result).")

        stop_reason = result.get("stop_reason")
        if stop_reason in ("max_tokens", "length") and not allow_incomplete:
            raise PartialGenerationError(
                f"Превышен лимит токенов (stop_reason={stop_reason})",
                partial_text=text,
                reason="LENGTH",
            )

        return text

    # ── Парсинг потока событий (--output-format stream-json) ─────────────
    def _parse_stream_json(self, stdout_text, allow_incomplete):
        if not stdout_text:
            raise ValidationFailedError("qodercli вернул пустой поток.")

        collected_text = ""
        stop_reason = None
        final_result = None
        is_error = False

        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            etype = event.get("type")

            if etype == "stream_event":
                inner = event.get("event") or {}
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta") or {}
                    # Берём только видимый текст, пропускаем thinking_delta.
                    if delta.get("type") == "text_delta":
                        collected_text += delta.get("text") or ""
                elif inner.get("type") == "message_delta":
                    reason = (inner.get("delta") or {}).get("stop_reason")
                    if reason:
                        stop_reason = reason

            elif etype == "result":
                final_result = event.get("result") or ""
                is_error = bool(event.get("is_error"))
                if event.get("stop_reason"):
                    stop_reason = event.get("stop_reason")

        self._debug_record_response(
            stdout_text,
            status=stop_reason or "stream",
            extra={"mode": "stream", "is_error": is_error},
        )

        if is_error:
            self._raise_for_result_error({"result": final_result})

        text = final_result if final_result else collected_text
        if not text:
            raise ValidationFailedError("qodercli вернул пустой ответ (поток).")

        if stop_reason in ("max_tokens", "length") and not allow_incomplete:
            raise PartialGenerationError(
                f"Превышен лимит токенов (stop_reason={stop_reason})",
                partial_text=text,
                reason="LENGTH",
            )

        return text

    # ── Классификация ошибок ─────────────────────────────────────────────
    def _raise_for_cli_error(self, returncode, stderr_text, stdout_text):
        blob = f"{stderr_text}\n{stdout_text}".lower()

        if any(k in blob for k in ("rate limit", "429", "too many requests")):
            raise TemporaryRateLimitError(
                "Лимит запросов Qoder.", delay_seconds=20
            )
        if any(k in blob for k in ("unauthorized", "401", "403", "not logged in",
                                   "authentication", "login")):
            raise RateLimitExceededError(
                f"Ошибка авторизации Qoder CLI. Выполните вход (qodercli login). "
                f"Детали: {stderr_text[:150]}"
            )
        if any(k in blob for k in ("model", "not found")) and "not found" in blob:
            raise ModelNotFoundError(
                f"Модель {self.worker.model_id} не найдена в Qoder CLI."
            )
        if any(k in blob for k in ("content", "blocked", "policy", "safety")):
            raise ContentFilterError(
                "Qoder заблокировал запрос (content policy). Глава пропущена."
            )
        raise NetworkError(
            f"qodercli завершился с ошибкой (код {returncode}): "
            f"{stderr_text[:200] or stdout_text[:200]}"
        )

    def _raise_for_result_error(self, result):
        text = (result.get("result") or "").lower()
        if any(k in text for k in ("rate limit", "429")):
            raise TemporaryRateLimitError("Лимит запросов Qoder.", delay_seconds=20)
        if any(k in text for k in ("content", "blocked", "policy", "safety")):
            raise ContentFilterError(
                "Qoder заблокировал запрос (content policy). Глава пропущена."
            )
        raise NetworkError(
            f"qodercli вернул ошибку: {result.get('result', '')[:200]}"
        )
