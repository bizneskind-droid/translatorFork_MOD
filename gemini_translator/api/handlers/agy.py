# gemini_translator/api/handlers/agy.py
"""
Antigravity CLI (agy) handler.

Вызывает локально установленный `agy` (Antigravity CLI, Google) как подпроцесс
и возвращает текст перевода. Авторизация берётся из OAuth-сессии самого CLI
(`~/.gemini/antigravity-cli/antigravity-oauth-token`), поэтому API-ключ в
приложении не нужен — сюда подходит любой плейсхолдер.

Транспорт (проверено на agy 1.2.16):

* промпт уходит в **stdin обычным текстом** — `agy` без флага `-p` читает его
  оттуда. Это важно: `-p "<промпт>"` передаёт промпт аргументом, а Linux режет
  один argv-элемент на MAX_ARG_STRLEN = 128 КиБ, тогда как глава с глоссарием
  легко даёт 300+ КиБ. Обёртка вида `-p "$(cat file)"` этот предел НЕ обходит;
* ответ читается как NDJSON-поток (`--output-format stream-json`): события
  `init` / `step_update` / `result`. Итоговый текст берётся из
  `result.response`, а `step_update` с `text_delta` служит запасным источником
  и признаком того, что модель полезла в инструменты.

`agy` — это агент со своим набором инструментов (run_command, view_file,
write_to_file, ...), а не чистый chat-эндпоинт. В неинтерактивном режиме
`permission_mode` = request-review, то есть попытка вызова инструмента
останавливает ход. Поэтому промпт дополняется жёсткой директивой «инструментов
нет, верни перевод текстом» (_NO_TOOLS_DIRECTIVE), ход запускается в свежем
временном каталоге, а любой не-`agent_response` шаг считается аномалией и
валит задачу честной ошибкой вместо тихой порчи главы.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile

from ..base import BaseApiHandler
from ..errors import (
    ContentFilterError,
    NetworkError,
    RateLimitExceededError,
    ModelNotFoundError,
    ValidationFailedError,
)

# Шаблоны путей установки agy (официальный инсталлер кладёт бинарь сюда).
_FALLBACK_BINARIES = (
    "~/.local/bin/agy",
    "/usr/local/bin/agy",
)

_DEFAULT_TIMEOUT = 1800


class AgyApiHandler(BaseApiHandler):
    """
    Хендлер для Antigravity CLI (`agy`).

    HTTP/aiohttp не используется — общение с CLI через подпроцесс. Формат
    вывода — NDJSON, итоговый текст в `result.response`.
    """

    _logger = logging.getLogger("agy.handler")

    # Жёсткий запрет инструментов на уровне промпта.
    #
    # У agy нет флага --disallowed-tools, а его собственные инструменты
    # (run_command, write_to_file, view_file, grep_search, ...) в headless-режиме
    # требуют подтверждения и роняют ход. Директива нужна, чтобы модель сразу
    # отдавала перевод текстом, а не пыталась «сохранить» его в файл.
    _NO_TOOLS_DIRECTIVE = (
        "\n\n## TOOL POLICY (STRICT)\n"
        "You have NO tools available. Do NOT call any tool "
        "(no run_command, view_file, write_to_file, grep_search, or any "
        "function call).\n"
        "Return the full translation DIRECTLY as your text answer only. "
        "Do not attempt to save, write, or store the result anywhere — "
        "output it inline as plain text/HTML as instructed above.\n"
    )

    # Уровни, которые понимает `agy --effort`.
    _VALID_EFFORT = {"low", "medium", "high", "xhigh", "max"}

    def setup_client(self, client_override=None, proxy_settings=None):
        # HTTP-сессия не нужна, но базовый setup сохраняет proxy_settings и пр.
        super().setup_client(client_override, proxy_settings)

        provider_cfg = self.worker.provider_config or {}
        self.model_id = self.worker.model_config.get("id") or "default"
        self.timeout = int(
            provider_cfg.get("timeout_seconds")
            or provider_cfg.get("base_timeout")
            or _DEFAULT_TIMEOUT
        )
        self.binary = self._resolve_binary(provider_cfg.get("binary"))
        self.extra_args = [
            str(a) for a in (provider_cfg.get("extra_args") or []) if str(a).strip()
        ]

        if not self.binary:
            self.worker._post_event(
                "log_message",
                {
                    "message": (
                        "[AGY] ⚠️ Исполняемый файл `agy` не найден. Установите его: "
                        "curl -fsSL https://antigravity.google/cli/install.sh | bash "
                        "(либо укажите путь в поле 'binary' конфигурации провайдера)."
                    )
                },
            )

        return True

    @staticmethod
    def _resolve_binary(preferred=None):
        """Путь из конфига → PATH → известные места установки."""
        if preferred and str(preferred).strip():
            candidate = os.path.expanduser(str(preferred).strip())
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        found = shutil.which("agy")
        if found:
            return found
        for pattern in _FALLBACK_BINARIES:
            candidate = os.path.expanduser(pattern)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _resolve_effort(self):
        """
        Уровень мышления для `--effort`.

        Приоритет: значение из конфига провайдера → значение модели. Отдельного
        флага у agy для «выключить thinking» нет, а суффикс effort уже зашит в
        идентификатор модели (gemini-3.8-flash-high), поэтому по умолчанию флаг
        не передаётся вовсе и CLI использует свой дефолт.
        """
        provider_cfg = self.worker.provider_config or {}
        model_cfg = self.worker.model_config if isinstance(
            self.worker.model_config, dict
        ) else {}
        for raw in (provider_cfg.get("effort"), model_cfg.get("effort")):
            if raw:
                value = str(raw).strip().lower()
                if value in self._VALID_EFFORT:
                    return value
        return None

    def _build_command(self):
        cmd = [self.binary, "--output-format", "stream-json", "--disable-slash-commands"]
        if self.model_id and self.model_id != "default":
            cmd.extend(["--model", str(self.model_id)])
        effort = self._resolve_effort()
        if effort:
            cmd.extend(["--effort", effort])
        cmd.extend(self.extra_args)
        return cmd

    def _build_subprocess_env(self):
        env = dict(os.environ)
        extra_env = (self.worker.provider_config or {}).get("env")
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                if value is not None:
                    env[str(key)] = str(value)
        return env

    # ── Основной вызов ───────────────────────────────────────────────────
    async def call_api(
        self,
        prompt,
        log_prefix,
        allow_incomplete=False,
        use_stream=True,
        debug=False,
        max_output_tokens=None,
    ):
        if not self.binary:
            raise ModelNotFoundError(
                "Antigravity CLI (`agy`) не найден. Установите его и авторизуйтесь, "
                "либо укажите путь в поле 'binary' конфигурации провайдера."
            )

        # У agy нет управления max_tokens — сообщаем об этом один раз, а не молчим.
        if max_output_tokens is not None and debug:
            print(
                f"[AGY DEBUG] max_output_tokens={max_output_tokens} игнорируется: "
                "у agy нет соответствующего флага."
            )

        full_prompt = prompt + self._NO_TOOLS_DIRECTIVE
        cmd = self._build_command()

        self._debug_record_request(
            {
                "method": "SUBPROCESS",
                "binary": self.binary,
                "args": cmd[1:],
                "stdin": full_prompt,
                "stdin_chars": len(full_prompt),
            },
            extra={"use_stream": use_stream, "allow_incomplete": allow_incomplete},
        )

        self.worker._post_event(
            "log_message",
            {
                "message": (
                    f"[AGY] {log_prefix} Промпт: {len(full_prompt):,} символов, "
                    f"модель: {self.model_id}"
                )
            },
        )

        workdir = tempfile.mkdtemp(prefix="agy_work_")
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=self._build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise ModelNotFoundError(f"Не удалось запустить agy: {exc}") from exc
        except OSError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise NetworkError(f"Ошибка запуска agy: {exc}") from exc

        async def _feed_stdin(stream, data):
            try:
                stream.write(data)
                await stream.drain()
                stream.close()
            except Exception:
                pass

        async def _drain_stderr(stream):
            chunks = []
            try:
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
            except Exception:
                pass
            return b"".join(chunks)

        async def _drain_stdout(stream):
            lines = []
            got = False
            while True:
                line = await stream.readline()
                if not line:
                    break
                lines.append(line)
                if b'\"event\":\"result\"' in line or b'\"event\": \"result\"' in line:
                    got = True
                    break
            return b"".join(lines), got

        got_result = False
        stdin_task = None
        stderr_task = None
        try:
            stdin_task = asyncio.create_task(
                _feed_stdin(process.stdin, full_prompt.encode("utf-8"))
            )
            stderr_task = asyncio.create_task(_drain_stderr(process.stderr))
            stdout_data, got_result = await asyncio.wait_for(
                _drain_stdout(process.stdout),
                timeout=self.timeout,
            )

            # Если получен терминальный result или процесс завершился, аккуратно гасим его,
            # не дожидаясь вечного висения фоновых горутин agy CLI
            try:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
            except ProcessLookupError:
                pass

            # Добираем stderr
            try:
                stderr_data = await asyncio.wait_for(stderr_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                stderr_task.cancel()
                stderr_data = b""

        except asyncio.TimeoutError:
            if stdin_task and not stdin_task.done():
                stdin_task.cancel()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            raise NetworkError(
                f"agy не ответил за {self.timeout} секунд (задача будет перезапущена).",
                delay_seconds=30,
            )
        except asyncio.CancelledError:
            if stdin_task and not stdin_task.done():
                stdin_task.cancel()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            try:
                process.kill()
            except ProcessLookupError:
                pass
            raise
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        stdout_text = stdout_data.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_data.decode("utf-8", errors="replace").strip()

        if not got_result and process.returncode != 0:
            self._debug_record_response(
                stderr_text or stdout_text,
                status=f"exit_{process.returncode}",
                extra={"returncode": process.returncode, "mode": "error"},
            )
            self._raise_for_cli_error(process.returncode, stderr_text, stdout_text)

        return self._parse_stream_json(stdout_text, allow_incomplete)

    # ── Разбор NDJSON-потока событий ─────────────────────────────────────
    def _parse_stream_json(self, stdout_text, allow_incomplete):
        if not stdout_text:
            raise ValidationFailedError("agy вернул пустой вывод.")

        collected_text = ""
        final_result = None
        result_event = None
        init_event = None
        foreign_steps = []
        unparsed = 0

        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                unparsed += 1
                continue
            if not isinstance(event, dict):
                continue

            kind = event.get("event")

            if kind == "result":
                result_event = event.get("result") or {}
                final_result = result_event.get("response") or ""
            elif kind == "init":
                init_event = event.get("init") or {}
            elif kind == "step_update":
                step = event.get("step_update") or {}
                step_type = step.get("step_type")
                if step_type == "agent_response":
                    if step.get("text_delta"):
                        collected_text += step["text_delta"]
                elif step_type and step_type not in ("user_input", "error_message", "system_message"):
                    # Любой другой шаг = модель пошла работать агентом, а не
                    # переводить. Главу принимать нельзя.
                    foreign_steps.append((step_type, step.get("state")))

        self._debug_record_response(
            result_event if result_event is not None else stdout_text,
            status=(result_event or {}).get("status") or "stream",
            extra={
                "mode": "stream",
                "unparsed_lines": unparsed,
                "foreign_steps": foreign_steps[:10],
                "num_turns": (result_event or {}).get("num_turns"),
                "usage": (result_event or {}).get("usage"),
            },
        )

        if not result_event:
            raise NetworkError(
                "agy не вернул событие result. Хвост вывода: "
                f"{stdout_text[-300:]!r}"
            )

        text = collected_text if len(collected_text.strip()) >= len((final_result or "").strip()) else (final_result or "")

        status = str(result_event.get("status") or "").upper()
        if status and status != "SUCCESS":
            # Если agy вернул статус ошибки из-за отсечки токенов, но модель успешно продолжила
            # генерацию и отдала валидный HTML с абзацами — принимаем результат
            if not (re.search(r"<p\b", text, re.I) and not foreign_steps):
                self._raise_for_result_error(result_event)

        if foreign_steps:
            raise ValidationFailedError(
                "agy отработал как агент (шаги: "
                + ", ".join(t for t, _ in foreign_steps[:5])
                + "). Глава не принята."
            )
        if not text.strip():
            raise ContentFilterError(
                "agy вернул пустой ответ (сработал safety/content policy фильтр). "
                f"status={status or 'нет'}, usage={(result_event or {}).get('usage')}"
            )

        # Если ответ модели пришёл без тегов <p>, но содержит реальный текст перевода,
        # не отбрасываем попытку фатальной ошибкой валидации и не кромсаем эвристиками,
        # а передаём как есть на этап структурного выравнивания ИИ (realign).
        if not re.search(r"<p\b", text, re.I):
            if len(text.strip()) > 200:
                logger.warning(
                    f"agy вернул текст без HTML-абзацев ({len(text)} символов). "
                    "Передаём ответ на этап выравнивания структуры ИИ (realign)."
                )
            else:
                raise ValidationFailedError(
                    f"agy вернул невалидный ответ без текста: {text[:200]!r}"
                )

        return text

    # ── Классификация ошибок ─────────────────────────────────────────────
    def _raise_for_cli_error(self, returncode, stderr_text, stdout_text):
        blob = f"{stderr_text}\n{stdout_text}".lower()

        if "is not authenticated" in blob or "please run 'agy'" in blob:
            raise RateLimitExceededError(
                "Antigravity CLI не авторизован. Выполните вход: запустите `agy` "
                "в терминале и пройдите OAuth."
            )
        if any(k in blob for k in ("rate limit", "429", "too many requests", "quota")):
            raise RateLimitExceededError(
                "Лимит/квота подписки Antigravity исчерпаны. "
                f"Детали: {stderr_text[:200] or stdout_text[:200]}"
            )
        if any(k in blob for k in ("network issue", "stream reading error", "connection reset", "connection refused", "timeout", "timed out")):
            raise NetworkError(
                f"Сетевая ошибка agy: {stderr_text[:200] or stdout_text[:200]}",
                delay_seconds=15,
            )
        if any(k in blob for k in ("unauthorized", "401", "403", "invalid_token", "oauth error")):
            raise RateLimitExceededError(
                f"Ошибка авторизации Antigravity CLI: {stderr_text[:200] or stdout_text[:200]}"
            )
        if "not found" in blob and "model" in blob:
            raise ModelNotFoundError(
                f"Модель {self.model_id} не найдена в agy. Проверьте `agy models`."
            )
        if "user location is not supported" in blob:
            raise NetworkError(
                "Геоблок Antigravity: 'User location is not supported'. "
                "Нужен исходящий трафик из поддерживаемого региона.",
                delay_seconds=60,
            )
        if any(k in blob for k in ("content", "blocked", "policy", "safety", "filter")):
            raise ContentFilterError(
                "Antigravity заблокировал запрос (content policy). Глава пропущена."
            )
        raise NetworkError(
            f"agy завершился с ошибкой (код {returncode}): "
            f"{stderr_text[:300] or stdout_text[:300]}"
        )

    def _raise_for_result_error(self, result_event):
        blob = json.dumps(result_event, ensure_ascii=False).lower()
        if any(k in blob for k in ("quota", "rate limit", "limit exceeded", "429")):
            raise RateLimitExceededError(
                f"Лимит/квота Antigravity исчерпаны: {blob[:300]}"
            )
        if any(k in blob for k in ("authenticat", "unauthor", "login", "invalid_token")):
            raise RateLimitExceededError(
                f"Antigravity CLI не авторизован: {blob[:300]}"
            )
        if any(k in blob for k in ("content", "blocked", "policy", "safety", "filter")):
            raise ContentFilterError(
                f"Antigravity заблокировал запрос: {blob[:300]}"
            )
        raise NetworkError(f"agy вернул статус ошибки: {blob[:300]}")
