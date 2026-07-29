# gemini_translator/api/handlers/duo_cli.py
"""
GitLab Duo CLI handler.

Вызывает `duo run` (или `glab duo cli run`) в headless-режиме через subprocess
и возвращает текстовый ответ как результат перевода.

Для больших промптов (перевод глав с глоссарием и инструкциями) промпт
записывается во временный файл и передаётся через shell-подстановку:
  duo run --goal "$(cat /tmp/duo_prompt_XXXX.txt)"

Это обходит лимит длины аргументов командной строки (~2 МБ на Linux).

Установка Duo CLI:
  - Через glab:   glab duo cli
  - Standalone:   bash <(curl -fsSL https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp/-/raw/main/packages/cli/scripts/install_duo_cli.sh)

Аутентификация:
  - Через glab:   автоматически (glab auth login)
  - Standalone:   duo auth login --token <PAT>
"""

import subprocess
import shutil
import os
import re
import time
import tempfile

from ..base import BaseApiHandler
from ..errors import (
    NetworkError,
    ModelNotFoundError,
    ValidationFailedError,
    TemporaryRateLimitError,
)

# Максимальное время ожидания ответа от Duo CLI (секунды)
_DEFAULT_TIMEOUT = 600

# Порог длины промпта (в символах), после которого используется временный файл
# вместо прямой передачи через --goal. 100 КБ — безопасный запас до лимита ARG_MAX.
_PROMPT_FILE_THRESHOLD = 100_000

# Паттерн для очистки ANSI escape-кодов из вывода терминала
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def _find_duo_executable(preferred: str | None = None) -> str | None:
    """
    Ищет исполняемый файл Duo CLI.
    Порядок поиска:
      1. Путь из настроек (preferred)
      2. `duo` в PATH
      3. `glab` в PATH (используется как `glab duo cli run`)
    Возвращает строку-идентификатор: "duo", "glab" или полный путь.
    """
    if preferred:
        if os.path.isfile(preferred) and os.access(preferred, os.X_OK):
            return preferred

    if shutil.which("duo"):
        return "duo"

    if shutil.which("glab"):
        return "glab"

    return None


def _build_base_command(executable: str, model_id: str | None = None) -> list[str]:
    """Собирает базовую команду (без --goal) в зависимости от типа исполняемого файла."""
    if executable == "glab" or (executable and executable.endswith("glab")):
        cmd = ["glab", "duo", "cli", "run"]
    else:
        cmd = [executable, "run"]

    if model_id and model_id != "default":
        cmd += ["--model", model_id]

    return cmd


class DuoCliApiHandler(BaseApiHandler):
    """
    Синхронный хендлер для GitLab Duo CLI.

    Стратегия передачи промпта:
      - Короткие промпты (< 100 КБ): напрямую через --goal "..."
      - Длинные промпты (>= 100 КБ): через временный файл + shell:
          sh -c '<base_cmd> --goal "$(cat /tmp/duo_prompt_XXXX.txt)"'

    В конфиге api_providers.json должно быть "is_async": false.
    """

    def setup_client(self, client_override=None, proxy_settings=None):
        super().setup_client(client_override, proxy_settings)

        # Путь к исполняемому файлу может быть задан как «api_key» в настройках,
        # либо определяется автоматически.
        custom_path = getattr(client_override, "api_key", None) or None
        if custom_path and custom_path.strip() in ("", "auto"):
            custom_path = None

        self.executable = _find_duo_executable(custom_path)
        self.worker.model_id = self.worker.model_config.get("id", "default")
        self.timeout = self.worker.provider_config.get("timeout_seconds", _DEFAULT_TIMEOUT)

        if not self.executable:
            self.worker._post_event(
                "log_message",
                {
                    "message": (
                        "[DuoCLI] ⚠️ Исполняемый файл Duo CLI не найден. "
                        "Установите `duo` или `glab` и убедитесь, что они доступны в PATH."
                    )
                },
            )

        return True

    # ------------------------------------------------------------------
    # Основной метод вызова API
    # ------------------------------------------------------------------

    def call_api(
        self,
        prompt: str,
        log_prefix: str,
        allow_incomplete: bool = False,
        use_stream: bool = True,
        debug: bool = False,
        max_output_tokens: int | None = None,
    ) -> str:
        if not self.executable:
            raise ModelNotFoundError(
                "GitLab Duo CLI не найден. "
                "Установите `duo` или `glab` и перезапустите приложение."
            )

        prompt_len = len(prompt)
        use_tempfile = prompt_len >= _PROMPT_FILE_THRESHOLD
        base_cmd = _build_base_command(self.executable, self.worker.model_id)

        self.worker._post_event(
            "log_message",
            {
                "message": (
                    f"[DuoCLI] {log_prefix} Промпт: {prompt_len:,} символов"
                    f"{' (через файл)' if use_tempfile else ''}"
                )
            },
        )

        if debug:
            print(f"[DuoCLI DEBUG] executable: {self.executable}")
            print(f"[DuoCLI DEBUG] model: {self.worker.model_id}")
            print(f"[DuoCLI DEBUG] prompt length: {prompt_len}")
            print(f"[DuoCLI DEBUG] use_tempfile: {use_tempfile}")

        start = time.perf_counter()
        tmp_path = None

        try:
            if use_tempfile:
                result = self._run_with_tempfile(base_cmd, prompt, debug)
            else:
                cmd = base_cmd + ["--goal", prompt]
                result = self._run_subprocess(cmd, debug)
        finally:
            # Временный файл удаляется в _run_with_tempfile
            pass

        elapsed = time.perf_counter() - start

        stdout = _strip_ansi(result.stdout or "").strip()
        stderr = _strip_ansi(result.stderr or "").strip()

        if debug:
            print(f"[DuoCLI DEBUG] returncode: {result.returncode}")
            print(f"[DuoCLI DEBUG] stdout ({len(stdout)} chars): {stdout[:500]}")
            print(f"[DuoCLI DEBUG] stderr: {stderr[:300]}")

        self.worker._post_event(
            "log_message",
            {
                "message": (
                    f"[DuoCLI] {log_prefix} Завершено за {elapsed:.1f}с "
                    f"(код возврата: {result.returncode}, ответ: {len(stdout):,} символов)"
                )
            },
        )

        # Ненулевой код возврата — ошибка
        if result.returncode != 0:
            self._raise_for_error(result.returncode, stderr, stdout)

        if not stdout:
            raise ValidationFailedError(
                "Duo CLI вернул пустой ответ. "
                "Проверьте, что аутентификация выполнена (`duo auth login`)."
            )

        return stdout

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _run_subprocess(self, cmd: list[str], debug: bool = False) -> subprocess.CompletedProcess:
        """Запускает команду напрямую (для коротких промптов)."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired:
            raise TemporaryRateLimitError(
                f"GitLab Duo CLI не ответил за {self.timeout} секунд. "
                "Попробуйте позже или увеличьте timeout в настройках провайдера."
            )
        except FileNotFoundError:
            raise ModelNotFoundError(
                f"Исполняемый файл '{self.executable}' не найден. "
                "Проверьте установку Duo CLI."
            )
        except OSError as exc:
            raise NetworkError(f"Ошибка запуска Duo CLI: {exc}") from exc

    def _run_with_tempfile(self, base_cmd: list[str], prompt: str, debug: bool = False) -> subprocess.CompletedProcess:
        """
        Записывает промпт во временный файл и передаёт через shell-подстановку.
        Это обходит лимит ARG_MAX для больших промптов.
        """
        tmp_fd = None
        tmp_path = None
        try:
            # Создаём временный файл с промптом
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix="duo_prompt_", suffix=".txt", text=True
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(prompt)
            tmp_fd = None  # fdopen закрыл дескриптор

            if debug:
                print(f"[DuoCLI DEBUG] temp file: {tmp_path} ({os.path.getsize(tmp_path):,} bytes)")

            # Собираем shell-команду с подстановкой содержимого файла
            # Экранируем путь к файлу на случай пробелов
            escaped_tmp = tmp_path.replace("'", "'\\''")
            base_cmd_str = " ".join(
                arg.replace("'", "'\\''") if " " in arg else arg
                for arg in base_cmd
            )
            shell_cmd = f"{base_cmd_str} --goal \"$(cat '{escaped_tmp}')\""

            if debug:
                print(f"[DuoCLI DEBUG] shell_cmd: {shell_cmd[:200]}...")

            try:
                return subprocess.run(
                    shell_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={**os.environ},
                )
            except subprocess.TimeoutExpired:
                raise TemporaryRateLimitError(
                    f"GitLab Duo CLI не ответил за {self.timeout} секунд. "
                    "Попробуйте позже или увеличьте timeout в настройках провайдера."
                )
            except FileNotFoundError:
                raise ModelNotFoundError(
                    f"Исполняемый файл '{self.executable}' не найден. "
                    "Проверьте установку Duo CLI."
                )
            except OSError as exc:
                raise NetworkError(f"Ошибка запуска Duo CLI: {exc}") from exc
        finally:
            # Гарантированно удаляем временный файл
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _raise_for_error(returncode: int, stderr: str, stdout: str):
        """Анализирует вывод ошибки и бросает типизированное исключение."""
        error_detail = stderr or stdout or "нет вывода"
        lower = error_detail.lower()

        if "rate limit" in lower or "too many requests" in lower:
            raise TemporaryRateLimitError(f"Duo CLI: rate limit — {error_detail[:200]}")

        if "unauthorized" in lower or "authentication" in lower or "token" in lower:
            raise ValidationFailedError(
                f"Duo CLI: ошибка аутентификации — {error_detail[:200]}. "
                "Выполните `duo auth login` или `glab auth login`."
            )

        if "not found" in lower or "command not found" in lower:
            raise ModelNotFoundError(f"Duo CLI: команда не найдена — {error_detail[:200]}")

        raise NetworkError(
            f"Duo CLI завершился с ошибкой (код {returncode}): {error_detail[:300]}"
        )
