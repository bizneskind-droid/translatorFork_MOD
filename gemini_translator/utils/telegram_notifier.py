# gemini_translator/utils/telegram_notifier.py
"""
Telegram-нотификатор для ключевых событий переводчика.
Подписывается на EventBus и отправляет сообщения в Telegram-чат
при старте/завершении сессии, ошибках и по прогрессу (каждые N глав).
"""

import threading
import time
import traceback
from typing import Optional

import requests
from PyQt6.QtCore import QObject, pyqtSlot


TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier(QObject):
    """
    Слушает EventBus и отправляет уведомления в Telegram.

    Поддерживаемые события:
      - session_started   → «Сессия запущена: модель X, N задач»
      - session_finished  → «Сессия завершена: причина»
      - fatal_error       → «ОШИБКА: ...»
      - geoblock_detected → «Гео-блокировка!»
      - task_finished (success=True) → прогресс каждые `notify_every_n` глав
    """

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        enabled: bool = False,
        notify_every_n: int = 10,
        parent=None,
    ):
        super().__init__(parent)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.notify_every_n = max(1, notify_every_n)

        # Счётчики для прогресса внутри сессии
        self._completed_count = 0
        self._failed_count = 0
        self._total_tasks = 0
        self._session_model = ""
        self._session_start_time: Optional[float] = None
        self._session_active = False

    # ------------------------------------------------------------------ #
    #  Конфигурация (вызывается из UI / SettingsManager)
    # ------------------------------------------------------------------ #

    def configure(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool,
        notify_every_n: int = 10,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.notify_every_n = max(1, notify_every_n)

    # ------------------------------------------------------------------ #
    #  Подключение к шине
    # ------------------------------------------------------------------ #

    def connect_to_bus(self, bus):
        """Подписаться на event_posted сигнал EventBus."""
        bus.event_posted.connect(self.on_event)

    # ------------------------------------------------------------------ #
    #  Обработка событий
    # ------------------------------------------------------------------ #

    @pyqtSlot(dict)
    def on_event(self, event_data: dict):
        if not self.enabled or not self.bot_token or not self.chat_id:
            return

        event_name = event_data.get("event", "")
        data = event_data.get("data", {})

        try:
            if event_name == "session_started":
                self._on_session_started(data)
            elif event_name == "session_finished":
                self._on_session_finished(data)
            elif event_name == "fatal_error":
                self._on_fatal_error(data)
            elif event_name == "geoblock_detected":
                self._on_geoblock()
            elif event_name == "task_finished":
                self._on_task_finished(data)
            elif event_name == "log_message":
                self._on_log_message(data)
        except Exception:
            # Никогда не роняем основной поток из-за Telegram
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    #  Обработчики конкретных событий
    # ------------------------------------------------------------------ #

    def _on_session_started(self, data: dict):
        self._completed_count = 0
        self._failed_count = 0
        self._total_tasks = data.get("total_tasks", 0)
        self._session_model = data.get("model_id", "?")
        self._session_start_time = time.time()
        self._session_active = True
        self._sent_log_keys = set()

        text = (
            f"▶ Сессия запущена\n"
            f"Модель: {self._session_model}\n"
            f"Задач: {self._total_tasks}"
        )
        self._send(text)

    def _on_session_finished(self, data: dict):
        self._session_active = False
        reason = data.get("reason", "—")

        elapsed = ""
        speed = ""
        if self._session_start_time:
            secs = time.time() - self._session_start_time
            mins = secs / 60
            if mins >= 60:
                elapsed = f"\nВремя: {mins / 60:.1f} ч"
            else:
                elapsed = f"\nВремя: {mins:.1f} мин"
            if self._completed_count > 0 and secs > 0:
                per_ch = secs / self._completed_count
                if per_ch >= 60:
                    speed = f"\nСкорость: {per_ch / 60:.1f} мин/глава"
                else:
                    speed = f"\nСкорость: {per_ch:.0f} сек/глава"

        failed_line = ""
        if self._failed_count > 0:
            failed_line = f"\nПровалено: {self._failed_count}"

        text = (
            f"■ Сессия завершена\n"
            f"Переведено: {self._completed_count} / {self._total_tasks}"
            f"{failed_line}{elapsed}{speed}\n"
            f"Причина: {reason}"
        )
        self._send(text)
        self._session_start_time = None

    def _on_fatal_error(self, data: dict):
        payload = data.get("payload", {})
        error_type = payload.get("type", "unknown")
        exc = payload.get("exception", "")
        text = f"Фатальная ошибка: {error_type}\n{exc}"
        self._send(text[:4000])

    def _on_geoblock(self):
        self._send("Обнаружена гео-блокировка! Сессия будет остановлена.")

    def _on_task_finished(self, data: dict):
        if not self._session_active:
            return

        if data.get("success"):
            self._completed_count += 1
            if self._completed_count % self.notify_every_n == 0:
                pct = ""
                if self._total_tasks > 0:
                    pct = f" ({self._completed_count * 100 // self._total_tasks}%)"
                text = f"Прогресс: {self._completed_count} / {self._total_tasks}{pct}"
                self._send(text)
        else:
            self._failed_count += 1
            # Сообщаем о провале задачи
            error_type = data.get("error_type", "")
            message = data.get("message", "")
            task_info = data.get("task_info", ())
            task_name = ""
            if task_info and len(task_info) > 1:
                import os
                payload = task_info[1]
                if payload and len(payload) > 2:
                    task_type = payload[0]
                    if task_type in ("epub", "epub_chunk"):
                        task_name = os.path.basename(str(payload[2]))
                    elif task_type == "epub_batch" and isinstance(payload[2], list):
                        names = [os.path.basename(str(p)) for p in payload[2][:3]]
                        task_name = ", ".join(names)

            short_msg = str(message)[:200] if message else error_type
            text = f"Провал: {task_name or '?'}\n{short_msg}"
            self._send(text[:2000])

    # Ключевые слова в log_message, которые стоит пересылать в Telegram
    _ERROR_KEYWORDS = (
        "ОКОНЧАТЕЛЬНЫЙ ПРОВАЛ",
        "ПРОВАЛ ПОПЫТОК",
        "[FATAL]",
        "[CRITICAL]",
        "NETWORK",
        "окончательно заблокирована",
        "временно заморожена",
        "ТУПИК",
        "РАБОТА ОСТАНОВЛЕНА",
        "все ключи исчерпаны",
    )

    def _on_log_message(self, data: dict):
        if not self._session_active:
            return
        msg = data.get("message", "")
        if not msg:
            return
        for kw in self._ERROR_KEYWORDS:
            if kw in msg:
                # Не спамить одним и тем же — дедупликация по первым 80 символам
                short_key = msg[:80]
                if not hasattr(self, '_sent_log_keys'):
                    self._sent_log_keys = set()
                if short_key in self._sent_log_keys:
                    return
                self._sent_log_keys.add(short_key)
                self._send(msg[:2000])
                return

    # ------------------------------------------------------------------ #
    #  Отправка (в отдельном потоке, чтобы не блокировать GUI)
    # ------------------------------------------------------------------ #

    def _send(self, text: str):
        """Отправить сообщение в Telegram в фоновом потоке."""
        t = threading.Thread(
            target=self._send_sync,
            args=(text,),
            daemon=True,
        )
        t.start()

    def _send_sync(self, text: str):
        try:
            url = TELEGRAM_API_URL.format(token=self.bot_token)
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            resp = requests.post(url, json=payload, timeout=10)
            if not resp.ok:
                print(f"[TelegramNotifier] Ошибка отправки: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"[TelegramNotifier] Не удалось отправить: {e}")

    # ------------------------------------------------------------------ #
    #  Тест соединения
    # ------------------------------------------------------------------ #

    def send_test_message(self) -> tuple[bool, str]:
        """
        Отправить тестовое сообщение. Возвращает (ok, описание).
        Вызывается синхронно из UI.
        """
        if not self.bot_token or not self.chat_id:
            return False, "Не заданы bot_token или chat_id"
        try:
            url = TELEGRAM_API_URL.format(token=self.bot_token)
            payload = {
                "chat_id": self.chat_id,
                "text": "✅ Тестовое сообщение от EPUB Translator",
                "disable_web_page_preview": True,
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.ok:
                return True, "Сообщение отправлено"
            else:
                return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return False, str(e)
