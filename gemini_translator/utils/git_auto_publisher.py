# gemini_translator/utils/git_auto_publisher.py
"""
Автопуш переведённых глав на GitHub.
Слушает EventBus: при каждом успешном task_finished
коммитит и пушит новые файлы из указанной папки.
"""

import os
import subprocess
import threading
import traceback

from PyQt6.QtCore import QObject, pyqtSlot


class GitAutoPublisher(QObject):
    """
    Подписывается на EventBus и при успешном завершении задачи
    выполняет git add + commit + push в фоновом потоке.

    Требования к папке:
      - Должна быть git-репозиторием (git init + remote уже настроены)
      - Авторизация через token в remote URL или SSH-ключ
    """

    def __init__(self, repo_path: str = "", enabled: bool = False, parent=None):
        super().__init__(parent)
        self.repo_path = repo_path
        self.enabled = enabled
        self._push_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Конфигурация
    # ------------------------------------------------------------------ #

    def configure(self, repo_path: str, enabled: bool):
        self.repo_path = repo_path
        self.enabled = enabled

    # ------------------------------------------------------------------ #
    #  Подключение к шине
    # ------------------------------------------------------------------ #

    def connect_to_bus(self, bus):
        bus.event_posted.connect(self.on_event)

    # ------------------------------------------------------------------ #
    #  Обработка событий
    # ------------------------------------------------------------------ #

    @pyqtSlot(dict)
    def on_event(self, event_data: dict):
        if not self.enabled or not self.repo_path:
            return

        event_name = event_data.get("event", "")
        data = event_data.get("data", {})

        if event_name == "task_finished" and data.get("success"):
            self._auto_push()

    # ------------------------------------------------------------------ #
    #  Git операции (в фоновом потоке)
    # ------------------------------------------------------------------ #

    def _auto_push(self):
        t = threading.Thread(target=self._do_git_push, daemon=True)
        t.start()

    def _do_git_push(self):
        """Коммит + пуш. Пропускает если нечего коммитить."""
        if not self._push_lock.acquire(blocking=False):
            # Другой пуш уже идёт — пропускаем, следующий task_finished подберёт
            return
        try:
            repo = self.repo_path

            # Проверяем есть ли изменения
            status = self._run_git(["status", "--porcelain"], repo)
            if not status.strip():
                return  # Нечего коммитить

            # Считаем новые/изменённые файлы для сообщения коммита
            changed = [l.strip() for l in status.strip().splitlines() if l.strip()]
            new_files = []
            for line in changed:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    new_files.append(os.path.basename(parts[1]))

            if len(new_files) == 1:
                msg = f"Add {new_files[0]}"
            else:
                msg = f"Add {len(new_files)} chapters"

            self._run_git(["add", "-A"], repo)
            self._run_git(["commit", "-m", msg], repo)
            self._run_git(["-c", "http.proxy=", "push"], repo)
            print(f"[GitAutoPublisher] Pushed: {msg}")

        except Exception as e:
            print(f"[GitAutoPublisher] Ошибка: {e}")
            traceback.print_exc()
        finally:
            self._push_lock.release()

    @staticmethod
    def _run_git(args: list, cwd: str) -> str:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
        return result.stdout
