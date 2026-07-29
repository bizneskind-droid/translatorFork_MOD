# gemini_translator/ui/dialogs/setup_dialogs/duo_cli_setup_dialog.py
# -*- coding: utf-8 -*-
"""
Диалог настройки GitLab Duo CLI.

Позволяет:
  - Проверить, установлен ли Duo CLI (duo / glab)
  - Указать кастомный путь к исполняемому файлу
  - Запустить аутентификацию прямо из UI
  - Скопировать команды установки
"""

import subprocess
import shutil
import os

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QTextEdit, QFileDialog,
    QDialogButtonBox, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject


# ---------------------------------------------------------------------------
# Фоновый воркер для запуска команд без блокировки UI
# ---------------------------------------------------------------------------

class _CommandWorker(QObject):
    finished = pyqtSignal(int, str, str)  # returncode, stdout, stderr

    def __init__(self, cmd: list[str]):
        super().__init__()
        self._cmd = cmd

    def run(self):
        try:
            result = subprocess.run(
                self._cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ},
            )
            self.finished.emit(result.returncode, result.stdout.strip(), result.stderr.strip())
        except subprocess.TimeoutExpired:
            self.finished.emit(-1, "", "Timeout: команда не ответила за 30 секунд")
        except FileNotFoundError:
            self.finished.emit(-1, "", f"Файл не найден: {self._cmd[0]}")
        except Exception as exc:
            self.finished.emit(-1, "", str(exc))


# ---------------------------------------------------------------------------
# Основной диалог
# ---------------------------------------------------------------------------

class DuoCliSetupDialog(QDialog):
    """
    Диалог настройки и проверки GitLab Duo CLI.
    Вызывается из KeyManagementWidget при выборе провайдера 'duo_cli'.
    """

    def __init__(self, parent=None, settings_manager=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self._thread = None
        self._worker = None

        self.setWindowTitle("Настройка GitLab Duo CLI")
        self.setMinimumSize(620, 520)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self._build_ui()
        self._load_saved_path()
        self._auto_detect()

    # ------------------------------------------------------------------
    # Построение UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # --- Статус ---
        status_group = QGroupBox("Статус Duo CLI")
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel("🔍 Проверка...")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.check_btn = QPushButton("🔄 Проверить снова")
        self.check_btn.clicked.connect(self._auto_detect)
        btn_row.addWidget(self.check_btn)
        btn_row.addStretch()
        status_layout.addLayout(btn_row)

        root.addWidget(status_group)

        # --- Путь к исполняемому файлу ---
        path_group = QGroupBox("Путь к исполняемому файлу (необязательно)")
        path_layout = QFormLayout(path_group)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Оставьте пустым для автоопределения (duo / glab из PATH)")
        path_row.addWidget(self.path_edit)

        browse_btn = QPushButton("📂")
        browse_btn.setFixedWidth(32)
        browse_btn.setToolTip("Выбрать файл")
        browse_btn.clicked.connect(self._browse_executable)
        path_row.addWidget(browse_btn)

        path_layout.addRow("Путь:", path_row)

        hint = QLabel(
            "Если <b>duo</b> или <b>glab</b> уже есть в PATH — поле можно оставить пустым.\n"
            "Введите <b>auto</b> или оставьте пустым для автоопределения."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 9pt;")
        path_layout.addRow("", hint)

        root.addWidget(path_group)

        # --- Аутентификация ---
        auth_group = QGroupBox("Аутентификация")
        auth_layout = QVBoxLayout(auth_group)

        auth_info = QLabel(
            "Для работы Duo CLI необходима аутентификация в GitLab.\n"
            "Нажмите кнопку ниже, чтобы запустить процесс входа в терминале."
        )
        auth_info.setWordWrap(True)
        auth_layout.addWidget(auth_info)

        auth_btn_row = QHBoxLayout()
        self.auth_duo_btn = QPushButton("🔑 duo auth login")
        self.auth_duo_btn.setToolTip("Запустить аутентификацию через standalone duo")
        self.auth_duo_btn.clicked.connect(lambda: self._run_auth("duo"))

        self.auth_glab_btn = QPushButton("🔑 glab auth login")
        self.auth_glab_btn.setToolTip("Запустить аутентификацию через glab")
        self.auth_glab_btn.clicked.connect(lambda: self._run_auth("glab"))

        auth_btn_row.addWidget(self.auth_duo_btn)
        auth_btn_row.addWidget(self.auth_glab_btn)
        auth_btn_row.addStretch()
        auth_layout.addLayout(auth_btn_row)

        root.addWidget(auth_group)

        # --- Установка ---
        install_group = QGroupBox("Установка Duo CLI")
        install_layout = QVBoxLayout(install_group)

        install_label = QLabel("Если Duo CLI не установлен, скопируйте нужную команду:")
        install_layout.addWidget(install_label)

        linux_row = QHBoxLayout()
        linux_cmd = 'bash <(curl -fsSL "https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp/-/raw/main/packages/cli/scripts/install_duo_cli.sh")'
        self.linux_edit = QLineEdit(linux_cmd)
        self.linux_edit.setReadOnly(True)
        linux_copy = QPushButton("📋")
        linux_copy.setFixedWidth(32)
        linux_copy.setToolTip("Скопировать")
        linux_copy.clicked.connect(lambda: self._copy_to_clipboard(linux_cmd, linux_copy))
        linux_row.addWidget(QLabel("Linux/macOS:"))
        linux_row.addWidget(self.linux_edit)
        linux_row.addWidget(linux_copy)
        install_layout.addLayout(linux_row)

        glab_row = QHBoxLayout()
        glab_cmd = "glab duo cli"
        self.glab_edit = QLineEdit(glab_cmd)
        self.glab_edit.setReadOnly(True)
        glab_copy = QPushButton("📋")
        glab_copy.setFixedWidth(32)
        glab_copy.setToolTip("Скопировать")
        glab_copy.clicked.connect(lambda: self._copy_to_clipboard(glab_cmd, glab_copy))
        glab_row.addWidget(QLabel("Через glab:  "))
        glab_row.addWidget(self.glab_edit)
        glab_row.addWidget(glab_copy)
        install_layout.addLayout(glab_row)

        root.addWidget(install_group)

        # --- Лог ---
        log_group = QGroupBox("Вывод команд")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QtGui.QFont("Consolas", 9))
        self.log_view.setMaximumHeight(120)
        log_layout.addWidget(self.log_view)
        root.addWidget(log_group)

        # --- Кнопки диалога ---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        btn_box.button(QDialogButtonBox.StandardButton.Save).setText("💾 Сохранить путь")
        btn_box.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        btn_box.accepted.connect(self._save_path)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Логика
    # ------------------------------------------------------------------

    def _load_saved_path(self):
        if not self.settings_manager:
            return
        saved = self.settings_manager.get("duo_cli_executable_path", "")
        if saved:
            self.path_edit.setText(saved)

    def _auto_detect(self):
        self.status_label.setText("🔍 Проверка...")
        self.check_btn.setEnabled(False)

        duo_path = shutil.which("duo")
        glab_path = shutil.which("glab")

        custom = self.path_edit.text().strip()
        custom_ok = bool(custom and custom not in ("auto", "") and os.path.isfile(custom))

        lines = []
        if custom_ok:
            lines.append(f"✅ Кастомный путь: <b>{custom}</b>")
        if duo_path:
            lines.append(f"✅ <b>duo</b> найден: {duo_path}")
        if glab_path:
            lines.append(f"✅ <b>glab</b> найден: {glab_path}")

        if not custom_ok and not duo_path and not glab_path:
            self.status_label.setText(
                "❌ <b>Duo CLI не найден.</b> Установите <code>duo</code> или <code>glab</code> "
                "и убедитесь, что они доступны в PATH."
            )
            self.status_label.setStyleSheet("color: #e74c3c;")
        else:
            self.status_label.setText("<br>".join(lines))
            self.status_label.setStyleSheet("color: #27ae60;")

        self.check_btn.setEnabled(True)

    def _browse_executable(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать исполняемый файл Duo CLI", "", "Все файлы (*)"
        )
        if path:
            self.path_edit.setText(path)
            self._auto_detect()

    def _run_auth(self, tool: str):
        """Запускает `duo auth login` или `glab auth login` в фоне и показывает вывод."""
        if tool == "duo":
            cmd = ["duo", "auth", "login"]
        else:
            cmd = ["glab", "auth", "login"]

        self._log(f"$ {' '.join(cmd)}\n")
        self._run_command_async(cmd)

    def _run_command_async(self, cmd: list[str]):
        self._thread = QThread()
        self._worker = _CommandWorker(cmd)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_command_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_command_finished(self, returncode: int, stdout: str, stderr: str):
        if stdout:
            self._log(stdout)
        if stderr:
            self._log(f"[stderr] {stderr}")
        if returncode == 0:
            self._log("✅ Команда выполнена успешно.\n")
        else:
            self._log(f"⚠️ Код возврата: {returncode}\n")
        self._auto_detect()

    def _log(self, text: str):
        self.log_view.append(text.rstrip())

    def _copy_to_clipboard(self, text: str, btn: QPushButton):
        QtWidgets.QApplication.clipboard().setText(text)
        original = btn.text()
        btn.setText("✓")
        btn.setEnabled(False)
        QtCore.QTimer.singleShot(2000, lambda: (btn.setText(original), btn.setEnabled(True)))

    def _save_path(self):
        path = self.path_edit.text().strip()
        if self.settings_manager:
            self.settings_manager.set("duo_cli_executable_path", path)
        QtWidgets.QMessageBox.information(self, "Сохранено", "Путь к Duo CLI сохранён.")
        self.accept()
