# -*- coding: utf-8 -*-
"""
Диалог настроек Telegram-уведомлений и Git Auto-Push.
"""
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QDialogButtonBox, QSpinBox,
    QGroupBox, QFileDialog,
)


class TelegramSettingsDialog(QDialog):
    def __init__(self, parent=None, settings_manager=None, notifier=None):
        super().__init__(parent)
        self.setWindowTitle("Telegram и Git Auto-Push")
        self.setMinimumWidth(450)

        if settings_manager is None:
            app = QtWidgets.QApplication.instance()
            self.settings_manager = getattr(app, 'settings_manager', None)
            if self.settings_manager is None:
                raise RuntimeError("SettingsManager не найден.")
        else:
            self.settings_manager = settings_manager

        self.notifier = notifier
        if self.notifier is None:
            app = QtWidgets.QApplication.instance()
            self.notifier = getattr(app, 'telegram_notifier', None)

        app = QtWidgets.QApplication.instance()
        self.git_publisher = getattr(app, 'git_auto_publisher', None)

        self._init_ui()
        self._load_settings()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ==================== Telegram ====================
        tg_group = QGroupBox("Telegram-уведомления")
        tg_layout = QVBoxLayout(tg_group)

        self.tg_enabled_cb = QCheckBox("Включить уведомления в Telegram")
        tg_layout.addWidget(self.tg_enabled_cb)

        # Bot Token
        tg_layout.addWidget(QLabel("Bot Token:"))
        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText("123456:ABC-DEF...")
        tg_layout.addWidget(self.token_edit)

        # Chat ID
        tg_layout.addWidget(QLabel("Chat ID:"))
        self.chat_id_edit = QLineEdit()
        self.chat_id_edit.setPlaceholderText("-1001234567890 или ваш user id")
        tg_layout.addWidget(self.chat_id_edit)

        # Notify every N chapters
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Уведомлять каждые"))
        self.freq_spin = QSpinBox()
        self.freq_spin.setMinimum(1)
        self.freq_spin.setMaximum(999)
        self.freq_spin.setValue(10)
        freq_layout.addWidget(self.freq_spin)
        freq_layout.addWidget(QLabel("глав"))
        freq_layout.addStretch()
        tg_layout.addLayout(freq_layout)

        # Test button
        self.test_btn = QPushButton("Отправить тест")
        self.test_btn.clicked.connect(self._send_test)
        tg_layout.addWidget(self.test_btn)

        layout.addWidget(tg_group)

        # ==================== Git Auto-Push ====================
        git_group = QGroupBox("Git Auto-Push")
        git_layout = QVBoxLayout(git_group)

        self.git_enabled_cb = QCheckBox("Автоматически пушить после каждой главы")
        git_layout.addWidget(self.git_enabled_cb)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Папка репозитория:"))
        self.repo_path_edit = QLineEdit()
        self.repo_path_edit.setPlaceholderText("~/novels/my_divine_diary/EPUB")
        path_layout.addWidget(self.repo_path_edit, 1)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(36)
        self.browse_btn.clicked.connect(self._browse_repo)
        path_layout.addWidget(self.browse_btn)
        git_layout.addLayout(path_layout)

        layout.addWidget(git_group)

        # ==================== Buttons ====================
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Сохранить")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ------------------------------------------------------------------ #
    #  Load / Save
    # ------------------------------------------------------------------ #

    def _load_settings(self):
        # Telegram
        s = self.settings_manager.load_telegram_settings()
        self.tg_enabled_cb.setChecked(s.get("enabled", False))
        self.token_edit.setText(s.get("bot_token", ""))
        self.chat_id_edit.setText(s.get("chat_id", ""))
        self.freq_spin.setValue(s.get("notify_every_n", 10))

        # Git
        g = self.settings_manager.load_git_autopush_settings()
        self.git_enabled_cb.setChecked(g.get("enabled", False))
        self.repo_path_edit.setText(g.get("repo_path", ""))

    def _save_and_accept(self):
        # Telegram
        tg = {
            "enabled": self.tg_enabled_cb.isChecked(),
            "bot_token": self.token_edit.text().strip(),
            "chat_id": self.chat_id_edit.text().strip(),
            "notify_every_n": self.freq_spin.value(),
        }
        self.settings_manager.save_telegram_settings(tg)

        if self.notifier:
            self.notifier.configure(
                bot_token=tg["bot_token"],
                chat_id=tg["chat_id"],
                enabled=tg["enabled"],
                notify_every_n=tg["notify_every_n"],
            )

        # Git
        git_settings = {
            "enabled": self.git_enabled_cb.isChecked(),
            "repo_path": self.repo_path_edit.text().strip(),
        }
        self.settings_manager.save_git_autopush_settings(git_settings)

        if self.git_publisher:
            self.git_publisher.configure(
                repo_path=git_settings["repo_path"],
                enabled=git_settings["enabled"],
            )

        self.accept()

    # ------------------------------------------------------------------ #
    #  Telegram Test
    # ------------------------------------------------------------------ #

    def _send_test(self):
        if self.notifier is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "TelegramNotifier не инициализирован.")
            return

        self.notifier.configure(
            bot_token=self.token_edit.text().strip(),
            chat_id=self.chat_id_edit.text().strip(),
            enabled=True,
            notify_every_n=self.freq_spin.value(),
        )
        ok, msg = self.notifier.send_test_message()
        if ok:
            QtWidgets.QMessageBox.information(self, "Успех", msg)
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", msg)

        self.notifier.enabled = self.tg_enabled_cb.isChecked()

    # ------------------------------------------------------------------ #
    #  Git Browse
    # ------------------------------------------------------------------ #

    def _browse_repo(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку Git-репозитория")
        if folder:
            self.repo_path_edit.setText(folder)
