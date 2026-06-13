import weakref
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QSizePolicy, QTableWidget
from PyQt6.QtCore import Qt, pyqtSignal


class ExpandingTextEdit(QtWidgets.QTextEdit):
    geometryChangeRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Откладываем сигнал через QTimer чтобы не триггерить resizeRowToContents
        # синхронно во время инициализации делегата — это приводит к segfault в PyQt6
        self.textChanged.connect(lambda: QtCore.QTimer.singleShot(0, self.geometryChangeRequested.emit))

    def sizeHint(self):
        doc = self.document()
        vp = self.viewport()
        width = vp.width() if vp is not None and vp.width() > 0 else 200
        doc.setTextWidth(width)
        return QtCore.QSize(self.width(), int(doc.size().height()) + 5)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.updateGeometry()
        # Отложенный сигнал — не синхронный, не вызовет crash во время resize
        QtCore.QTimer.singleShot(0, self.geometryChangeRequested.emit)


class ExpandingTextEditDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = ExpandingTextEdit(parent)
        table = self.parent()
        if isinstance(table, QtWidgets.QTableWidget):
            row = index.row()
            table_ref = weakref.ref(table)
            def _resize_row():
                tbl = table_ref()
                if tbl is not None and 0 <= row < tbl.rowCount():
                    tbl.resizeRowToContents(row)
            editor.geometryChangeRequested.connect(_resize_row)
        editor.installEventFilter(self)
        return editor

    def setEditorData(self, editor, index):
        table = self.parent()
        if isinstance(table, QtWidgets.QTableWidget):
            table.blockSignals(True)
        try:
            value = index.model().data(index, QtCore.Qt.ItemDataRole.EditRole)
            editor.setPlainText(str(value) if value is not None else "")
        finally:
            if isinstance(table, QtWidgets.QTableWidget):
                table.blockSignals(False)

    def setModelData(self, editor, model, index):
        value = editor.toPlainText()
        model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)
        table = self.parent()
        if isinstance(table, QtWidgets.QTableWidget):
            table_ref = weakref.ref(table)
            row = index.row()
            QtCore.QTimer.singleShot(0, lambda: (
                (tbl := table_ref()) and
                (0 <= row < tbl.rowCount()) and
                tbl.resizeRowToContents(row)
            ))

    def sizeHint(self, option, index):
        if not index.isValid():
            return super().sizeHint(option, index)
        text = index.model().data(index, QtCore.Qt.ItemDataRole.DisplayRole)
        doc = QtGui.QTextDocument(str(text) if text else "")
        doc.setDefaultFont(option.font)
        doc.setTextWidth(max(option.rect.width() - 10, 1))
        height = int(doc.size().height()) + 10
        return QtCore.QSize(option.rect.width(), height)

    def updateEditorGeometry(self, editor, option, index):
        super().updateEditorGeometry(editor, option, index)

    def eventFilter(self, editor, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) \
               and not (event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                self.commitData.emit(editor)
                self.closeEditor.emit(editor, self.EndEditHint.SubmitModelCache)
                return True
        return super().eventFilter(editor, event)



class SmartTextEdit(ExpandingTextEdit):
    """
    "Умный" редактор, который автоматически сохраняет свои изменения
    при потере фокуса. Он знает, какой термин и какое поле он редактирует.
    """
    # Сигнал: (идентификатор_термина, имя_поля, новый_текст)
    data_committed = pyqtSignal(object, str, str)

    def __init__(self, identifier, field_name, initial_text, parent=None):
        super().__init__(parent)
        self.identifier = identifier
        self.field_name = field_name
        self.setPlainText(initial_text)
        self.setPlaceholderText(f"Введите {field_name}…")

    def focusOutEvent(self, event: QtGui.QFocusEvent):
        """Вызывается, когда виджет теряет фокус."""
        super().focusOutEvent(event)
        # Это главный триггер: при потере фокуса сообщаем о новом значении.
        self.data_committed.emit(self.identifier, self.field_name, self.toPlainText())

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Обрабатываем нажатие Enter для удобства."""
        # Если нажат Enter без Shift, считаем ввод завершенным
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) \
           and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            # Убираем фокус, что автоматически вызовет focusOutEvent и сохранение
            self.clearFocus()
        else:
            # В остальных случаях обрабатываем нажатие как обычно
            super().keyPressEvent(event)



class SingleRowTableWidget(QTableWidget):
    """
    Специализированная таблица, которая всегда состоит из одной строки и
    корректно сообщает компоновщику свой истинный, минимально необходимый размер,
    динамически подстраиваясь под высоту контента.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRowCount(1)
        
        # --- ИЗМЕНЕНИЕ 1: Более строгая политика ---
        # Policy.Fixed говорит: "Моя высота - это ТОЧНО мой sizeHint. Не растягивать!"
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Подключаемся к сигналу изменения размера хедера, чтобы реагировать на перенос слов
        self.horizontalHeader().sectionResized.connect(lambda: self.resizeRowToContents(0))


    def sizeHint(self) -> QtCore.QSize:
        """Переопределяем, чтобы сообщить идеальный размер."""
        total_height = 0
        if self.horizontalHeader().isVisible():
            total_height += self.horizontalHeader().height()
        
        if self.rowCount() > 0:
            # Учитываем высоту строки, которую установил делегат
            total_height += self.rowHeight(0)
        
        total_height += self.frameWidth() * 2

        return QtCore.QSize(super().sizeHint().width(), total_height)

    # --- ИЗМЕНЕНИЕ 2: "Недостающее звено" ---
    def resizeRowToContents(self, row: int):
        """
        Переопределяем стандартный метод. Сначала выполняем стандартное действие,
        а затем принудительно сообщаем компоновщику, что наш общий размер изменился.
        """
        super().resizeRowToContents(row)
        # Вот он, ключевой вызов!
        self.updateGeometry()