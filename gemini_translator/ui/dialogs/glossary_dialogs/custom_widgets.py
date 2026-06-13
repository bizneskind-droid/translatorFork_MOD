import weakref
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QSizePolicy, QTableWidget
from PyQt6.QtCore import Qt, pyqtSignal


class ExpandingTextEdit(QtWidgets.QTextEdit):
    geometryChangeRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textChanged.connect(lambda: QtCore.QTimer.singleShot(0, self.geometryChangeRequested.emit))

    def sizeHint(self):
        doc = self.document()
        vp = self.viewport()
        width = vp.width() if vp is not None and vp.width() > 0 else 200
        doc.setTextWidth(width)
        return QtCore.QSize(self.width(), int(doc.size().height()) + 5)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self.updateGeometry)
        QtCore.QTimer.singleShot(0, self.geometryChangeRequested.emit)


class ExpandingTextEditDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        # Используем QLineEdit — QTextEdit/QPlainTextEdit крашат PyQt6 в делегатах
        editor = QtWidgets.QLineEdit(parent)
        return editor

    def setEditorData(self, editor, index):
        table = self.parent()
        if isinstance(table, QtWidgets.QTableWidget):
            table.blockSignals(True)
        try:
            value = index.model().data(index, QtCore.Qt.ItemDataRole.EditRole)
            editor.setText(str(value) if value is not None else "")
        finally:
            if isinstance(table, QtWidgets.QTableWidget):
                table.blockSignals(False)

    def setModelData(self, editor, model, index):
        value = editor.text()
        model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        super().updateEditorGeometry(editor, option, index)


class SmartTextEdit(ExpandingTextEdit):
    """
    "Умный" редактор, который автоматически сохраняет свои изменения
    при потере фокуса.
    """
    data_committed = pyqtSignal(object, str, str)

    def __init__(self, identifier, field_name, initial_text, parent=None):
        super().__init__(parent)
        self.identifier = identifier
        self.field_name = field_name
        self.setPlainText(initial_text)
        self.setPlaceholderText(f"Введите {field_name}…")

    def focusOutEvent(self, event: QtGui.QFocusEvent):
        super().focusOutEvent(event)
        self.data_committed.emit(self.identifier, self.field_name, self.toPlainText())

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) \
           and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.clearFocus()
        else:
            super().keyPressEvent(event)


class SingleRowTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRowCount(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalHeader().sectionResized.connect(lambda: self.resizeRowToContents(0))

    def sizeHint(self) -> QtCore.QSize:
        total_height = 0
        if self.horizontalHeader().isVisible():
            total_height += self.horizontalHeader().height()
        if self.rowCount() > 0:
            total_height += self.rowHeight(0)
        total_height += self.frameWidth() * 2
        return QtCore.QSize(super().sizeHint().width(), total_height)

    def resizeRowToContents(self, row: int):
        super().resizeRowToContents(row)
        self.updateGeometry()
