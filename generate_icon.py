from PyQt6.QtGui import QImage, QPainter, QLinearGradient, QColor, QFont
from PyQt6.QtCore import Qt, QRectF
import os
import sys
from PyQt6.QtWidgets import QApplication

# Need QApplication context for QFont
app = QApplication(sys.argv)

os.makedirs('assets', exist_ok=True)
size = 1024
img = QImage(size, size, QImage.Format.Format_ARGB32)
img.fill(Qt.GlobalColor.transparent)

painter = QPainter(img)
painter.setRenderHint(QPainter.RenderHint.Antialiasing)
painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

# Blue Gradient
grad = QLinearGradient(0, 0, size, size)
grad.setColorAt(0.0, QColor("#00c6ff"))  # Light blue
grad.setColorAt(1.0, QColor("#0072ff"))  # Deep blue

# Draw rounded rect
painter.setBrush(grad)
painter.setPen(Qt.PenStyle.NoPen)
painter.drawRoundedRect(QRectF(50, 50, 924, 924), 200, 200)

# Draw Text
font = QFont("Helvetica Neue", 420, QFont.Weight.Bold)
painter.setFont(font)
painter.setPen(QColor("white"))
painter.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "LF")

painter.end()
img.save('assets/icon_1024x1024.png')
print("Icon generated at assets/icon_1024x1024.png")
