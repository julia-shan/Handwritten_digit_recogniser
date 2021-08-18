from PyQt5.QtWidgets import QWidget

from PyQt5.QtGui import (
    QImage, 
    QPen, 
    QPainter, 
    QColor,
    QResizeEvent,
    )
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5 import QtCore, QtGui


"""
Class that represents our drawing canvas on the right hand side, 
it inherits from QWidget. 
"""
class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()

        """
        Class that represents our drawing canvas on the right hand side, 
        it inherits from QWidget. 
        """
        self.resizeImage = QImage(0, 0, QImage.Format_RGB32)

        """
        Sets our image and sets the drawing canvas to be filled in white. 
        """
        self.image = QImage(self.width(), self.height(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False # Sets default drawing settings to be false.
        self.brushSize = 30 # Sets size of the drawing brush strokes
        self.brushColor = Qt.black # Sets the colour of the brush to be Black. 

        self.lastPoint = QPoint() # Initialised a point used to draw lines.
        self.setMinimumWidth(150) # Sets mimimum width of the Drawing Canvas so it isnt too small.

    """
    This method is called when the window is resized. Sets the image to be scaled 
    correctly to the resized window.The input to the method is event 
    which checks if any changes are made to the state of the window.
    There are no outputs to the method.
    """
    def resizeEvent(self, event):
        self.image = self.image.scaled(self.width(), self.height())

    """
    This method is called when a button of the mouse is pressed, it sets a point in the.
    canvas when the mouse is clicked. The input to the method is event 
    which checks if any changes are made to the state of the window.
    There are no outputs to the method.
    """
    def mousePressEvent(self, event):
        """
        Checks only for left click as that is what we used to draw.
        """
        if event.button() == Qt.LeftButton:
            
            painter = QPainter(self.image)  # QPainter used to allow drawings to occur on the image.
            """
            Sets a Pen that allows drawing on the canvas with the default settings 
            of the brush colour, brush size, type of line, type of line end and type of arc between
            lines filled.
            """
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPoint(event.pos()) # draws point where the left click has occured.
            self.drawing = True  # Enables draw mode.
            self.lastPoint = event.pos()  # The new point made is set as last point.
            
            self.update() # Updates the widget something has been drawn.

    """
    This method is called when a button of the mouse is pressed, it sets a point in the.
    canvas when the mouse is clicked. The input to the method is event 
    which checks if any changes are made to the state of the window.
    There are no outputs to the method.
    """
    def mouseMoveEvent(self, event):
        """
        Checks that a left click is occuring and drawing mode is set true.
        """
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)  # QPainter used to allow drawings to occur on the image.
            """
            Sets a Pen that allows drawing on the canvas with the default settings 
            of the brush colour, brush size, type of line, type of line end and type of arc between
            lines filled.
            """
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos()) # draws line from first click to current position.
            self.lastPoint = event.pos() # The new point made is set as last point.
            self.update() # Updates the widget something has been drawn.

    """
    This method is called when a button of the mouse is released, sets the 
    resized image to the current drawing canvas. The input to the method is event 
    which checks if any changes are made to the state of the window.
    There are no outputs to the method.
    """
    def mouseReleaseEvent(self, event):
        """
        Checks if the left button is pressed
        """
        if event.button == Qt.LeftButton:
            self.resizeImage = self.image # Sets the drawing canvas to the resized saved image.
            self.drawing = False # Disables the drawing mode.

    """
    This method is called when drawing on the canvas. The input to the method 
    is event which checks if any changes are made to the state of the window.
    There are no outputs to the method.
    """
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())