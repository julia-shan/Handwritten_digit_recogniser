from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout


"""
The Left Side Class is the Tool Bar on the Left Hand Side of the drawing 
Canvas that contains the buttons, the bar chart and the predicted number 
label. It inherits from QWidget.
"""
class LeftToolBar(QWidget):
    def __init__(self):
        super().__init__()

        """
        Sets the maximum and mimimum size of the tool bar gets,
        when the size of the window is increased. They are both set
        the same hence when increased the tool bar remains the same size
        when size is increased of decreased.
        """
        self.setMaximumWidth(250)
        self.setMinimumWidth(250)

        """
        Sets a the default layout of every thing contained in the tool bar
        as a vertical box layout.
        """
        self.vertbox = QVBoxLayout()
        self.setLayout(self.vertbox)