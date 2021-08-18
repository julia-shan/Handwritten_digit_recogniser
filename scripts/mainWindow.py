import sys
from PyQt5.QtGui import QPixmap 
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from trainingWindow import TrainingWindow
from imageWindow import imageWindow
from predictWindow import PredictWindow

#Main window with menu bars to access the other windows
#The execution code to show the main window is called from main.py
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        top = 200
        left = 200
        width = 800
        height = 600

        self.setWindowTitle("Handwritten Digit Recognizer")
        self.setGeometry(top, left, width, height)
        
        #Add menu bar with three menus
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        viewMenu = mainMenu.addMenu("View")
        predictMenu = mainMenu.addMenu("Test")

        #Menu action for opening the training window
        trainAction = QAction("Train Model", self)
        fileMenu.addAction(trainAction)
        trainAction.triggered.connect(self.openTrainWindow)

        #Menu action for quitting program
        quitAction = QAction("Quit", self)
        fileMenu.addAction(quitAction)
        quitAction.setShortcut('Ctrl+Q')
        quitAction.triggered.connect(qApp.quit)

        #Menu action for opening the training images window
        viewtrainingAction = QAction("View Training Images", self)
        viewMenu.addAction(viewtrainingAction)
        viewtrainingAction.triggered.connect(self.openTrainImageWindow)

        #Menu action for opening the testing images window
        viewtestingAction = QAction("View Testing Images", self)
        viewMenu.addAction(viewtestingAction)  
        viewtestingAction.triggered.connect(self.openTestImageWindow)       

        #Menu action for opening the prediction window
        predictModelAction = QAction("Predict Model", self) 
        predictMenu.addAction(predictModelAction)     
        predictModelAction.triggered.connect(self.openPredictWindow)

 
    def openTrainWindow(self):
        self.dialog = TrainingWindow() #opens trainingwindow
        self.dialog.show()

    def openTrainImageWindow(self):
        self.dialog = imageWindow('train') #opens a imageWindow with input 'train' to view training images
        self.dialog.show()    

    def openTestImageWindow(self):
        self.dialog = imageWindow('test') #opens a imageWindow with input 'test' to view testing images
        self.dialog.show()

    def openPredictWindow(self):
        self.dialog = PredictWindow() #opens a predictWindow
        self.dialog.show()


if __name__ == "__main__":
    """Comment here to avoid EOF parsing error"""
    #The execution code to show the main window is called from main.py