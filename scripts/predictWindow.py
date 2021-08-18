"""
Imports all the required Python Libraries
and files required to run the Prediction Window.
"""
import sys
import os.path
from PyQt5.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QMainWindow, 
    QGridLayout, 
    QAction, 
    QGroupBox,
    QLabel, 
    QPushButton, 
    QApplication, 
    QFileDialog, 
    QMessageBox,
    QErrorMessage,
    QRadioButton,
    )

from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5 import QtCore, QtGui
import pyqtgraph as pg
from model import RunNN
from left_tool_bar import LeftToolBar
from drawing_canvas import DrawingCanvas


"""
The Predict Window class represents the Window that contains the 
drawing canvas and the tool bar.
"""
class PredictWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.x1 = [0,0,0,0,0,0,0,0,0,0] # sets the x values for the bar chart
        self.chosen_learning_model = "CNN" # sets current learning model as CNN

        self.setWindowTitle("Prediction Window") # sets the title of the window
        self.setGeometry(100, 100, 900, 600)  # sets window geometry top, left, width, height

        """
        Initialises Gridview for layout management, the LeftToolBar and DrawingCanvas
        classes and calls methods defined further below which represents 
        each of the sections in the Left Tool Bar.
        """
        self.grid = QGridLayout()
        self.leftToolbar = LeftToolBar()
        self.canvasArea = DrawingCanvas()
        self.setTools()
        self.setModels()
        self.setBarGraph()
        self.setPredictedValue()

        """
        The Drawing Canvas and Left Side are assigned to the grid. 
        Sets a widget to be the central widget of the window and assigns 
        the grid to this widget. 
        """
        self.grid.addWidget(self.leftToolbar, 0, 0, 1, 1) # sets the left tool bar on row 0 col 0, taking up 1 rowspan and 1 columnspan.
        self.grid.addWidget(self.canvasArea, 0, 1, 1, 6) # sets the drawing on row 0 col 1, taking up 1 rowspan and 6 columnspans.
        win = QWidget()
        win.setLayout(self.grid)
        self.setCentralWidget(win)

        
        """
        Sets a menubar and adds an file menu item in it.
        """
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu(" File") 
        
        """
        Sets save action to the File menu and calls save method
        when it is clicked.
        """
        saveAction = QAction("Save", self)
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)

        """
        Sets clear action to the File menu and calls clear method
        when it is clicked.
        """
        clearAction = QAction("Clear", self)
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        """
        Sets quit action to the File menu and calls quitProgram method
        when it is clicked.
        """
        quitAction = QAction("Quit", self)
        fileMenu.addAction(quitAction)
        quitAction.triggered.connect(self.quitProgram)

        self.canvasArea.update() # updates the canvas area with the default settings

    """
    This method is called to set up the bargraph seen on the prediction window. 
    The bargraph is produced using the pyqtgraph.
    It is setup as a horizontal bargraph with the y values from 0 to 9
    and the x values from the self.x1 attribute which is set as a list of zeros.
    There are no inputs to this method.
    There are no outputs to this method.
    """
    def setBarGraph(self):
        self.classprob = QGroupBox("Class Proabablity") # sets title of group box 
        self.classprob.setMaximumHeight(500) # sets max height of group box so it doesnt get too large when resized.

        
        self.plot = pg.plot() # creates a plot from the pyqtgraph module.
        
        
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # y axis values representing numbers.
  
        self.bargraph = pg.BarGraphItem(x0 = 0, y=y , height = 0.6 , width = self.x1, brush ='g') # sets the bargraph values and sets the color of the bars as green.
  
        self.plot.addItem(self.bargraph) # adds the bargraph values to the plot.

        qv = QVBoxLayout()
        qv.addWidget(self.plot) # adds plot to the vertical box layout.
        self.classprob.setLayout(qv) # sets group box to the vertical box layout.
        self.leftToolbar.vertbox.addWidget(self.classprob) #sets the group box to the left tool bar.

    """
    This method is called to set up the Clear and Recognize buttons.
    There are no inputs to this method.
    There are no outputs to this method.
    """
    def setTools(self):
        self.groupBoxTools = QGroupBox("Tools")
        self.groupBoxTools.setMaximumHeight(300) # sets max height of group box so it doesnt get too large when resized.

        self.clearbtn = QPushButton("Clear")
        self.clearbtn.clicked.connect(self.clear) # calls the clear method when button pressed

        self.recognizebtn = QPushButton("Recognize")  
        self.recognizebtn.clicked.connect(self.recognize)  # calls the recognize method when button pressed

        qv = QVBoxLayout()
        
        qv.addWidget(self.clearbtn) # adds clear button to the vertical box layout 
        qv.addWidget(self.recognizebtn) # adds recognize button to the  vertical box layout 

        self.groupBoxTools.setLayout(qv) # adds clear button to the vertical box layout.

        self.leftToolbar.vertbox.addWidget(self.groupBoxTools)

    """
    This method is called to set up the choices of models 
    the user wants to use.
    There are no inputs to this method.
    There are no outputs to this method.
    """
    def setModels(self):
        self.model_type = QGroupBox("Models")
        self.model_type.setMaximumHeight(200) # sets max height of group box so it doesnt get too large when resized.

        self.modelBtn1 = QRadioButton("CNN")
        self.modelBtn1.clicked.connect(lambda: self.changeModelType(self.modelBtn1)) # when button clicked calls changeModelType method 
        self.modelBtn2 = QRadioButton("Lenet") 
        self.modelBtn2.clicked.connect(lambda: self.changeModelType(self.modelBtn2)) # when button clicked calls changeModelType method

        self.modelBtn1.setChecked(True)
        qv = QVBoxLayout()
        qv.addWidget(self.modelBtn1) # Adds CNN radio button to verical box layout
        qv.addWidget(self.modelBtn2) # Adds Lenet radio button to verical box layout
        self.model_type.setLayout(qv)
        self.leftToolbar.vertbox.addWidget(self.model_type)

    """
    This method is called to set up the label that 
    will be used to show the number the model predicts.
    There are no inputs to this method.
    There are no outputs to this method.
    """
    def setPredictedValue(self):
        self.groupBoxNumValue = QGroupBox("Predicted Value")
        self.groupBoxNumValue.setMaximumHeight(100) # sets max height of group box so it doesnt get too large when resized.


        self.predictednumber = QLabel()
        self.predictednumber.setText("N/A") # Sets default label as N/A 

        qv = QVBoxLayout()
        qv.addWidget(self.predictednumber) # adds label to vertical box layout
        self.groupBoxNumValue.setLayout(qv)

        self.leftToolbar.vertbox.addWidget(self.groupBoxNumValue)

    """
    This method is called to change the attribute chosen_learning_model
    depending on what button of the model the user wants to use.
    The method takes an input of the radio button that the user has selected.
    There are no outputs to this method.
    """
    def changeModelType(self, button):
        if button.text() == "CNN":
            if button.isChecked():
                self.chosen_learning_model = "CNN" # assigns chosen_learning_model attribute to CNN
        if button.text() == "Lenet":
            if button.isChecked():
                self.chosen_learning_model = "Lenet" # assigns chosen_learning_model attribute to Lenet

    """
    This method is called to save the image drawn on the drawing canvas.
    There are no inputs to this method.
    There are no outputs to this method.
    """
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPG(*.jpg *.jpeg);;All Files (*.*)")
        if filePath == "":
            return
        self.canvasArea.image.save(filePath)

    """
    This method is called to clear the drawing canvas to a plain white canvas,
    it also sets the predicted number label back to N/A and resets the barchart 
    with no values.
    There are no inputs to this method.
    There are no outputs to this method.
    """
    def clear(self):
        self.canvasArea.image.fill(Qt.white) # Fills back to a plain white one.
        self.canvasArea.update()
        x_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # A List of zeros to reset the bargraph. 
        self.updateBarChart(x_vals) # calls the clearBarChart method and passes the list of zeros
        self.updatePredictedNum("N/A") # calls the updatePredictedNum method and passes the value N/A 

    """
    This method is called to close the prediction window
    There are no inputs to this method.
    There are no outputs to this method.
    """
    def quitProgram(self):
        QtCore.QCoreApplication.quit()

    """
    This method is called take the image that the user has drawn and predict what 
    number they have drawn using the models trained. It also displays the data 
    on the bargraph and show the number on the label.
    There are no inputs to this method.
    There are no outputs to this method.
    """
    def recognize(self):
        
        filePath = "results/test.png" # sets the filepath for the saved drawing
        self.canvasArea.image.save(filePath) # saves the drawing to the filepath

        model_path_cnn = "results/model_cnn.pt" # sets a path to where the saved cnn model is 
        model_path_lenet = "results/model_lenet.pt" # sets a path to where the saved lenet model is 
        
        model = RunNN() # creates an object of the RunNN class
        if os.path.isfile(model_path_cnn) and os.path.isfile(model_path_lenet): # checks if the model files are in the file paths (Models have been trained)

            if self.chosen_learning_model == "CNN": 
                """
                Calls the prediciton method of the RunNN class and it returns a list of the probabilties of each of the numbers and the number with the highest probability.
                """
                predictionList, predictedNum = model.prediction(filePath, "results/model_cnn.pt") 
            else:
                """
                Calls the prediciton method of the RunNN class and it returns a list of the probabilties of each of the numbers and the number with the highest probability.
                """
                predictionList, predictedNum = model.prediction(filePath, "results/model_lenet.pt")

            self.updateBarChart(predictionList[0]) # passes the values from the predictionList as the x values for the bargraph
            self.updatePredictedNum(str(predictedNum)) # passes the number with the highest probabilty as the x values for the bargraph

        else: # if the saved model files are not present
            self.errorMsg() # call the errorMsg method

    """
    This method is called to update the barchart after calling the recognize button.
    The method takes an input of the x_vals which is the list of predicted numbers 
    to update the barchart to.
    There are no outputs to this method.
    """
    def updateBarChart(self, x_vals):

        x1 = x_vals
        y = [0,1,2,3,4,5,6,7,8,9] # sets the y values for the barchart

        self.bargraph.setOpts(y=y, width = x1) # updates to the new values 
        self.plot.addItem(self.bargraph) 

    """
    This method is called to update the label with the predicted value after calling 
    the recognize button. The method takes an input of the value which is the value that 
    should be displayed on the prediction window.
    There are no outputs to this method.
    """
    def updatePredictedNum(self, value):
        self.predictednumber.setText(value) # sets the test of the label to the input of the method
        self.predictednumber.update()

    """
    This method is called to display an error message dialog.
    There are no inputs to this method.
    There are no outputs to this method.
    """ 
    def errorMsg(self):
        error_dialog = QErrorMessage() # sets Error Message QtWidget
        error_dialog.showMessage("Please Train Models before Recognizing!") # Sets message to be displayed
        error_dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictWindow()  # creates instance of the PredictWindow class
    window.show() # shows MainWindow class
    app.exec()