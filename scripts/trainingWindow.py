import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from model import RunNN
import time

#Some code for progress bar and multithreading sourced from https://stackoverflow.com/questions/45157006/python-pyqt-pulsing-progress-bar-with-multithreading
#Training model window 
class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initTrainingUI()
         
    def initTrainingUI(self):
        top = 400
        left = 400
        width = 400
        height = 400

        #Sets title and size of window
        self.setWindowTitle("Dialog")
        self.setGeometry(top, left, width, height)
        
        #create textbox
        self.textbox = QTextBrowser(self)
        
        #create progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setFixedWidth(width)
        self.pbar.setFixedHeight(25)
        self.pbar.setAlignment(Qt.AlignCenter)
        self.progress = pyqtSignal(int) #creates a new signal 
        self.updateBar = Bar() #new instance of bar class
        self.updateBar.progress.connect(self.setProgress) #connects the progress bar to the singals emitted from QThread 

        #Create, name and set action for Download MNIST button
        self.downloadBtn = QPushButton("Download MNIST")
        self.downloadBtn.clicked.connect(self.downloadDataset)
        self.downloadBtn.setFixedWidth(150)
        self.download = Download()

        #Create, name and set action for Train button
        self.trainBtn = QPushButton("Train")
        self.trainBtn.clicked.connect(self.trainModel)
        self.trainBtn.setFixedWidth(100)
        self.train = Train()

        #Create, name and set action for Cancel button
        cancelBtn = QPushButton("Cancel")
        cancelBtn.clicked.connect(self.cancelAction)
        
        #create a QHboxlayout instance)
        btnlayout = QHBoxLayout()
        btnlayout.addStretch(1)
        btnlayout.addWidget(self.downloadBtn)
        btnlayout.addWidget(self.trainBtn)
        btnlayout.addWidget(cancelBtn)
        
        #create a QVBoxLayout instance
        vlayout = QVBoxLayout()
        vlayout.addStretch(1)
        vlayout.addWidget(self.textbox)
        vlayout.addStretch(1)
        vlayout.addWidget(self.pbar)
        vlayout.addStretch(1)
        vlayout.addLayout(btnlayout)
        vlayout.addStretch(1)
        self.setLayout(vlayout)
    
    def downloadDataset(self):
        self.textbox.append("Downloading MNIST dataset...") #prints text to the textbrowser
        self.downloadBtn.setEnabled(False) #disables the download button
        self.updateBar.start() #start updating progress bar
        self.download.start() #start download action
        self.download.finished.connect(lambda : self.updateBar.terminate()) #terminates progress bar updating when download is finished
        self.download.finished.connect(lambda : self.pbar.setValue(100)) #sets the progress bar to maximum when training is finished
        self.download.finished.connect(lambda : self.textbox.append("Download done!")) #prints text to the textbrowser
        self.download.finished.connect(lambda : self.downloadBtn.setEnabled(True)) #enables the download buton
    
    def trainModel(self):
        self.textbox.append("Training model...") #prints text to the textbrowser
        self.trainBtn.setEnabled(False) #disables the train button
        self.updateBar.start() #start updating progress bar
        self.train.start() #start train action
        self.train.finished.connect(lambda : self.updateBar.terminate()) #terminates progress bar updating when training is finished
        self.train.finished.connect(lambda : self.pbar.setValue(100)) #sets progress bar to maximum when training is finished
        self.train.finished.connect(lambda : self.textbox.append("Model trained!")) #prints text to the textbrowser
        self.train.cnn.connect(self.dispCNNAccuracy) #displays accuracy for new_cnn model
        self.train.lenet.connect(self.dispLenetAccuracy) #displays accuracy for lenet model
        self.train.finished.connect(lambda : self.trainBtn.setEnabled(True)) #enables the train button
    
    def cancelAction(self):
        self.updateBar.stop() #stops updating progress bar
        self.pbar.setValue(0) #sets progress bar value to min 
        self.download.terminate() #terminates download aciton
        self.train.terminate() #terminates train action 
        self.downloadBtn.setEnabled(True) #enables download button
        self.trainBtn.setEnabled(True) #enables train button 

    def setProgress(self, progress):
        self.pbar.setValue(progress) #passes the progress signal
    
    def dispCNNAccuracy(self, cnn):
        self.textbox.append("CNN accuracy - train: " + str(int(cnn[0])) + "% test: " + str(int(cnn[1])) + "%")

    def dispLenetAccuracy(self, lenet):
        self.textbox.append("Lenet accuracy - train: " + str(int(lenet[0])) + "% test: " + str(int(lenet[1])) + "%")

#Multithreading classes for the progress bar, download action and train action
class Bar(QThread):
    #finished = pyqtSignal()
    progress = pyqtSignal(int)
    def __init__(self):
        QThread.__init__(self)
        self.threadactive = True
    #time taken for progress bar to reach 100 is based on the average time taken for downloading the dataset
    def run(self):
        while(self.threadactive):
            for i in range(1, 101):
                self.progress.emit(i) #emits new value to update progress bar
                time.sleep(0.2) #waits 0.2 second
    
    def stop(self):
        self.threadactive = False
        self.finished.emit()
        self.exit()

class Download(QThread):
    finished = pyqtSignal()
    def __init__(self):
        QThread.__init__(self)
    
    def run(self):
        self.model = RunNN() #new instance of RunNN class
        self.model.setup() #downlaods dataset (if not already downloaded) from setup function in RunNN class
        self.finished.emit() #emits finish signal

class Train(QThread):
    #Signals created to be passed back into main thread of code 
    finished = pyqtSignal()
    cnn = pyqtSignal(tuple)
    lenet = pyqtSignal(tuple)
    
    def __init__(self):
        QThread.__init__(self)
        self.threadactive = True

    def run(self):
        model = RunNN() #new instance of RunNN class from model.py
        x = model.train("CNN") #trains the new_cnn model, sets accuracy value to x
        y = model.train("Lenet") #trains the lenet model, set accuracy value to y
        self.cnn.emit(x) #emits x value as signal
        self.lenet.emit(y) #emits y value as signal
        self.finished.emit() #emits finish signal

        