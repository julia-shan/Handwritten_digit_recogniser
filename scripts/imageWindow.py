import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import * #imports all widgets for convenience
from PyQt5.QtCore import Qt
from model import RunNN, dispImg

#Displays image viewing window for both training and testing set
class imageWindow(QWidget):
    def __init__(self, dataset): #dataset will be either 'train' or 'test' determined from mainWindow
        self.dataset = dataset 
        super().__init__()
        self.initTestImageUI()
    
    def initTestImageUI(self):
        top = 500; left = 500; width = 500; height = 500        
        self.setGeometry(top, left, width, height) #set window size
        self.start = 0 #sets start to the first image index 

        """self.max is first image on the last page and as each page displays 100 images, max is determined by the number of images - 100"""
        if(self.dataset == 'train'):
            self.max = 60000-100 
            self.setWindowTitle("Train Images") 
        else:
            self.max = 10000-100
            self.setWindowTitle("Test Images")

        #@Scroll grid layout code sourced from https://python-forum.io/Thread-PyQt-QScrollArea-with-a-gridlayout
        self.layout = QVBoxLayout(self) #creates base layout for scroll area and horizontal button layout
        hlayout = QHBoxLayout() #horizontal layout for buttons
        
        """Create, name and set actions for previous and next buttons"""
        self.prevbutton = QPushButton("Previous")
        self.prevbutton.clicked.connect(self.prevSet)
        self.nextbutton = QPushButton("Next")
        self.nextbutton.clicked.connect(self.nextSet)

        """Aligns previous and next buttons horizontally """
        hlayout.addWidget(self.prevbutton)
        hlayout.addWidget(self.nextbutton)

        """Create grid layout with scroll for displaying images"""
        self.scrollArea = QScrollArea(self) #creates scoll area
        self.scrollArea.setWidgetResizable(True) #allows resizing
        self.scrollAreaWidgetContents = QWidget() #contents of scroll area as a widget instead of layout
        self.gridLayout = QGridLayout(self.scrollAreaWidgetContents) #creates grid layout andadds scroll area to grid layout
        self.scrollArea.setWidget(self.scrollAreaWidgetContents) #adds grid to scroll area
        self.layout.addWidget(self.scrollArea) #adds the scroll area to the base vertical layout
        self.layout.addLayout(hlayout) #adds the horizontal button layout to the base vertical layout below the grid scroll area
        dispImg(self.start, self.start + 100, self.dataset) #downloads and saves images
        self.showImages() #sets up first page of images
        self.prevbutton.setEnabled(False) #disable previous button as it is first page
        self.show() 

    def showImages(self):
        count = self.start #image index number
        #displays 100 images at a time in 20 rows and 5 columns
        for row in range(20):
            for column in range(5):
                pixmap = QPixmap('../images/' + self.dataset + '/image'+ str(count) + '.png') #sets saved png image as pixmap
                pixmap = pixmap.scaled(56, 56) #scales original 28x28 image by a factor of 2
                img = QLabel(self) 
                img.setPixmap(pixmap) #sets the pixmap to the label
                self.gridLayout.addWidget(img, row, column) #adds the image label to the current row and column
                count = count + 1 #moves to next image

        """When a page of images is shown, the next page of images is downloaded and saved to allow time for download and to  avoid displaying null pixmaps"""
        dispImg(count, count + 100, self.dataset) 

    """Pagination for image display
    Action for when next button is clicked"""
    def nextSet(self) :
        #set start and max as local variables for comparator
        start = self.start 
        max = self.max 

        #if not on last page change start value to first image index of next page, and enables next page button
        if(start != max): 
            self.start = self.start + 100 
            self.nextbutton.setEnabled(True) 
        start = self.start #updates local start value
        
        """Following if statements make sure the previous and next buttons are enabled/disables on the correct pages"""
        #if not on first page, previous button is enabled
        if(start != 0): 
            self.prevbutton.setEnabled(True) 
        
        #if on last page, next button is disabled
        if(start == max): 
            self.nextbutton.setEnabled(False) 
        
        #refreshes page and shows new set of images
        self.showImages() 
        self.show() 

    """Action for when previous button is clicked"""
    def prevSet(self):
        #set start and max as local variables for comparator
        start = self.start 
        max = self.max 

        #if not on first page change start value to first image index of previous page, and enables previous page button
        if(start != 0): 
            self.start = self.start - 100 
            self.prevbutton.setEnabled(True) 
        start = self.start #updates local start value
        
        """Following if statements make sure the previous and next buttons are enabled/disables on the correct pages"""
        #if on first page, previous button is disabled
        if(start == 0): 
            self.prevbutton.setEnabled(False)
        
        #if not on last page, next button is enabled
        if(start != max):
            self.nextbutton.setEnabled(True)
        
        #refreshes page and shows new set of images
        self.showImages() 
        self.show() 