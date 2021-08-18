# Handwritten Digit Recognizer

This project aims to provide a Artificial Intelligence solution to recognize Handwritten Digits. The program builds and trains two different convolutional neual networks and gives a prediction based on a user-drawn digit. 
The models are trained using the MNIST dataset, and the interface is developed in Python with PyQt for an easy GUI for user interaction

Authors: Team 28 (<b>Julia Shan</b>, <b>Akash Ashok</b>)


## Screenshots of Project

#### Main Window
![Alt text](/screenshots/mainWindow.PNG?raw=true "Main Window")
#### Training Window
![Alt text](/screenshots/trainingWindow.PNG?raw=true "Training Window")
#### Trained Image Viewer
![Alt text](/screenshots/trainImages.PNG?raw=true "Trained Image Viewer")
#### Tested Image Viewer
![Alt text](/screenshots/testimages.PNG?raw=true "Tested Image Viewer")
#### Blank Prediction Window
![Alt text](/screenshots/predictionWindow.PNG?raw=true "Blank Prediction Window")
#### Predicted Window
![Alt text](/screenshots/predictedWindow.PNG?raw=true "Predicted Window")

## Versions 
- <b> Version 1.2 </b>
		-Required libraries and packages: refer to requirement.txt
		-Program executes from  main.py
		-This version contains two convolutional neural network models, one based on the existing Lenet model and another custom model.
		-Progress bar implemented with multi-threading allowing download and train actions to run while progress bar updates.
		-Drawing canvas fitted on the prediction window; the prediction bar graph is implemented to diplay class probabilities
		-Training and testing image sets can be viewed
		-The model accuracies are displayed on textbrowser when model is trained
- <b> Version 1.1</b>
		-Required libraries and packages: refer to requirement.txt
		-Program executes from GUI.py
		-This version contains two convolutional neural network models, one based on the existing Lenet model and another custom model.
		-This version contains a progress bar on the training window
		-The drawing canvas is on a separate window to the prediction window
		-Training and testing image sets can be viewed
- <b> Version 1.0</b>
		-Required libraries and packages: refer to requirement.txt
		-Program executed from GUI.py
		-This version contains one convolutional network model. 

## Setup

- Install Python 3.8 
- Install miniconda3
- Install PyQt5
- Install required packages (refer to requirement.txt)

## How to Run 

- Run main.py
- To open the training window, click 'File' and then'Train Model'
	- To download the dataset, click 'Download MNIST'
	- To train the model and view accuracy, click 'Train'
	- To cancel downloading or training, click 'Cancel'
- To view images from the training set, click 'View' then 'View Training Images'
- To view images from the training set, click 'View' then 'View Testing Images'
	- To view next page of images click 'Next'
	- To view previous page of images click 'Previous'
- To open prediction window, click 'Test' then 'Predict Model'
	- To recognise drawn digit on canvas click 'Recognize'
	- To clear drawn digit on canvas click 'Clear'
	- To select model choose between the 'CNN' and 'Lenet' buttons
- To exit the program, click 'File' then 'Quit' in the main window, or type 'Ctrl + Q'

## File save locations
- The dataset is saved in /dataset
- The images from the dataset are save in /images/test and /images/train. Do not delete the .keep files as it creates the directory for the images to be saved
- The trained models are saved in /scripts/results
- The user-drawn digit is saved as a png in /scripts/results

	


