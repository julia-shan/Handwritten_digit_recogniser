from __future__ import print_function
import torch
import torchvision
import torch.optim as optim
import torch.onnx as onnx
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.lenet5 import create_lenet
from models.new_cnn import CNN
from PIL import Image
from io import BytesIO
import numpy as np
import time
import gzip
import requests

"""Called from imageWindow to convert and save images"""
def dispImg(start, stop, dataset):
    """some code sourced from https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
    code sourced to see how to extract image data and convert them into images
        extracting data from dataset"""
    train = gzip.open('../dataset/MNIST/raw/train-images-idx3-ubyte.gz', 'r')
    test = gzip.open('../dataset/MNIST/raw/t10k-images-idx3-ubyte.gz', 'r')
    image_size = 28
    num_images = 60000
    num_test_images = 10000
    #reading magic number, number of images, number of rows, and number of columns
    train.read(16) 
    test.read(16)
    trainbuf = train.read(image_size * image_size * num_images)
    testbuf = test.read(image_size * image_size * num_test_images)

    #converting image data into array and reshaped  
    trainData = np.frombuffer(trainbuf, dtype=np.uint8).astype(np.float32)
    trainData = trainData.reshape(num_images, image_size, image_size, 1) 
    testData = np.frombuffer(testbuf, dtype=np.uint8).astype(np.float32)
    testData = testData.reshape(num_test_images, image_size, image_size, 1)
    

    #looping through each image in dataset
    if(dataset == 'train'): #for training dataset
        for i in range(start, stop): 
            image = np.asarray(trainData[i]).squeeze() #converts dataset data into 2D 28x28 array
            im = Image.fromarray(image).convert('RGB') #converts image into a saveable rgb image
            im.save('../images/train/image' + str(i) + '.png', 'PNG') #saves image
    elif(dataset == 'test'): #for testing dataset
        for i in range(start, stop): 
            image1 = np.asarray(testData[i]).squeeze() #converts dataset data into 2D 28x28 array
            im1 = Image.fromarray(image1).convert('RGB') #converts image into a saveable rgb image
            im1.save('../images/test/image' + str(i) + '.png', 'PNG')
    
"""
RunNN class sets up the Neural Network, Trains the model and makes predictions.
"""
class RunNN():
    
    #Code sourced from https://stackoverflow.com/questions/66577151/http-error-when-trying-to-download-mnist-data to avoid https error when downloading dataset
    datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
    ]

    def __init__(self):

        self.saving_name = "model.pt" # Initialises the saving name of the model.
        self.path = "test.png" # Initialises the path of the saved drawn image.
        self.learning_rate = 0.001  # The step size at each iteration whihc moving towards a minimum loss function.
        self.batch_size = 64 # number of training examples used in one iteration of training.
        self.num_epochs = 3 # Initialises the number of times the algorithm works through the training dataset.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # checks if device supports cuda tensor types else uses CPU tensors.


    """
    This method is for testing purposes the CNN model.
    The method sets the saving name for the trained CNN model, sets up the 
    model and trains the model.
    The method takes no Inputs.
    The method has no Outputs.
    """
    def runCNN(self):

        self.saving_name = "results/model_cnn.pt"
        setup_list = self.setup()
        model = CNN()
        self.train("CNN")

    """
    This method is for testing purposes the Lenet model.
    The method sets the saving name for the trained Lenet model, sets up the 
    model and trains the model.
    The method takes no Inputs.
    The method has no Outputs.
    """
    def runLenet5(self):

        self.saving_name = "results/model_lenet.pt"
        setup_list = self.setup()
        model = create_lenet()
        self.train("Lenet")

    """
    This method gets the training and testing data from the MNIST Library
    The method takes no Inputs.
    The method returns a list with the training and testing data as well as 
    the dataloaders for the training and testing datasets.
    """
    def setup(self):

        train_dataset = datasets.MNIST('../dataset', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST('../dataset', train=False, transform=transforms.ToTensor(), download=True)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        return [train_dataset, test_dataset, train_loader, test_loader]

    """
    This method trains the model.
    The method takes an Input of the model type which is set as default as CNN.
    The method has Outputs of the accuracy of the Testing and Training datasets.
    """
    def train(self, model_type="CNN"):
        if model_type == "CNN":
            model = CNN()
            self.saving_name = "results/model_cnn.pt" # sets the saving path for the trained model
        else:
            model = create_lenet()
            self.saving_name = "results/model_lenet.pt" # sets the saving path for the trained model

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate) # implements the Adam Algorithm for stochastic optimisation

        print("Starting training!!!")
        
        for epoch in range(self.num_epochs):
            for batch_idx, (data, targets) in enumerate(self.setup()[2]):
        
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)

                scores = model(data)
                loss = criterion(scores, targets)
        
                optimizer.zero_grad() # sets the gradients to 0 before doing backpropragation
                loss.backward() # computes the derivative of the loss for every parameter 

                optimizer.step() # performs parameter update based on current gradient
                
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), 60000, 100. * batch_idx / 938, loss.item()))

        torch.save(model.state_dict(), self.saving_name) # saves the trained model to the path from self.saving_name
        print(f"Accuracy on training set: {self.check_accuracy(self.setup()[2], model, model_type)*100:.2f}") # Prints accuracy of training
        print(f"Accuracy on test set: {self.check_accuracy(self.setup()[3], model, model_type)*100:.2f}") # Prints accuracy of testing

        return round(float(self.check_accuracy(self.setup()[2], model, model_type))*100,2), round(float(self.check_accuracy(self.setup()[3], model, model_type))*100,2)

    """
    This method gives a prediction of the image.
    The method takes an Input of the path to where the image is located and the trained model.
    The method has Outputs of the list of probabilities of the predicted numbers 
    and the number with the highest probability.
    """
    def prediction(self, path, saving_name):

        pred = self.inference(path, saving_name)
        print(pred)
        pred = pred.tolist() # converts output of the inference method to a list
        pred_idx = np.argmax(pred) # grabs the largest probability of the pred list
        print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx]*100} %")
        return pred, pred_idx

    """
    This method checks the accuracy of the trained model.
    The method takes an Inputs of the train/test dataloader, the model itself, and which type of model to pass to the train function.
    The method has Outputs of the accuracy of the Testing and Training datasets.
    """
    def check_accuracy(self, loader, model, model_type):

        num_correct = 0
        num_samples = 0
        test_loss = 0
        model.eval() # turns some parts of the model that behave differently during training/inference off 

        with torch.no_grad(): # turns off gradient computation
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = model(x)
                test_loss += F.nll_loss(scores, y, reduction='sum').item()
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum() # sums total number of correct predictions
                num_samples += predictions.size(0) # finds total number of samples
        
        test_loss /= 10000
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, num_correct, 10000, 100. * num_correct / 10000))
        model.train(model_type) # reverese effects of model.eval() and turns back into training mode 
        return num_correct/num_samples

    """
    This method finds out the probability of each digit that the image could be.
    The method takes an Input of the path to where the image is located and the trained model.
    The method has Outputs of the list of probabilities that each digit could be.
    """
    def inference(self, image_path, saving_name):
        if saving_name[-6:] == "cnn.pt": # checks of the trained model passed in is CNN or not
            model = CNN()
        else:
            model = create_lenet()
            
        model.load_state_dict(torch.load(saving_name)) # loads the saved trained model
        model.eval() # turns some parts of the model that behave differently during training/inference off 

        img = Image.open(image_path).convert(mode="L") # opens the images and converts it to greyscale
        T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])
        img1 = img.resize((28, 28)) # resizes image to 28 x 28

        """
        Following Code Based on: https://medium.com/@krishna.ramesh.tx/training-a-cnn-to-distinguish-between-mnist-digits-using-pytorch-620f06aa9ffa 
        """
        x = (255 - np.expand_dims(np.array(img1), -1))/255.
    
        with torch.no_grad(): # temporarily sets the requires_grad flags to false
            pred = model(torch.unsqueeze(T(x), axis=0).float().to(self.device)) # adds a "fake" dimension and  we send in the image as a float
            return F.softmax(pred, dim=-1).cpu().numpy() # Softmax normalizes the output to a probability 

    
    

