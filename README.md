# Convolutional Neural Network and Multi Layer Perceptron #
## Description ##
in this tutorial, i would like to discuss about Convolutional Neural Network (CNN) and Multi Layer Perceptron (MLP) and its implementation in ** Pytroch **.  I move to pytorch because i need dynamic structure of neural network, it means we don't need to define computational graph and running the Graph like in Tensorflow. Both Tensorflow and Pytorch has plus minus respectively. MNIST dataset will be used in this tutorial. 

## CNN vs MLP ##

## Program ##
- Import Library
```python
import torch #import torch library
import torch.nn as nn #import torch neural network library
import torch.nn.functional as F #import functional neural network module
import torch.optim as optim #import optimizer neural network module
from torch.autograd import Variable #import variable that connect to automatic differentiation
from torchvision import datasets, transforms #import torchvision for datasets and transform
import argparse
```
- Defining Multilayer Perceptron Model
```python
class DNN(nn.Module):
	def __init__(self):
		super(DNN, self).__init__() #load super class for training data
		self.fc1 = nn.Linear(784, 320) #defining fully connected with input 784 and output 320
		self.fc2 = nn.Linear(320, 50) #defining fully connected with input 320 and output 50
		self.fc3 = nn.Linear(50, 10) #defining fully connected with input 50 and output 10
		self.relu = nn.ReLU() #defining Rectified Linear Unit as activation function
	
	def forward(self, x): #feed forward
		layer1 = x.view(-1, 784) #make it flat in one dimension from 0 - 784
		layer2 = self.relu(self.fc1(layer1)) #layer2 = layer1 -> fc1 -> relu
		layer3 = self.relu(self.fc2(layer2)) #layer3 = layer2 -> fc2 -> relu
		layer4 = self.relu(self.fc3(layer3)) #layer4 = layer3 -> fc2 -> relu
		return F.log_softmax(layer4) #softmax activation to layer4
```
- Defining Convolutional Neural Network Model
```python
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__() #load super class for training data
		self.conv1 = nn.Conv2d(1, 10, 5) #Convolutional modul: input 1, output 10, kernel 5
		self.conv2 = nn.Conv2d(10, 20, 5) #Convolutional modul: input 10, output 20, kernel 5
		self.maxpool = nn.MaxPool2d(2) #maxpooling modul: kernel
		self.relu = nn.ReLU() #activation relu modul
		self.dropout2d = nn.Dropout2d() #dropout modul
		self.fc1 = nn.Linear(320, 50) #Fully Connected modul: input, output
		self.fc2 = nn.Linear(50, 10)# Fully Connected modul: input, output

	def forward(self, x): #feed forward
		layer1 = self.relu(self.maxpool(self.conv1(x))) # layer1 = x -> conv1 -> maxpool -> relu
		layer2 = self.relu(self.maxpool(self.dropout2d(self.conv2(layer1)))) #layer2 = layer1 -> conv2 -> dropout -> maxpool -> relu
		layer3 = layer2.view(-1, 320) #make it flat from 0 - 320
		layer4 = self.relu(self.fc1(layer3)) #layer4 = layer3 -> fc1 -> relu
		layer5 = self.fc2(layer4) #layer5 = layer4 -> fc2
		return F.log_softmax(layer5) #softmax activation to layer5

```
## References ##

