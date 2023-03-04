# Author Dr. M. Alwarawrah
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
# PyTorch Library
import torch

# Import Class Linear
from torch.nn import Linear
# Set random seed
torch.manual_seed(1)

# Library for this section
from torch.nn import Linear
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch import sigmoid
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils

from sklearn.model_selection import train_test_split

#print if the code is using GPU/CUDA or CPU
if torch.cuda.is_available() == True:
    print('This device is using CUDA')
    device = torch.device("cuda:0")
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
else:    
    print('This device is using CPU')
    device = torch.device("cpu")

# start recording time
t_initial = time.time()

# Class Linear_reg for Neural Network Model
class Linear_reg(nn.Module):
    # Constructor
    def __init__(self, in_size, out_size):

        super(Linear_reg, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    
    # Prediction
    def forward(self, x):
        x = self.linear(x)
        return x

#normalization function
def normalize(features):
    norm = (features - features.mean()) / features.std()
    return norm

#accuracy and Percent Error function
def accu_func(y_pred, y_test):
    error = (abs(y_pred-y_test)/y_test)
    percent_error = error.mean()
    accuracy = 100 - percent_error
    return accuracy, percent_error

#plot Loss and Accuracy vs epoch
def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc):
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(train_loss, color='k', label = 'Training Loss')
    ax.plot(val_loss, color='r', label = 'Validation Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(train_acc, color='b', label = 'Training Accuracy')
    ax2.plot(val_acc, color='g', label = 'Validation Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=16)
    fig.legend(loc ="center")
    fig.tight_layout()
    plt.savefig('loss_accuracy_epoch.png')

#plot Data output after prediction
def plot_data(x_test, y_test, y_pred,colnames_features, colnames_target):
    plt.clf()
    plt.scatter(x_test,y_test, color='k', label = 'Validation')
    plt.scatter(x_test, y_pred, color='r', label = 'Prediction')
    plt.xlabel('%s'%colnames_features, fontsize=14)
    plt.ylabel('%s'%colnames_target, fontsize=16)
    plt.title('Accuracy: %5.3f and PE: %5.3f'%(float(accu_func(y_pred, y_test)[1]),float(accu_func(y_pred, y_test)[0])), fontsize='12')
    plt.tight_layout()
    plt.legend()
    plt.savefig('%s_vs_%s.png'%(colnames_target, colnames_features))

# Download data from: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv
concrete_data = pd.read_csv('concrete_data.csv')

output_file = open('output.txt','w')
print('Torch version: {}'.format(torch.__version__), file=output_file)

#print
print(concrete_data.head(), file=output_file)
print('data domensions: {}'.format(concrete_data.shape), file=output_file)
print(concrete_data.describe(), file=output_file)

#check if data has null
print('number of null in data:', file=output_file)
print(concrete_data.isnull().sum(), file=output_file)

#define features and target data
features = concrete_data.drop(['Strength'],axis=1)
colnames_features = features.columns.values
target = concrete_data['Strength']
colnames_target = 'Strength'

#print
print('Features:', file=output_file)
print(features.head(), file=output_file)
print('target:', file=output_file)
print(target.head(), file=output_file)


#convert to tensor
features_tensor = torch.tensor(features.values).to(torch.float32)
target_tensor = torch.tensor(target.values).to(torch.float32)

#split the dataset to train and test
X_train, X_test, y_train, y_test = train_test_split( features_tensor, target_tensor, test_size=0.3, random_state=4)

#normalize features x train and x test
x_train = normalize(X_train)
x_test = normalize(X_test)

#create training and validation dataset
train_tensor = data_utils.TensorDataset(x_train, y_train) 
val_tensor = data_utils.TensorDataset(x_test, y_test) 

#input and output dimension
input_dim = x_train.shape[1]
output_dim = 1

# load training and validation data as batches
train_loader=DataLoader(dataset=train_tensor, batch_size=50)
validation_loader = DataLoader(dataset=val_tensor, batch_size=50)

# Create model
model  = Linear_reg(input_dim, output_dim)
#load the model to device
model = model.to(device)
#define a criterion to calculate the loss
criterion = torch.nn.MSELoss() 
#define an optimizer
learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Define the function to train model
#define lists
train_loss = []
train_percent_error = []
train_accuracy = []
val_loss = []
val_percent_error = []
val_accuracy = []
epochs = 100
#train the model
for epoch in range(epochs):
    acc = []
    loss_sum = 0
    PE = []
    #Training
    for x,y in train_loader:
        #load the data to device
        x,y = x.to(device), y.to(device)
        #get prediction
        z = model(x)
        #calculate loss
        loss = criterion(z, y.unsqueeze(1))
        #Sets the gradients of all optimized torch.Tensor s to zero
        optimizer.zero_grad()
        # computes dloss/dx for every parameter x
        loss.backward()
        #Performs a single optimization step (parameter update)
        optimizer.step()
        #sum loss
        loss_sum +=  loss.data.item()
        #use accu_func to calculate percent error and accuracy
        percent_error, accuracy = accu_func(z, y)
        #append values
        acc.append(accuracy.data.item())
        PE.append(percent_error.data.item())
    #append training values for loss, PE and accuracy    
    train_loss.append(loss_sum)
    train_percent_error.append(np.mean(PE))
    train_accuracy.append(np.mean(acc))
    #print training info to screen and file
    print("Training, epoch: %d, loss: %5.2f: %d"%(epoch, loss_sum,epoch), file=output_file)
    print("Training, epoch: %d, loss: %5.2f: %d"%(epoch, loss_sum,epoch))
    print("Training, epoch: %d, Accuracy: %5.3f & Percent Error: %5.3f"%(epoch, np.mean(acc),np.mean(PE)), file=output_file)
    print("Training, epoch: %d, Accuracy: %5.3f & Percent Error: %5.3f"%(epoch, np.mean(acc),np.mean(PE)))

    acc = []
    loss_sum = 0
    PE = []
    #validation
    for x,y in validation_loader:
        #load the data to device
        x,y = x.to(device), y.to(device)
        #get prediction
        yhat = model(x)
        #calculate loss
        loss = criterion(yhat, y.unsqueeze(1))
        #sum loss
        loss_sum +=  loss.data.item()
        #use accu_func to calculate percent error and accuracy
        percent_error, accuracy = accu_func(z, y)
        #append values
        acc.append(accuracy.data.item())
        PE.append(percent_error.data.item())
    #append validation values for loss, PE and accuracy    
    val_loss.append(loss_sum)
    val_percent_error.append(np.mean(PE))
    val_accuracy.append(np.mean(acc))
    #print Validation info to screen and file
    print("Validation, epoch: %d, loss: %5.2f: %d"%(epoch, loss_sum,epoch), file=output_file)
    print("Validation, epoch: %d, loss: %5.2f: %d"%(epoch, loss_sum,epoch))
    print("Validation, epoch: %d, Accuracy: %5.3f & Percent Error: %5.3f"%(epoch, np.mean(acc),np.mean(PE)), file=output_file)
    print("Validation, epoch: %d, Accuracy: %5.3f & Percent Error: %5.3f"%(epoch, np.mean(acc),np.mean(PE)))

#plot training and validation loss
plot_loss_accuracy(train_loss, val_loss, train_accuracy, val_accuracy)
output_file.close()

#store parameters
paramters = []
for param in model.parameters():
    paramters.append(param.data)

weights = paramters[0].cpu().numpy()
bias = paramters[1].cpu().numpy()

#plot test data with prediction
x_test = x_test.to(device)
model.eval()
y_pred = model(x_test)
#use X_test before normalization
for i in range(0,x_test.shape[1]):
    plot_data(X_test[:,i].detach().numpy(), y_test.detach().numpy(), y_pred.cpu().detach().numpy(),colnames_features[i], colnames_target)

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))