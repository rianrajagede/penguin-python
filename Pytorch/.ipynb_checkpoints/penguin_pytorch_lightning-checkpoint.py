import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as LF
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# seed function for reproducibility
def random_seed(seed_value):
    np.random.seed(seed_value) 
    torch.manual_seed(seed_value) 
    random.seed(seed_value) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
        
random_seed(113) 

"""
SECTION 1 : Load and setup data

the datasets separated in two files from originai datasets:
penguin-clean-train.csv = datasets for training purpose, 70% from the original data
penguin-clean-test.csv  = datasets for testing purpose, 30% from the original data
"""

#Section 1.1 Data Loading

#load
datatrain = pd.read_csv('../Datasets/penguins-clean-train.csv')
datatest = pd.read_csv('../Datasets/penguins-clean-test.csv')

#Section 1.2 Preprocessing

#change string value to numeric
datatrain.loc[datatrain['species']=='Adelie', 'species']=0
datatrain.loc[datatrain['species']=='Gentoo', 'species']=1
datatrain.loc[datatrain['species']=='Chinstrap', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)

datatest.loc[datatest['species']=='Adelie', 'species']=0
datatest.loc[datatest['species']=='Gentoo', 'species']=1
datatest.loc[datatest['species']=='Chinstrap', 'species']=2
datatest = datatest.apply(pd.to_numeric)

#change dataframe to array
datatrain_array = datatrain.values
datatest_array = datatest.values

#split x and y (feature and target)
xtrain = datatrain_array[:,1:]
ytrain = datatrain_array[:,0]
xtest = datatest_array[:,1:]
ytest = datatest_array[:,0]

#standardize
#palmer-penguin dataset has varying scales
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature from Palmer Penguin dataset
hidden layer : 20 neuron, activation using ReLU
output layer : 3 neuron, represents the number of species, Softmax Layer

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
epoch = 50
"""

pl.utilities.seed.seed_everything(113)

# convert all dataset to tensor, make sure the data type is correct
xtrain = torch.from_numpy(xtrain).float()
xtest  = torch.from_numpy(xtest).float()
ytrain = torch.from_numpy(ytrain).long()
ytest  = torch.from_numpy(ytest).long()

# create PyTorch DataLoaders, PyTorch Lightning expects input to be DataLoader
trainloaders = DataLoader(TensorDataset(xtrain, ytrain), batch_size=8)
testloaders = DataLoader(TensorDataset(xtest, ytest), batch_size=8)

#hyperparameters
hl = 20
lr = 0.01
num_epoch = 50

#build lightning model
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx): # what happen in each train step
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = LF.accuracy(output, y)
        # self.log('train_loss', loss, on_epoch=True) # use this for logging (e.g. using TensorBoard)
        return {'loss':loss, "accuracy":acc}
    
    def test_step(self, batch, batch_idx): # what happen in each test step
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = LF.accuracy(output, y)
        return {'loss':loss, "accuracy":acc}
    
    def test_epoch_end(self, outputs): # what happens after testing
        loss_avg = torch.Tensor([x['loss'] for x in outputs]).mean()
        acc_avg = torch.Tensor([x['accuracy'] for x in outputs]).mean()
        return {'loss_avg':loss_avg, "accuracy_avg":acc_avg}
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer
        
model = Net()
trainer = pl.Trainer(gpus=1, max_epochs=20)
trainer.fit(model, trainloaders)

"""
SECTION 3 : Testing model
"""
trainer.test(model, testloaders)