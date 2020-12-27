"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
penguin-clean-train.csv = datasets for training purpose, 70% from the original data
penguin-clean-test.csv  = datasets for testing purpose, 30% from the original data
"""

#Section 1.1 Data Loading
import pandas as pd

#load
datatrain = pd.read_csv('../Datasets/penguins-clean-train.csv')

#Section 1.2 Preprocessing
from sklearn.preprocessing import StandardScaler

#change string value to numeric
datatrain.loc[datatrain['species']=='Adelie', 'species']=0
datatrain.loc[datatrain['species']=='Gentoo', 'species']=1
datatrain.loc[datatrain['species']=='Chinstrap', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)

#change dataframe to array
datatrain_array = datatrain.values

#split x and y (feature and target)
xtrain = datatrain_array[:,1:]
ytrain = datatrain_array[:,0]

#standardize
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)

"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature from Palmer Penguin dataset
hidden layer : 10 neuron, activation using ReLU
output layer : 3 neuron, represents the number of species, Softmax Layer

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
epoch = 50
"""

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10),
                    solver='sgd',
                    learning_rate_init=0.01,
                    max_iter=50,
                    random_state=113)

# Train the model
mlp.fit(xtrain, ytrain)
print("Train accuration:", mlp.score(xtrain, ytrain))

"""
SECTION 3 : Testing model
"""
#load
datatest = pd.read_csv('../Datasets/penguins-clean-test.csv')

#change string value to numeric
datatest.loc[datatest['species']=='Adelie', 'species']=0
datatest.loc[datatest['species']=='Gentoo', 'species']=1
datatest.loc[datatest['species']=='Chinstrap', 'species']=2
datatest = datatest.apply(pd.to_numeric)

#change dataframe to array
datatest_array = datatest.values

#split x and y (feature and target)
xtest = datatest_array[:,1:]
ytest = datatest_array[:,0]

#standardization 
xtest = scaler.transform(xtest)

# Test the model
print("Test accuration:", mlp.score(xtest, ytest))



