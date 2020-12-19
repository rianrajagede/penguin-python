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
input layer : 4 neuron, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris, Softmax Layer

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = default from keras.optimizer.SGD, 0.001
epoch = 50
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical

#change target format
ytrain = to_categorical(ytrain) 

#build model
model = tf.keras.Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Activation("relu"))
model.add(Dense(3))
model.add(Activation("softmax"))

#choose optimizer and loss function
model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                metrics=['accuracy'])

#train
model.fit(xtrain, ytrain, epochs=50, batch_size=16)

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

#get prediction
classes = np.argmax(model.predict(xtest), axis=-1)

#get accuration
import numpy as np
accuration = np.sum(classes == ytest)/len(ytest) * 100

print("Test Accuration : " + str(accuration) + '%')
print("Prediction :")
print(classes)
print("Target :")
print(np.asarray(ytest,dtype="int32"))