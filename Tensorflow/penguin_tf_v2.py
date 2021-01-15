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
#palmer-penguin dataset has varying scales
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
epochs = 50
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

#change target format
cat_ytrain = to_categorical(ytrain) 

# create a class of Model
class Net(Model):
  def __init__(self):
    super(Net, self).__init__()
    self.d1 = Dense(10, activation='relu')
    self.d2 = Dense(3, activation='softmax')

  def call(self, x):
    x = self.d1(x)
    return self.d2(x)

#buat objek model
model = Net()
epochs = 50
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss = 0
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model(xtrain)
        loss = loss_object(cat_ytrain, predictions)

        if epoch%5==0:
            print("%d / 50 -- loss = %.4f" % (epoch, loss.numpy()))
            
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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
classes = np.argmax(model.predict(xtest), axis=1)

#get accuration
accuration = np.sum(classes == ytest)/len(ytest) * 100

print("Test Accuration : " + str(accuration) + '%')
print("Prediction :")
print(classes)
print("Target :")
print(np.asarray(ytest,dtype="int32"))