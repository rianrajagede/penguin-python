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
learning rate = 0.001
epoch = 50
"""
import os
# import tensorflow # if using tensoflow 1.x
import tensorflow.compat.v1 as tf # if using tensoflow 2.x
tf.disable_v2_behavior()  # if using tensoflow 2.x


# tensorflow configuration
cwd = os.path.abspath(os.path.dirname(__file__)) # path for saving model
saver_path = os.path.abspath(os.path.join(cwd, 'models/model_sess.ckpt'))

# tensorflow model
input = tf.placeholder(tf.float32, [None, 4])
label = tf.placeholder(tf.float32, [None])
onehot_label = tf.one_hot(tf.cast(label, tf.int32), 3)
hidden = tf.layers.dense(input, 10, tf.nn.relu, name="hidden")
output = tf.layers.dense(hidden, 3, tf.nn.relu, name="output")
soft_output = tf.nn.softmax(output)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# optimization
loss = -tf.reduce_sum(onehot_label * tf.log(soft_output))
optimizer = tf.train.GradientDescentOptimizer(0.01)
is_correct = tf.equal(tf.argmax(soft_output, 1), tf.argmax(onehot_label,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(init)

    # start training
    for e in range(50):
        if(e%5==0):
            print(e,"/ 50 - Loss:", sess.run(loss, feed_dict={input:xtrain, label:ytrain}))
        sess.run(train_step, feed_dict={input:xtrain, label:ytrain})

    # save model
    saver.save(sess, saver_path)
    print("Train accuracy",sess.run(accuracy, feed_dict={input:xtrain, label:ytrain}))

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

# get the model then test
with tf.Session() as sess:
    saver.restore(sess, saver_path)
    print("Test accuracy",sess.run(accuracy, feed_dict={input:xtest, label:ytest}))    