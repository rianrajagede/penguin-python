"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
penguin-clean-train.csv = datasets for training purpose, 70% from the original data
penguin-clean-test.csv  = datasets for testing purpose, 30% from the original data
"""

import csv
import random
import math
random.seed(113)

# Load dataset
with open('../Datasets/penguins-clean-train.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None) # skip header
    datatrain = list(csvreader)

# Change string value to numeric
for row in datatrain:
    if row[0]=="Adelie":
        row[0] = 0
    elif row[0]=="Gentoo":
        row[0] = 1
    else:
        row[0] = 2

    row[1:] = map(float, row[1:])

# Split x and y (feature and target)
train_X = [data[1:] for data in datatrain]
train_y = [data[0] for data in datatrain]

# Min-max Scaling
# palmer-penguin dataset has varying scales
feat_len = len(train_X[0])
data_len = len(train_X)
mnm = [0]*feat_len
mxm = [0]*feat_len
for f in range(feat_len):
    mnm[f] = train_X[0][f]
    mxm[f] = train_X[0][f]
    for d in range(data_len):
        mnm[f] = min(mnm[f], train_X[d][f])
        mxm[f] = max(mxm[f], train_X[d][f])
    
    for d in range(data_len):
        train_X[d][f] = (train_X[d][f] - mnm[f]) / (mxm[f] - mnm[f])

"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature from Palmer Penguin dataset
hidden layer : 3 neuron, activation using sigmoid
output layer : 3 neuron, represents the number of species
optimizer = gradient descent
loss function = Square ROot Error
learning rate = 0.005
epoch = 400
best result = 96.67%
"""

def matrix_mul_bias(A, B, bias): # Matrix multiplication (for Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def vec_mat_bias(A, B, bias): # Vector (A) x matrix (B) multiplication
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C


def mat_vec(A, B): # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C

def sigmoid(A, deriv=False):
    if deriv: # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Define parameter
learening = 0.01
epoch = 50
layers = [4, 10, 3] # number of neuron each layer

# Initiate weight and bias with 0 value
weight = [[0 for j in range(layers[1])] for i in range(layers[0])]
weight_2 = [[0 for j in range(layers[2])] for i in range(layers[1])]
bias = [0 for i in range(layers[1])]
bias_2 = [0 for i in range(layers[2])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(layers[0]):
    for j in range(layers[1]):
        weight[i][j] = 2 * random.random() - 1

for i in range(layers[1]):
    for j in range(layers[2]):
        weight_2[i][j] = 2 * random.random() - 1


for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X): # Update for each data; SGD
        
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(layers[2]):
            eror +=  (target[i] - X_2[i]) ** 2 
        cost_total += eror * 1 / layers[2]

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(layers[2]):
            delta_2.append(-1 * 2. / layers[2] * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))

        for i in range(layers[1]):
            for j in range(layers[2]):
                weight_2[i][j] -= learening * (delta_2[j] * X_1[i])
                bias_2[j] -= learening * delta_2[j]
        
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(layers[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
        
        for i in range(layers[0]):
            for j in range(layers[1]):
                weight[i][j] -=  learening * (delta_1[j] * x[i])
                bias[j] -= learening * delta_1[j]
    
    cost_total /= len(train_X)
    if(e % 10 == 0):
        print(e,"/ 50 - Loss:",cost_total) 

"""
SECTION 3 : Testing Model
"""

# Load dataset
with open('../Datasets/penguins-clean-test.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None) # skip header
    datatest = list(csvreader)

# Change string value to numeric
for row in datatest:
    if row[0]=="Adelie":
        row[0] = 0
    elif row[0]=="Gentoo":
        row[0] = 1
    else:
        row[0] = 2

    row[1:] = map(float, row[1:])

# Split x and y (feature and target)
test_X = [data[1:] for data in datatest]
test_y = [data[0] for data in datatest]

# Min-max Scaling
feat_len = len(test_X[0])
data_len = len(test_X)
for f in range(feat_len):
    for d in range(data_len):
        test_X[d][f] = (test_X[d][f] - mnm[f]) / (mxm[f] - mnm[f])

res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

# Get prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])

# Print prediction
print(preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print(acc / len(preds) * 100, "%")