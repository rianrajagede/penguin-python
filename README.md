# Simple Sample Codes for Palmer Penguin Dataset Classification

This is a collection of simple and easy-to-read programs for [Palmer Penguin](https://github.com/allisonhorst/palmerpenguins) classification (for teaching purposes).  I use several frameworks to classify this data, so the reader can compare the difference between one and another for the same usage. 

This repository is an improvement from the previous [iris-python repository](https://github.com/rianrajagede/iris-python). Compared to Iris dataset, Palmer Penguin dataset is little bit challenging, because there are some preprocessing steps that necessary for this data (remove missing values, standardization, etc.).

![Penguin Team](lter_penguins.png)

## Each program contains:

1. Load the [cleaned data](https://github.com/rianrajagede/penguin-python/tree/master/Datasets)
2. Preprocess (convert to numeric, standardize)
3. Train a model
4. Test

## Required Packages:

Please install `requirements.txt` file in each folder

---

## Keras (using Tensorflow backend)
- [Simple Neural Net using Keras][penguin_keras] (Multilayer perceptron model, with one hidden layer)

## Python & Scikit-Learn
- [Simple Neural Net without external library][penguin_plain] (Multilayer perceptron model, with one hidden layer) 
- [Simple Neural Net without external library][penguin_plain_2] (No-hidden layer model)
- [Simple Neural Net using Scikit-learn MLPClassifier][penguin_scikit] (Multilayer perceptron model, with one hidden layer)

## PyTorch
- [Simple Neural Net using PyTorch][penguin_pytorch] (Multilayer perceptron model, with one hidden layer)
- [Simple Neural Net using PyTorch Lightning][penguin_pytorch_lightning] (Multilayer perceptron model, with one hidden layer)

## Tensorflow
- [Simple Neural Net using Tensorflow][penguin_tf] (Multilayer perceptron model, with one hidden layer)
- [Simple Neural Net using Tensorflow version 2.x][penguin_tf_2] (Multilayer perceptron model, with one hidden layer)

[penguin_keras]:https://github.com/rianrajagede/penguin-python/blob/master/Keras/penguin_keras.py
[penguin_scikit]:https://github.com/rianrajagede/penguin-python/blob/master/Python/penguin_scikit_mlp.py
[penguin_scikit_rf]:https://github.com/rianrajagede/penguin-python/blob/master/Python/penguin_scikit_rf.py
[penguin_plain]:https://github.com/rianrajagede/penguin-python/blob/master/Python/penguin_plain_mlp.py
[penguin_plain_2]:https://github.com/rianrajagede/penguin-python/blob/master/Python/penguin_plain_slp.py
[penguin_pytorch]:https://github.com/rianrajagede/penguin-python/blob/master/Pytorch/penguin_pytorch.py
[penguin_pytorch_lightning]:https://github.com/rianrajagede/penguin-python/blob/master/Pytorch/penguin_pytorch_lightning.py
[penguin_tf]:https://github.com/rianrajagede/penguin-python/blob/master/Tensorflow/penguin_tf_v1.py
[penguin_tf_2]:https://github.com/rianrajagede/penguin-python/blob/master/Tensorflow/penguin_tf_v2.py