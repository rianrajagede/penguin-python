from fastai.tabular.all import *
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

y_names = 'species' #  fastai will automatically encode to integer
cont_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
procs = [Normalize] # normalization for each column

# if using TabularPandas splits is mandatory, it's not automatic
splits = RandomSplitter(valid_pct=.3, seed=113)(range_of(datatrain)) 

to = TabularPandas(datatrain, procs = procs,
                   cont_names = cont_names,
                   splits = splits,
                   y_names = y_names) # tell the program that this is a classification task

trainloader = to.dataloaders()

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

# default model for tabular is simple neural network
model = tabular_learner(trainloader, layers=[20], metrics=accuracy)

## TRY THIS if you want to define your own model (PyTorch's default style)
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(4, 20)
#         self.fc2 = nn.Linear(20, 3)
#     # fastai’s tabular dataloaders will always return two tensors (categorical, continuous)
#     def forward(self, cat, cont): 
#         cont = F.relu(self.fc1(cont))
#         cont = self.fc2(cont)
#         return cont 
# net = Net().cuda()
# model = Learner(trainloader, model=net, metrics=accuracy) # pay attention for the class name

#train
model.fit(n_epoch=50, lr=0.01)

"""
SECTION 3 : Testing model
"""

testloader = trainloader.test_dl(datatest)
pred, targs = model.get_preds(dl=testloader)
print(accuracy(pred, targs))
