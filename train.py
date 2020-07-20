import torch as t
from data import ChallengeDataset
from model import ResNet
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

# some hyper parameters
batch_size = 96
epochs = 50
learn_rate = 0.01
momentum = 0.9

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('data.csv', sep=';')
train_data, test_data = train_test_split(data)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_data = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size)
test_data = t.utils.data.DataLoader(ChallengeDataset(test_data, 'val'), batch_size)

# create an instance of our ResNet model
model = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.BCELoss()
optimizer = t.optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)
#optimizer = t.optim.Adam(model.parameters(), lr=learn_rate)
trainer = Trainer(model, criterion, optimizer, train_data, test_data, cuda=True, early_stopping_patience=-1)

# go, go, go... call fit on trainer
res = trainer.fit(epochs)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
