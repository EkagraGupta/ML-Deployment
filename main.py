import torch.nn as nn
from torch.optim import SGD

from model import MyModel
from load_data import load_dataset
from train import train
from test import test
from hyperparameters import N_EPOCHS, LEARNING_RATE, BATCH_SIZE

# Load CIFAR10 dataset
trainloader, testloader, classes = load_dataset(batch_size=BATCH_SIZE)

# Initialize model and respective modules (loss and optimizer)
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = SGD(params=model.parameters(), lr=LEARNING_RATE)

# Train the model and save it under specified path
train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    trainloader=trainloader,
    n_epochs=N_EPOCHS,
)

# Test the model
test(model=model, testloader=testloader, batch_size=BATCH_SIZE, classes=classes)