# %%
# In this file, we attempt to perform grid-search for the smallest model with the highest accuracy
# We will attempt to model a tanh function with N hidden layers and M neurons per layer
# y = tanh(2x)
# argmin(N,M)argmax(accuracy)
# We limit ourselves to only Linear (Dense) neural networks with RELU activation functions

# First we create a dataset of samples
# These will consist of SAMPLE_COUNT samples long sine waves with random frequency and phase concatenated together

import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Global macros
VERBOSE = False
SAMPLE_COUNT = 100
TRAIN = False

# Refer to generators.py for the code that generates the dataset
# Assuming `generateTanh2xDataset` is provided
from generators import generateTanh2xDataset
from plotters import plotResultsFromGridSearch

# Generate the dataset
x_data, y_data = generateTanh2xDataset(VERBOSE, SAMPLE_COUNT)

# Get x_train, x_test, y_train, y_test via 80:20 split
split = 0.8
splitIndex = int(len(x_data) * split)
x_train = x_data[:splitIndex]
x_test = x_data[splitIndex:]
y_train = y_data[:splitIndex]
y_test = y_data[splitIndex:]

# Convert to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).view(
    -1, 1
)  # Add a second dimension
x_test = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)  # Add a second dimension
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Dataset loader
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=False)

test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)


# Define the grid search model creator
def createModel(N, M):
    layers = []
    # Add the input layer
    layers.append(nn.Linear(1, M))
    layers.append(nn.ReLU())
    # Add the hidden layers
    for i in range(N):
        layers.append(nn.Linear(M, M))
        layers.append(nn.ReLU())
    # Add the output layer
    layers.append(nn.Linear(M, 1))
    return nn.Sequential(*layers)


# Define the training function
def trainModel(model, train_loader, test_loader, epochs=100, learning_rate=0.001):
    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for t in range(epochs):
        for batch_x, batch_y in train_loader:
            # Forward pass: compute prediction using the model
            y_pred = model(batch_x)

            # Reshape batch_y to match y_pred shape
            batch_y = batch_y.view_as(y_pred)

            # Compute loss
            loss = loss_fn(y_pred, batch_y)

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for testing
        test_loss = 0
        for batch_x, batch_y in test_loader:
            y_pred = model(batch_x)
            batch_y = batch_y.view_as(y_pred)  # Reshape to match y_pred

            # Compute loss
            loss = loss_fn(y_pred, batch_y)
            test_loss += loss.item()

    return test_loss


# Perform grid search with valid N and M values
maxM = 30
maxN = 30
bestModel = None
bestLoss = float("inf")
bestN = 0
bestM = 0
# List to store results for plotting later
results = []
counter = 0
if TRAIN:
    # Ensure N and M are >= 1 to avoid invalid model creation
    for N in range(1, maxM + 1):  # N starts from 1
        for M in range(1, maxN + 1):  # M starts from 1
            model = createModel(N, M)
            loss = trainModel(
                model, train_loader, test_loader, epochs=100, learning_rate=0.001
            )

            # Store the architecture and its corresponding loss
            results.append({"N": N, "M": M, "loss": loss})

            # Check if it's the best model
            if loss < bestLoss:
                bestModel = model
                bestLoss = loss
                bestN = N
                bestM = M
            counter += 1
            print(f"{counter} models of {maxM * maxN} trained")

    # After the loop, all architectures and their losses are stored in `results`
    print("Best model has N =", bestN, "and M =", bestM, "with loss", bestLoss)

    # Save results to a file
    with open("grid_search_results.pkl", "wb") as f:
        pickle.dump(results, f)


# Load results from the file
with open("grid_search_results.pkl", "rb") as f:
    results = pickle.load(f)

# Plot the results
plotResultsFromGridSearch(results)

# %%
