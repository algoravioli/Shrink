import matplotlib.pyplot as plt
import numpy as np


def plotResultsFromGridSearch(results):
    # Extract N, M, and loss values from results
    N_values = [res["N"] for res in results]
    M_values = [res["M"] for res in results]
    loss_values = [res["loss"] for res in results]

    # Take the log of loss values to enhance contrast
    log_loss_values = np.log(loss_values)

    # Plotting the log-transformed loss values
    plt.figure(figsize=(10, 6))
    plt.scatter(N_values, M_values, c=log_loss_values, cmap="viridis", marker="o")
    cbar = plt.colorbar(label="Log(Loss)")  # Update colorbar label to Log(Loss)

    # Set x and y axis labels
    plt.xlabel("Number of Hidden Layers (N)")
    plt.ylabel("Number of Neurons per Layer (M)")
    plt.title("Logarithmic Loss Values for Different Architectures")

    # Set x and y ticks to be integers based on the range of N and M values
    plt.xticks(np.arange(min(N_values), max(N_values) + 1, 1))  # Integer x-ticks
    plt.yticks(np.arange(min(M_values), max(M_values) + 1, 1))  # Integer y-ticks

    # Show the plot
    plt.show()
