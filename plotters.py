import matplotlib.pyplot as plt
import numpy as np


def calculate_score(N, M):
    # Calculate V based on the condition provided
    V = (M // 4) + 1 if M % 4 != 0 else M // 4
    # Calculate the score
    score = V * N
    return score


def find_top_architectures(results, top_n=10):
    # Extract N, M, and loss values from results
    N_values = [res["N"] for res in results]
    M_values = [res["M"] for res in results]
    loss_values = np.array([res["loss"] for res in results])  # Convert to NumPy array

    # Find the top N architectures with the lowest unique losses
    top_architectures = []

    for _ in range(top_n):
        # Find the index of the current minimum loss
        min_loss_idx = np.argmin(loss_values)
        min_loss = loss_values[min_loss_idx]

        # Get the corresponding architecture
        best_N = N_values[min_loss_idx]
        best_M = M_values[min_loss_idx]
        best_loss = loss_values[min_loss_idx]
        best_score = calculate_score(best_N, best_M)

        # Append to the list of top architectures
        top_architectures.append(
            {"N": best_N, "M": best_M, "loss": best_loss, "score": best_score}
        )

        # Set the current minimum loss to infinity so it won't be considered again
        loss_values[min_loss_idx] = float("inf")

    return top_architectures


def plotResultsFromGridSearch(results, top_n=10):
    # Extract N, M, and loss values from results
    N_values = [res["N"] for res in results]
    M_values = [res["M"] for res in results]
    loss_values = [res["loss"] for res in results]

    # Get the top N architectures with the lowest losses and their scores
    top_architectures = find_top_architectures(results, top_n=top_n)

    # Print the top N architectures with their scores, formatted in scientific notation
    print(f"Top {top_n} architectures with the lowest losses:")
    for i, arch in enumerate(top_architectures):
        formatted_loss = "{:.3e}".format(
            arch["loss"]
        )  # Format loss in scientific notation
        print(
            f"{i+1}. N = {arch['N']}, M = {arch['M']}, Loss = {formatted_loss}, Score = {arch['score']}"
        )

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

    # Set x and y ticks to be multiples of 5
    plt.xticks(
        np.arange(min(N_values), max(N_values) + 1, 5)
    )  # Multiples of 5 for x-ticks
    plt.yticks(
        np.arange(min(M_values), max(M_values) + 1, 5)
    )  # Multiples of 5 for y-ticks

    # Show the plot
    plt.show()
