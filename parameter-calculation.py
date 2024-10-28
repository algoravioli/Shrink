# %%
import numpy as np


def calculate_parameter_count_direct(x_data, y_data, error_desired):
    """
    Calculate the number of parameters required in a neural network to approximate the function
    mapping from x_data to y_data within the desired error.

    Parameters:
    - x_data: Input data array.
    - y_data: Output data array.
    - error_desired: The desired approximation error (epsilon).

    Returns:
    - num_parameters: Estimated number of parameters required.
    - function_complexity_measures: Dictionary containing Lipschitz constant and max second derivative.
    """

    # Ensure the data lengths are the same
    min_length = min(len(x_data), len(y_data))
    x_data = x_data[:min_length]
    y_data = y_data[:min_length]

    # Compute the sampling interval delta_x
    delta_x = np.max(np.abs(np.diff(x_data)))

    # Compute the first derivative
    delta_f = np.diff(y_data) / delta_x

    # Compute the Lipschitz constant L
    L = np.max(np.abs(delta_f))

    # Compute the second derivative using central differences
    second_derivative = (y_data[:-2] - 2 * y_data[1:-1] + y_data[2:]) / (delta_x**2)
    M = np.max(np.abs(second_derivative))

    # Modulus of Continuity omega_f(delta)
    omega_f = np.max(np.abs(delta_f * delta_x))

    # Compute the minimal number of intervals N required
    h = np.sqrt(2 * error_desired / M)
    total_x_range = np.abs(np.max(x_data) - np.min(x_data))
    N = int(np.ceil(total_x_range / h))

    # Compute the number of parameters required in the neural network
    num_parameters = 1.3 * N

    # Prepare the function complexity measures to return
    function_complexity_measures = {
        "Lipschitz_constant": L,
        "Max_second_derivative": M,
        "Modulus_of_continuity": omega_f,
    }

    # Print the results
    print("Function Complexity Measures:")
    print(f"Lipschitz Constant L: {L}")
    print(f"Maximum Second Derivative M: {M}")
    print(f"Modulus of Continuity ω_f(δ): {omega_f}")
    print(f"Desired Error ε: {error_desired}")
    print(f"Computed Interval Length h: {h}")
    print(f"Total X Range: {total_x_range}")
    print(f"Number of Intervals N: {N}")
    print(f"Estimated Number of Parameters: {num_parameters}")

    return num_parameters, function_complexity_measures


# Create some data
x_data = np.linspace(-1, 1, 48000)
y_data = np.tanh(2 * x_data)

# Call the function
num_parameters, function_complexity_measures = calculate_parameter_count_direct(
    x_data, y_data, 0.0000001
)
