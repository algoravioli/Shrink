import numpy as np
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generateSineWave(frequency, phase, length_in_samples, sampling_frequency):
    # Generate a time vector with the correct length and sampling rate
    t = np.arange(length_in_samples) / sampling_frequency

    # Calculate the sine wave based on frequency, phase, and time vector
    output = np.sin(2 * np.pi * frequency * t + phase)

    return output


def generateTanh2xDataset(VERBOSE, SAMPLE_COUNT):
    # We generate an array of 10 frequencies between 20 and 10000 Hz
    # The sampling frequency is 48000 Hz
    arrayOfFrequencies = np.random.uniform(20, 10000, 10)
    arrayOfPhases = np.random.uniform(0, 2 * np.pi, 10)
    if VERBOSE:
        print("These are the frequencies we will use to generate the dataset:")
        for frequency in arrayOfFrequencies:
            print(str(frequency))

    # Generate the dataset
    x_data = np.array([])
    for frequency in arrayOfFrequencies:
        # Get the index of the current frequency
        phaseIndex = np.where(arrayOfFrequencies == frequency)[0][0]
        currentPhase = arrayOfPhases[phaseIndex]

        # Generate sine wave data using the correct phase
        currentFrequencyData = generateSineWave(
            frequency, currentPhase, SAMPLE_COUNT, 48000
        )

        # Concatenate the generated data
        x_data = np.concatenate((x_data, currentFrequencyData))

    # Get the y-data
    y_data = np.tanh(2 * x_data)

    if VERBOSE:
        print("Generated datasets (input,output) with shape:")
        print(x_data.shape, y_data.shape)
    return x_data, y_data
