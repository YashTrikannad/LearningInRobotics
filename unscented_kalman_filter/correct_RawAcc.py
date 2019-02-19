# Function to Correct Raw Acceleration Values
import numpy as np


def correct_acc(raw_data):

    vref = 3.3
    sensitivity = 0.330
    scale_factor = vref/(1023*sensitivity)

    bias = np.mean(raw_data[:, :200], axis=1)
    bias[2] = bias[2] - 1 / (vref / 1023 / sensitivity)
    corrected_data = (raw_data - np.expand_dims(bias, axis=1))*scale_factor

    return corrected_data

