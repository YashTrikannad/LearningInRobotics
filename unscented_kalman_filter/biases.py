import numpy as np
import matplotlib.pyplot as plt


def find_RPY(vicon_data):

    temp = np.sqrt(vicon_data[0, 0, :] * vicon_data[0, 0, :] + vicon_data[1, 0, :] * vicon_data[1, 0, :])
    x = np.arctan2(vicon_data[2, 1, :], vicon_data[2, 2, :])
    y = np.arctan2(-vicon_data[2, 0, :], temp)
    z = np.arctan2(vicon_data[1, 0, :], vicon_data[0, 0, :])
    return np.vstack((x, y, z))


def find_biases(vicon_data):
    vicon_rots = np.asarray(vicon_data['rots'])
    timestamps = np.asarray(vicon_data['ts'])

    euler_angles = find_RPY(vicon_rots)

    # plt.figure()
    # plt.plot(np.squeeze(timestamps), euler_angles[0, :])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(np.squeeze(timestamps), euler_angles[1, :])
    # plt.show()

    plt.figure()
    plt.plot(np.squeeze(timestamps), euler_angles[1, :])
    plt.show()

    print("Finding Bias")
    # return bias
