import scipy.io as sio
import numpy as np
from unscented_kalman_filter.ukf import Model
import unscented_kalman_filter.biases as bias


def estimate_rot(data_num):
    # your code goes here
    imu_data = sio.loadmat("imu/imuRaw%d.mat" % data_num)
    vicon_data = sio.loadmat("vicon/viconRot%d" % data_num)

    # Find Biases
    bias.find_biases(vicon_data)

    imu_measurements = np.asarray(imu_data['vals'])
    linear_acc = imu_measurements[0:3, :]
    angular_vel = imu_measurements[3:, :]
    imu_ts = np.asarray(imu_data['ts'])

    ukf = Model()
    outState = np.zeros((np.size(linear_acc[1]), 6))

    for i in range(1, 500):
        ukf.estimate_process_update(imu_ts[0, i]-imu_ts[0, i-1])
        ukf.measurement_update(linear_acc[:, i])
        outState[i, :] = ukf.getState()

estimate_rot(1)
