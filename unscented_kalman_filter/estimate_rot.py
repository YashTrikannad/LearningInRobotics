import scipy.io as sio
import numpy as np
from unscented_kalman_filter.ukf import Model
from unscented_kalman_filter.correct_RawAcc import correct_acc
import matplotlib.pyplot as plt
from unscented_kalman_filter.biases import find_biases

def estimate_rot(data_num):
    # your code goes here
    imu_data = sio.loadmat("imu/imuRaw%d.mat" % data_num)
    vicon_data = sio.loadmat("vicon/viconRot%d" % data_num)

    # # Find Biases
    find_biases(vicon_data)

    imu_measurements = np.asarray(imu_data['vals'])
    Rawacc = imu_measurements[0:3, :]
    angular_vel = imu_measurements[3:, :]
    imu_ts = np.asarray(imu_data['ts'])

    # Correct Acceleration
    correct_Acc = correct_acc(Rawacc)

    ukf = Model()
    outState6 = np.zeros((6, 5500))

    for i in range(1, 5500):
        ukf.estimate_process_update(imu_ts[0, i+1]-imu_ts[0, i])
        ukf.measurement_update(correct_Acc[:, i])
        outState6[:, i] = np.squeeze(ukf.getState())

    plt.figure()
    plt.plot(np.squeeze(imu_ts[0, :5500]), outState6[1, :])
    plt.show()

    print("Hope For The Best")

estimate_rot(1)