import scipy.io as sio
import numpy as np


def estimate_rot(data_num):
	# your code goes here
	imu_data = sio.loadmat("imu/imuRaw%d.mat", data_num)
	imu_measurements = np.asarray(imu_data['vals'])
	imu_ts = np.asarray(imu_data['ts'])

	# return roll,pitch,yaw
