import scipy.io as sio
import numpy as np
from unscented_kalman_filter.ukf import ukf
import helper_functions

def run(acc, vel, ts):

	# Initialize P_k-1 and Q
	P = np.eye(6)
	Q = np.eye(6)

	q = [1/2, 1/2, 1/2, 1/2]
	w = [1, 1, 1]
	x = np.transpose(np.concatenate([q, w]))
	# Do transformation of sigma point
	u = ukf()

	Wi = u.sigma_w_calculation(P, Q)
	u.Xi_calculation(Wi, x)

def estimate_rot(data_num):
	# your code goes here
	imu_data = sio.loadmat("imu/imuRaw%d.mat" % data_num)
	imu_measurements = np.asarray(imu_data['vals'])
	linear_acc = imu_measurements[0:3, :]
	angular_vel = imu_measurements[3:, :]
	imu_ts = np.asarray(imu_data['ts'])
	run(linear_acc, angular_vel, imu_ts)


estimate_rot(1)
