import scipy.io as sio
import numpy as np
from unscented_kalman_filter.ukf import Model


# def run(acc, vel, ts):
#
#     # Initialize P_k-1 and Q
#     P = np.eye(6)
#     Q = np.eye(6)
#     R = np.eye(3)
#     q = [1/2, 1/2, 1/2, 1/2]
#     w = [1, 1, 1]
#     x = np.transpose(np.concatenate([q, w]))
#     # Do transformation of sigma point
#     delta_time = (ts[0, 1]-ts[0, 0])
#
#     u = Ukf_base_class()
#
#     u.sigma_w_calculation()
#     u.xi_calculation()
#     Yi = u.xi_propagation(delta_time)
#     xk_, ei = u.apriori_mean_estimate()
#     Pk_, Wi6d_ = u.covariance_yi(x, Yi, ei)
#
#     # Measurement Model
#     zk_, Z_i, vk = u.getMeasurements(Yi, "gyro")
#     Pvv = u.getMeasurementCovariance(zk_, Z_i, R)
#     Pxz = u.CrossCorrelation(Wi6d_, Z_i, zk_)
#     u.KalmanGain(Pxz, Pvv, xk_,vk, Pk_ )

def estimate_rot(data_num):
    # your code goes here
    imu_data = sio.loadmat("imu/imuRaw%d.mat" % data_num)
    imu_measurements = np.asarray(imu_data['vals'])
    linear_acc = imu_measurements[0:3, :]
    angular_vel = imu_measurements[3:, :]
    imu_ts = np.asarray(imu_data['ts'])
    # run(linear_acc, angular_vel, imu_ts)
    ukf = Model()
    ukf.estimate_process_update(imu_ts[0, 1]-imu_ts[0, 0])




estimate_rot(1)
