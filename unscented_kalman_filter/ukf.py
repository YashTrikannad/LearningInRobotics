import numpy as np
from unscented_kalman_filter.quaternion import QuaternionClass


class Ukf_base_class(QuaternionClass):

    def __init__(self):
        self.x_cap_k_1 = np.array([1/2, 1/2, 1/2, 1/2, 1, 1, 1])  # State Vector at k-1
        self.x_cap_k = np.zeros_like(self.x_cap_k_1)
        self.x_cap_k_next = np.zeros_like(self.x_cap_k_1)
        self.P = np.eye(6)  # Covariance Matrix for State Update
        self.Q = np.eye(6)  # Noise Matrix for State Update
        self.R = np.eye(3)  # Noise Matrix for Measurement Update
        self.Pk_ = np.zeros_like(self.P)    # Covariance Matrix for Yi
        self.Pzz = np.zeros((3, 3))
        self.Pvv = np.zeros((3, 3))
        self.Pxz = np.array([])
        self.Wi = np.array([])    # Columns are Process Noise vectors of Sigma Points
        self.Wi6d = np.zeros((6, 12))  # For Set Yi
        self.Xi = np.zeros((7, 12))    # Sigma Points
        self.Yi = np.zeros((7, 12))    # Transformed Points
        self.Zi = np.zeros((3, 12))    # Set Zi for Measurement
        self.zk_ = np.array([])
        self.e_i = np.array([])
        self.vk = np.array([])
        self.Kk = np.array([])
        self.g = np.array([[0], [0], [-9.8]])
        self.g_ = np.array([])


class process(Ukf_base_class):

    def sigma_w_calculation(self):
        root = np.sqrt(12)*(np.linalg.cholesky(self.P + self.Q))
        self.Wi = np.hstack((root, -root))

    def xi_calculation(self):
        q = self.convert3dto4d(self.Wi[0:3, :])
        q_k = self.quaternion_multiplication(self.x_cap_k_1[0:4], q)
        # x_cap_k_1_repeated = np.transpose(np.repeat(np.reshape(self.x_cap_k_1[4:, ], (7, -1)), 12, axis=0))
        w_k = self.Wi[3:, :] + np.repeat(np.reshape(self.x_cap_k_1[4:, ], (3, -1)), 12, axis=1)
        self.Xi = np.vstack((q_k, w_k))

    def xi_propagation(self, delta_t):
        previous_angular = self.Xi[-3:]
        alpha_delta = np.linalg.norm(previous_angular, axis=0)*delta_t
        e_delta = np.divide(previous_angular, alpha_delta+0.00000001)
        # e_delta = e_delta/(np.linalg.norm(e_delta, axis=0))
        q_delta = np.vstack((np.cos(alpha_delta / 2), e_delta * np.sin(alpha_delta / 2)))
        q = self.quaternion_multiplication(self.Xi[:4, :], q_delta)
        self.Yi = np.vstack((q, previous_angular))

    def apriori_mean_estimate(self):
        omega_i = self.Yi[4:, :]
        q_i = self.Yi[:4, :]

        self.x_cap_k[4:] = np.sum(omega_i, axis=1) / 12

        q_t = self.x_cap_k_1[:4]
        self.e_i = np.zeros((4, 12))

        for t in range(10):
            for i in range(12):
                self.e_i[:, i] = self.quaternion_multiplication(q_i[:, i], self.quaternion_inverse(q_t)).squeeze()
            e_i_ = self.convert4dto3d(self.e_i)
            e_vect = np.sum(e_i_, axis=1) / 12
            e = self.convert3dto4done(e_vect)
            q_t = self.quaternion_multiplication(e, q_t)

        self.x_cap_k[:4] = np.squeeze(q_t)

    def covariance_yi(self):
        xk__omega = np.repeat(np.reshape(self.x_cap_k[4:], (3, -1)), 12, axis=1)
        self.Wi6d[3:, :] = self.Yi[4:, :] - xk__omega
        self.Wi6d[:3, :] = self.convert4dto3d(self.e_i)
        self.Pk_ = self.Wi6d.dot(np.transpose(self.Wi6d))/12


class MeasurementModel(Ukf_base_class):

    def getMeasurements(self, zk, sensor="gyro"):
        if sensor == "gyro":
            self.Zi = self.Yi[4:, :]
        elif sensor == "acc":
            self.g_ = self.quaternion_multiplication(self.quaternion_multiplication(self.x_cap_k[:4, ], self.g),
                                                     np.linalg.inv(self.x_cap_k[:4, ]))
            self.Zi = self.convert4dto3d(self.g_)

        self.zk_ = np.sum(self.Zi, axis=1) / 12
        self.vk = zk - self.zk_

    def getMeasurementCovariance(self):
        Wz = self.Zi - np.expand_dims(self.zk_, axis=1)
        self.Pzz = np.dot(Wz, np.transpose(Wz))/12
        self.Pvv = self.Pzz + self.R

    def CrossCorrelation(self):
        self.Pxz = np.dot(self.Wi6d, np.transpose(self.Zi-np.expand_dims(self.zk_, axis=1)))

    def calculateKalmanGain(self):
        self.Kk = np.dot(self.Pxz, np.linalg.inv(self.Pvv))

    def kalman_update(self):
        update = np.expand_dims(np.dot(self.Kk, self.vk), axis=1)
        self.x_cap_k_next[:4] = np.squeeze(self.quaternion_multiplication(self.x_cap_k[:4], np.squeeze(self.convert3dto4d(update[:3]))))
        self.x_cap_k_next[4:] = self.x_cap_k[4:] + np.squeeze(update[3:])


class Model(MeasurementModel, process):

    def estimate_process_update(self, delta_time):
        self.sigma_w_calculation()
        self.xi_calculation()
        self.xi_propagation(delta_time)
        self.apriori_mean_estimate()
        self.covariance_yi()

    def measurement_update(self, sensor_reading):
        self.getMeasurements(sensor_reading, "acc")
        self.getMeasurementCovariance()
        self.CrossCorrelation()
        self.calculateKalmanGain()
        self.kalman_update()

    def getState(self):
        return self.x_cap_k_next
