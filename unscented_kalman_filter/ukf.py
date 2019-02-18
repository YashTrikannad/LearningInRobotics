import numpy as np
from unscented_kalman_filter.quaternion import QuaternionClass


class Ukf_base_class(QuaternionClass):

    def __init__(self):
        self.x_cap_k_1 = np.array([1/2, 1/2, 1/2, 1/2, 1, 1, 1])  # State Vector at k-1
        self.x_cap_k = np.zeros_like(self.x_cap_k_1)
        self.P = np.eye(6)  # Covariance Matrix for State Update
        self.Q = np.eye(6)  # Noise Matrix for State Update
        self.R = np.eye(3)  # Noise Matrix for Measurement Update
        self.Pk_ = np.zeros_like(self.P)    # Covariance Matrix for Yi
        self.Wi = np.array([])    # Columns are Process Noise vectors of Sigma Points
        self.Wi6d = np.array([])  # For Set Yi
        self.Xi = np.array([])    # Sigma Points
        self.Yi = np.array([])    # Transformed Points
        self.Zi = np.array([])    # Set Zi for Measurement
        self.zk_ = np.array([])
        self.e_i = np.array([])
        self.innovation = np.array([])


class process(Ukf_base_class):

    def sigma_w_calculation(self):
        root = np.sqrt(12)*(np.linalg.cholesky(self.P + self.Q))
        self.Wi = np.hstack((root, -root))

    def xi_calculation(self):
        q = self.convert3dto4d(self.Wi[0:3, :])
        q_k = self.quaternion_multiplication(self.x_cap_k_1[0:4], q)
        x_cap_k_1_repeated = np.transpose(np.repeat(np.reshape(self.x_cap_k_1, (7, -1)), 12, axis=0))
        w_k = self.Wi[3:, :]+x_cap_k_1_repeated[4:, :]
        self.Xi = np.vstack((q_k, w_k))

    def xi_propagation(self, delta_t):
        previous_angular = self.Xi[-3:]
        angle = np.linalg.norm(previous_angular, axis=0)*delta_t
        axis = np.divide(previous_angular, angle+0.00000001)
        q_delta = np.vstack((np.cos(angle / 2), axis * np.sin(angle / 2)))
        q = self.quaternion_multiplication(self.Xi[:4, :], q_delta)
        self.Yi = np.vstack((q, previous_angular))

    def apriori_mean_estimate(self):
        omega_i = self.Yi[4:, :]
        q_i = self.Yi[:4, :]

        self.x_cap_k = np.zeros((7, 1))
        self.x_cap_k[4:] = np.expand_dims(np.sum(omega_i, axis=1), axis=1) / 12

        q_t = self.x_cap_k_1[:4]
        self.e_i = np.zeros((4, 12))

        for t in range(5):
            for i in range(12):
                self.e_i[:, i] = self.quaternion_multiplication(q_i[:, i], self.quaternion_inverse(q_t)).squeeze()
            e_i_ = self.convert4dto3d(self.e_i)
            e_vect = np.sum(e_i_, axis=1) / 12
            e = self.convert3dto4done(e_vect)
            q_t = self.quaternion_multiplication(e, q_t)

        self.x_cap_k[:4] = q_t

    def covariance_yi(self):
        xk__repeat = np.repeat(np.expand_dims(self.x_cap_k, axis=1), 12, axis=1)
        Wi7d = self.Yi - xk__repeat
        self.Wi6d = np.vstack((self.convert4dto3d(Wi7d[0:4, :]), Wi7d[4:, ]))
        self.Wi6d[:3, :] = self.e_i
        self.Pk_ = self.Wi6d.dot(np.transpose(self.Wi6d))/12


class MeasurementModel(Ukf_base_class):

    def getMeasurements(self, sensor="gyro"):

        if sensor == "gyro":
            Z_i = self.Yi[4:, :]
        else:
            return

        zk_ = np.sum(Z_i, axis=1)/12
        self.innovation = Z_i - np.expand_dims(zk_, axis=1)

        return zk_, Z_i

    def getMeasurementCovariance(self):
        Wz = self.Zi - np.expand_dims(self.zk_, axis=1)
        Pzz = np.dot(Wz, np.transpose(Wz))
        Pvv = Pzz + self.R
        return Pvv

    def CrossCorrelation(self):
        return np.dot(self.Wi6d, np.transpose(self.Zi-np.expand_dims(self.zk_, axis=1)))

    def KalmanGain(self, Pxz, Pvv, xk_, vk, Pk_):
        Kk = np.dot(Pxz, np.linalg.inv(Pvv))
        x_cap = xk_ + np.dot(Kk, vk)
        Pk = Pk_ - np.dot(Kk, Pvv, np.transpose(Kk))


class Model(MeasurementModel, process):

    def estimate_process_update(self, delta_time):
        self.sigma_w_calculation()
        self.xi_calculation()
        self.xi_propagation(delta_time)
        self.apriori_mean_estimate()
        self.covariance_yi()

    def measurement_update(self):
        self.getMeasurements("gyro")
        self.getMeasurementCovariance()
        self.CrossCorrelation()

        # Measurement Model
        # zk_, Z_i, vk = u.getMeasurements(Yi, "gyro")
        # Pvv = u.getMeasurementCovariance(zk_, Z_i, R)
        # Pxz = u.CrossCorrelation(Wi6d_, Z_i, zk_)
        # u.KalmanGain(Pxz, Pvv, xk_, vk, Pk_)