import numpy as np

class quaternion_class:

    def convert4dto3d(self, q):
        q = np.clip(q, -1, 1)
        theta = np.expand_dims(2*np.arccos(q[0]), axis=0)
        theta = np.repeat(theta, 3, axis=0)
        rot_vector = (theta / (np.sin((theta+0.00001) / 2))) * q[1:, :]
        return rot_vector

    def convert3dto4d(self, w):
        angle = np.linalg.norm(w, axis=0)
        axis = w/(angle+0.00000001)
        quaternion = np.vstack((np.cos(angle / 2), axis*np.sin(angle / 2)))
        return quaternion

    def convert3dto4done(self, w):
        angle = np.linalg.norm(w, axis=0)
        axis = w/(angle+0.00000001)
        quaternion = np.hstack((np.array(np.cos(angle / 2)), axis*np.sin(angle / 2)))
        return quaternion

    def quaternion_multiplication(self, q0_vec, q1_vec):
        w0, x0, y0, z0 = q0_vec[0], q0_vec[1], q0_vec[2], q0_vec[3]
        w1, x1, y1, z1 = q1_vec[0], q1_vec[1], q1_vec[2], q1_vec[3]
        w = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1])
        x = np.array([w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1])
        y = np.array([w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1])
        z = np.array([w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1])
        result = np.vstack((w, x, y, z))
        result = result/(np.linalg.norm(result, axis=0))
        return result

    def quaternion_inverse(self, q):
        q_inverse = np.array([q[0], -q[1], -q[2], -q[3]])
        mag = np.linalg.norm(q)
        return q_inverse/mag

class ukf(quaternion_class):

    def initialize_P_matrix(self, n):
        return np.eye(n)

    def sigma_w_calculation(self, p, q):
        root = np.sqrt(2)*(np.linalg.cholesky(p+q))
        root = np.hstack((root, -root))
        return root

    def Xi_calculation(self, root, statek_1):
        q = self.convert3dto4d(root[0:3, :])
        q_k = self.quaternion_multiplication(statek_1[0:4], q)
        statek_1_repeated = np.transpose(np.repeat(np.reshape(statek_1, (np.size(statek_1[0]), -1)), 12, axis=0))
        w_k = root[3:, ]+statek_1_repeated[4:, ]
        x_k = np.vstack((q_k, w_k))
        return x_k

    def Xi_propagation(self, x_k, delta_t):
        previous_angular = x_k[-3:]
        angle = np.linalg.norm(previous_angular, axis=0)*delta_t
        axis = previous_angular/(angle+0.00000001)
        quat_delta = np.vstack((np.cos(angle / 2), axis * np.sin(angle / 2)))
        q = self.quaternion_multiplication(x_k[:4, :], quat_delta)
        x_predicted = np.vstack((q, previous_angular))
        return x_predicted

    def apriori_mean_estimate(self, Yi, xk_1):
        omega_i = Yi[4:, :]
        q_i = Yi[:4, :]

        x_cap_k = np.zeros((7, 1))
        x_cap_k[4:] = np.expand_dims(np.sum(omega_i, axis=1), axis=1) / 12

        q_t = xk_1[:4]
        e_i = np.zeros((4, 12))
        for t in range(5):
            for i in range(12):
                e_i[:, i] = self.quaternion_multiplication(q_i[:, i], self.quaternion_inverse(q_t)).squeeze()
            e_i_ = self.convert4dto3d(e_i)
            e_vect = np.sum(e_i_, axis=1) / 12
            e = self.convert3dto4done(e_vect)
            q_t = self.quaternion_multiplication(e, q_t)

        x_cap_k[:4] = q_t
        return x_cap_k, e_i_

    def covariance_Yi(self, xk_, Yi, rw):
        xk__repeat = np.repeat(np.expand_dims(xk_, axis=1), 12, axis=1)
        Wi7d = Yi - xk__repeat
        Wi6d = np.vstack((self.convert4dto3d(Wi7d[0:4, :]), Wi7d[4:, ]))
        Wi6d[:3, :] = rw
        Pk_ = Wi6d.dot(np.transpose(Wi6d))/12
        return Pk_, Wi6d

    def getMeasurements(self, Yi, sensor = "gyro"):

        if sensor == "gyro":
            Z_i = Yi[4:, :]
        else:
            return

        zk_ = np.sum(Z_i, axis=1)/12
        innovation = Z_i - np.expand_dims(zk_, axis=1)

        return zk_, Z_i, innovation

    def getMeasurementCovariance(self, zk_, Zi, R):
        Wz = Zi - np.expand_dims(zk_, axis=1)
        Pzz = np.dot(Wz, np.transpose(Wz))
        Pvv = Pzz + R
        return Pvv

    def CrossCorrelation(self, Wi_, Zi, zk_):
        return np.dot(Wi_, np.transpose(Zi-np.expand_dims(zk_,axis=1)))

    def KalmanGain(self, Pxz, Pvv, xk_, vk, Pk_):
        Kk = np.dot(Pxz, np.linalg.inv(Pvv))
        x_cap = xk_ + np.dot(Kk, vk)
        Pk = Pk_ - np.dot(Kk, Pvv, np.transpose(Kk))








    


