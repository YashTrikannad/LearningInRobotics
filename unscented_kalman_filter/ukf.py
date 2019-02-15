import numpy as np


class quaternion_class:

    def convert4dto3d(self, q):
        theta = 2 * np.arccos(q[0])
        rot_vector = (theta / np.sin(theta / 2)) * [q[1:]]
        return rot_vector

    def convert3dto4d(self, w):
        angle = np.linalg.norm(w, axis=0)
        axis = w/(angle+0.00000001)
        quaternion = np.vstack((np.cos(angle / 2), axis*np.sin(angle / 2)))
        return quaternion

    def quaternion_multiplication(self, q0_vec, q1_vec):
        w0, x0, y0, z0 = q0_vec[0], q0_vec[1], q0_vec[2], q0_vec[3]
        w1, x1, y1, z1 = q1_vec[0], q1_vec[1], q1_vec[2], q1_vec[3]
        w = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1])
        x = np.array([w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1])
        y = np.array([w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1])
        z = np.array([w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1])
        result = np.vstack((w, x, y, z))
        return result


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
        x_k = np.hstack((np.expand_dims(statek_1, axis=1), x_k))
        return x_k

    def Xi_propagation(self, x_k, delta_t):
        previous_angular = x_k[-3:]
        angle = np.linalg.norm(previous_angular, axis=0)*delta_t
        axis = previous_angular/(angle+0.00000001)
        quat_delta = np.vstack((np.cos(angle / 2), axis * np.sin(angle / 2)))
        q = self.quaternion_multiplication(x_k[:4, :], quat_delta)
        x_predicted = np.vstack((q, previous_angular))
        return x_predicted




