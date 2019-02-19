import numpy as np


class QuaternionClass:

    def convert4dto3d(self, q):
        q = np.clip(q, -1, 1)
        theta = np.expand_dims(2*np.arccos(q[0]), axis=0)
        theta = np.repeat(theta, 3, axis=0)
        rot_vector = (theta / (np.sin((theta+0.00001) / 2))) * q[1:, :]
        return rot_vector

    def convert3dto4d(self, rot_v):
        angle = np.linalg.norm(rot_v, axis=0)
        axis = rot_v / (angle + 0.00000001)
        quaternion = np.vstack((np.cos(angle / 2), axis*np.sin(angle / 2)))
        return quaternion

    def convert3dto4done(self, rot_v):
        angle = np.linalg.norm(rot_v, axis=0)
        axis = rot_v / (angle + 0.00000001)
        quaternion = np.hstack((np.array(np.cos(angle / 2)), axis*np.sin(angle / 2)))
        return quaternion

    def quaternion_multiplication(self, q0, q1):
        w0, x0, y0, z0 = q0[0], q0[1], q0[2], q0[3]
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1])
        x = np.array([w0 * x1 + x0 * w1 - y0 * z1 + z0 * y1])
        y = np.array([w0 * y1 + x0 * z1 + y0 * w1 - z0 * x1])
        z = np.array([w0 * z1 - x0 * y1 + y0 * x1 + z0 * w1])
        mul = np.vstack((w, x, y, z))
        mul = mul/(np.linalg.norm(mul, axis=0))
        return mul

    def quaternion_inverse(self, q):
        q_inverse = np.array([q[0], -q[1], -q[2], -q[3]])
        mag = np.linalg.norm(q)
        return q_inverse/mag


