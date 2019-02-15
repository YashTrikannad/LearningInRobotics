import numpy as np



def normalize(vector):
    return vector/(np.sqrt(np.sum(np.power(vector, 2))))


def convert4dto3d(q):
    theta = 2*np.arccos(q[0])
    rot_vector = (theta/np.sin(theta/2))*[q[1:]]
    return rot_vector


def convert3dto4d(w):
    angle = np.linalg.norm(w)
    axis = w/angle
    quaternion = np.array([np.cos(angle/2), axis*(np.sin(angle/2))])
    quaternion = normalize(quaternion)
    return quaternion
