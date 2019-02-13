import scipy.io as scio
import numpy as np
imu_data = scio.loadmat("imu/imuRaw1.mat")

imu_measurements = np.asarray(imu_data['vals'])



print("In Progress")
