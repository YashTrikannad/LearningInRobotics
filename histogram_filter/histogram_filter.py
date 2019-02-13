import numpy as np
from scipy import signal


class HistogramFilter(object):

    def action_model(self, prior, last_action):

        prior = np.asarray(prior, dtype=np.float)
        if last_action[0] == 1:
            kernel = np.array([0, 0.1, 0.9])
            kernel = np.expand_dims(kernel, axis=0)
            state_evolution = signal.convolve2d(prior, kernel, mode='full')
            state_evolution[:, -2] = state_evolution[:, -1] + state_evolution[:, -2]
            state_evolution = state_evolution[:, 1:-1]

        elif last_action[0] == -1:
            kernel = np.array([0.9, 0.1, 0])
            kernel = np.expand_dims(kernel, axis=0)
            state_evolution = signal.convolve2d(prior, kernel, mode='full')
            state_evolution[:, 1] = state_evolution[:, 0] + state_evolution[:, 1]
            state_evolution = state_evolution[:, 1:-1]
        elif last_action[1] == -1:
            kernel = np.array([[0, 0.1, 0.9]]).transpose()
            state_evolution = signal.convolve2d(prior, kernel, mode='full')
            state_evolution[-2, :] = state_evolution[-1, :] + state_evolution[-2, :]
            state_evolution = state_evolution[1:-1, :]
        elif last_action[1] == 1:
            kernel = np.array([[0.9, 0.1, 0]]).transpose()
            state_evolution = signal.convolve2d(prior, kernel, mode='full')
            state_evolution[1, :] = state_evolution[0, :] + state_evolution[1, :]
            state_evolution = state_evolution[1:-1, :]
        else:
            state_evolution = prior

        return state_evolution

    def sensor_model(self, last_measurement, measurement_map, prediction_belief):
        measurement_map = np.asarray(measurement_map, dtype=np.float)
        if last_measurement == 1:
            measurement_map[measurement_map == 1] = 0.9
            measurement_map[measurement_map == 0] = 0.1
        else:
            measurement_map[measurement_map == 1] = 0.1
            measurement_map[measurement_map == 0] = 0.9
        measurement_belief = np.multiply(measurement_map, prediction_belief)
        normalized_belief = measurement_belief / np.sum(measurement_belief)
        return normalized_belief

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and
        corresponding observations. True belief is also given to compare your filters results with actual results.
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3

        ### Your Algorithm goes Below.'''

        # Action Model
        belief = self.action_model(belief, action)

        # Observation Model
        belief = self.sensor_model(observation, cmap, belief)

        yx_location = np.unravel_index(np.argmax(belief), belief.shape)

        mle = np.zeros_like(yx_location)
        mle[0] = yx_location[1]
        mle[1] = (np.size(cmap, axis=1)-1) - yx_location[0]

        return belief, mle

