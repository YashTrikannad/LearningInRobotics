import numpy as np
from histogram_filter import HistogramFilter


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    # Test your code here

    init_belief_grid = np.ones_like(cmap)
    numel = np.prod(init_belief_grid.shape)
    init_belief_grid = init_belief_grid / numel

    bayes = HistogramFilter()

    belief = init_belief_grid
    mle_arr = []

    for i in range(observations.size):
        belief, mle = bayes.histogram_filter(cmap, belief, actions[i], observations[i])
        mle_arr.append(mle)

    # MAXIMUM LIKELIHOOD BELIEF STATES
    mle_arr = np.asarray(mle_arr)

