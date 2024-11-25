'''
==============================================
Using the Unscented Kalman Filter and Smoother
==============================================

This simple example shows how one may apply the Unscented Kalman Filter and
Unscented Kalman Smoother to some randomly generated data.

The Unscented Kalman Filter (UKF) and Rauch-Rung-Striebel type Unscented Kalman
Smoother (UKS) are a generalization of the traditional Kalman Filter and
Smoother to models with non-linear equations describing state transitions and
observation emissions. Unlike the Extended Kalman Filter (EKF), which attempts
to perform the same task by using the numerical derivative of the appropriate
equations, the UKF selects a handful of "sigma points", passes them through the
appropriate function, then finally re-estimates a normal distribution around
those propagated points. Experiments have shown that the UKF and UKS are
superior to the EKF and EKS in nearly all scenarios.

The figure drawn shows the true, hidden state; the state estimates given by the
UKF; and finally the same given by the UKS.
'''
import numpy as np
import pylab as pl
from pykalman import UnscentedKalmanFilter
import warnings

warnings.filterwarnings('ignore')   # TODO

# initialize parameters
def transition_function(state, noise):
    x = state[0] + state[1]
    v = state[1] + state[2]
    bw_x = state[3] + state[4]
    bw_a = state[4] + state[5]
    return np.array([x,v, state[2], bw_x, bw_a, state[5]]) + noise

def observation_function(state, noise):
    return state[[0,3]] + noise

transition_covariance = np.eye(6)
random_state = np.random.RandomState(0)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
initial_state_mean = [0] * 6
initial_state_covariance = np.eye(6) + random_state.randn(6, 6) * 0.1

# sample from model
kf = UnscentedKalmanFilter(
    transition_function, observation_function,
    transition_covariance, observation_covariance,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)
states, observations = kf.sample(10, initial_state_mean)

# estimate state with filtering and smoothing
filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]

predicted_states = kf.sample(5, filtered_state_estimates[-1])[0]
filtered_state_estimates = np.vstack([filtered_state_estimates, predicted_states[1:]])

# draw estimates
pl.figure()
lines_true = pl.plot(states, color='b')
lines_filt = pl.plot(filtered_state_estimates, color='r', ls='-')
lines_smooth = pl.plot(smoothed_state_estimates, color='g', ls='-.')
pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
          ('true', 'filt', 'smooth'),
          loc='lower left'
)
pl.show()

class Kalman2DTracker:
    def __init__(self):
        self.inited = False
        self.kf = UnscentedKalmanFilter(
            transition_function, observation_function,
            transition_covariance, observation_covariance,
            initial_state_mean, initial_state_covariance,
            random_state=random_state
        )
        self.filtered_state_mean, self.filtered_state_covariance = \
            self.kf.filter_update(initial_state_mean, initial_state_covariance)
        self.last_state = [0.] * 6

    def update(self, observation):
        if not self.inited:
            _initial_state_mean = [observation[0], 0., 0., observation[1], 0., 0.]
            self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
                _initial_state_mean, initial_state_covariance
            )
        else:
            self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
                self.filtered_state_mean, self.filtered_state_covariance, observation)

    def sample(self, n):
        # sample n observations from the filter
        return self.kf.sample(n+1, self.last_state)[1:][..., [0, 3]]