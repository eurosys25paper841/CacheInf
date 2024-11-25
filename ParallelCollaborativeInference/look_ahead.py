
from multiprocessing import Queue
import numpy as np
from pykalman import UnscentedKalmanFilter
from collections import deque


def transition_function(state, noise):
    x = state[0] + state[1] # x + v_x
    v = state[1] + state[2] # v_x + acc_x
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


class Kalman2DTracker:
    def __init__(self, look_ahead_steps=5, init_len=10, buffer_len=10) -> None:
        self.init_len = init_len
        self.look_ahead_steps = look_ahead_steps
        self.observations = deque(maxlen=buffer_len)

    def observe(self, observation):
        self.observations.append(observation)

    def reset(self):
        self.observations.clear()

    @property
    def can_predict(self):
        return len(self.observations) >= self.init_len

    def predict(self):
        # predict for next look_ahead_steps
        assert self.can_predict
        _initial_state_mean = [self.observations[0][0], 0., 0., self.observations[0][1], 0., 0.]
        assert len(self.observations) > self.init_len
        kf = UnscentedKalmanFilter(
            transition_function, observation_function,
            transition_covariance, observation_covariance,
            _initial_state_mean, initial_state_covariance,
            random_state=random_state
        )
        temp_obs = list(self.observations) + [[0.,0.]] * self.look_ahead_steps
        temp_obs = np.ma.array(temp_obs)
        temp_obs[-self.look_ahead_steps:] = np.ma.masked
        predicted_obs = kf.filter(temp_obs)[0][-self.look_ahead_steps:, [0,3]]

        return predicted_obs

RESET = 0
OBS = 1
PREDICT = 2
def look_ahead_process(look_ahead_steps: int,
                       order_queue: Queue,
                       observation_queue: Queue,
                       predict_queue: Queue):
    tracker = Kalman2DTracker(look_ahead_steps)
    count = 0
    while True:
        tracker.observe(observation_queue.get())
        if tracker.can_predict and count % look_ahead_steps == 0:
            predict_queue.put(tracker.predict().mean(0))
            count = 0
        count += 1


def test():
    x_increase = 0.1
    bandwidth_increase = 0.5
    tracker = Kalman2DTracker()
    for i in range(30):
        x = i*x_increase
        x_increase *= 1.01
        bw = i*bandwidth_increase
        bandwidth_increase *= 1.01
        tracker.observe([x, bw])
        print(f"{i} step input {[x,bw]}")
    print(f"look ahead 5 steps {tracker.predict(5)}")

if __name__ == "__main__":
    test()