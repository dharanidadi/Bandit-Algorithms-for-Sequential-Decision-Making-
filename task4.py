import numpy as np

class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
        self.no_of_successes = np.zeros((2, num_arms))
        self.no_of_failures = np.zeros((2, num_arms))
    
    def give_pull(self):
        beta_sample = np.random.beta(self.no_of_successes+1, self.no_of_failures+1)
        beta_sample_average = np.mean(beta_sample, axis=0)
        return np.argmax(beta_sample_average)
    
    def get_reward(self, arm_index, set_pulled, reward):
        if reward == 1:
            self.no_of_successes[set_pulled][arm_index] += 1
        else:
            self.no_of_failures[set_pulled][arm_index] += 1

