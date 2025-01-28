import numpy as np
import math

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault # probability that the bandit returns a faulty pull
        self.no_of_successes = np.zeros(num_arms)
        self.no_of_failures = np.zeros(num_arms)
    
    def give_pull(self):
        beta_sample = np.random.beta((self.no_of_successes+1), (self.no_of_failures+1))
        return np.argmax(beta_sample)
    
    def get_reward(self, arm_index, reward):
        if reward == 1:
            self.no_of_successes[arm_index] += 1
        else:
            self.no_of_failures[arm_index] += 1
        
