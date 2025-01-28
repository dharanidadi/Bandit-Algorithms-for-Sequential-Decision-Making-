import numpy as np
import math

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

def KL(p, q):
    if p==q:
        return 0
    elif q==0 or q==1:
        return float("inf")
    elif p==1:
        return math.log(1/q)
    elif p==0:
        return math.log(1/(1-q))
    else:
        return p*math.log(p/q)+(1-p)*math.log((1-p)/((1-q)+10e-6))

    
def find_q(rhs, p_a):
    if p_a == 1:
        return 1
    q = np.arange(p_a, 1, 0.01)
    lhs = [KL(p_a, i) for i in q]
    lhs_array = np.array(lhs)
    difference = lhs_array - rhs
    difference = difference[difference <= 0]
    min_index = np.argmax(difference)
    return q[min_index]

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time_step = 0
        self.epsilon = 10e-8
    
    def give_pull(self):
        if self.time_step < self.num_arms:
            return self.time_step
        else:
            ucb_values = self.values + np.sqrt((2*math.log(self.time_step))/(self.counts+self.epsilon))
            return np.argmax(ucb_values)
        
    def get_reward(self, arm_index, reward):
        self.time_step +=1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time_step = 0
        self.epsilon = 10e-6
    
    def give_pull(self):
        if self.time_step < self.num_arms:
            return self.time_step
        else:
            kl_ucb_values = np.zeros(self.num_arms)
            for arm in range(self.num_arms):
                p_a = self.values[arm]
                rhs = math.log(self.time_step) / (self.counts[arm] + self.epsilon)
                kl_ucb_values[arm] = find_q(rhs, p_a)
            return np.argmax(kl_ucb_values)

    def get_reward(self, arm_index, reward):
        self.time_step +=1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.no_of_successes = np.zeros(num_arms)
        self.no_of_failures = np.zeros(num_arms)
    
    def give_pull(self):
        beta_sample = np.random.beta(self.no_of_successes+1, self.no_of_failures+1)
        return np.argmax(beta_sample)
    
    def get_reward(self, arm_index, reward):
        if reward == 1:
            self.no_of_successes[arm_index] += 1
        else:
            self.no_of_failures[arm_index] += 1
