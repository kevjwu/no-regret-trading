import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod


class Agent(object):
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, experts):
        self.experts = experts
        self.weights = np.ones(len(experts))/len(experts)
        self.rewards = None
        self.returns = None
        self.weights_history = []
        return None

    @abstractmethod
    def update(self):
        raise NotImplementedError("Agent subclass must implement update() function.")

class EG(Agent):

    def __init__(self, experts, eta):
        super(EG, self).__init__(experts)
        self.eta = eta
        return None

    
    def update(self):
        self.rewards = np.asarray([e.reward for e in self.experts.values()])

        if not np.all(self.rewards):
            return None

        multipliers = np.exp(self.eta * self.rewards/np.sum(
            self.weights * self.rewards))
        self.weights = (self.weights * multipliers)/np.sum(
                self.weights * multipliers) 
        
        self.weights_history.append(self.weights)

        return None


class BuyHold(Agent):

    def __init__(self, experts, weights=None):
        super(BuyHold, self).__init__(experts)
        if weights:
            self.weights = weights
        return None

    def update(self):
        self.rewards = np.asarray([e.reward for e in self.experts.values()])        
        if not np.all(self.rewards):
            return None
        weights = np.multiply(self.weights, self.rewards)
        self.weights = np.divide(weights, np.sum(weights))
        return None


class ConstantRebalancer(Agent):

    def __init__(self, experts, weights=None):
        super(ConstantRebalancer, self).__init__(experts)
        if weights:
            self.weights = weights
        return None

    def update(self):
        self.rewards = np.asarray([e.reward for e in self.experts.values()])   
        return None
