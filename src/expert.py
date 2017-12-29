import numpy as np
from Queue import Queue
from abc import ABCMeta, abstractmethod


class Expert(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name, n_obs=10):
        self.reward = 0.
        # Expert's recommendation: buy or not
        self.buy = False
        self.last_price = None

        if n_obs:
            self.history = Queue(maxsize=n_obs)
        return None

    @abstractmethod
    def update(self, data):
        raise NotImplementedError("Expert subclass needs to implement update() function.")

class Dummy(Expert):

    def __init__(self, name):
        super(Dummy, self).__init__(name, None)
        self.pick = True
        return None

    def update(self, price):
        price = float(price)
        if not self.last_price:
            self.last_price = price
            return None

        self.reward = price/self.last_price
        self.last_price = price
        return None

class MeanReversion(Expert):

    def __init__(self, name, n_obs, threshold):
        super(MeanReversion, self).__init__(name, n_obs)
        self.avg = 0.0
        self.std = 0.0
        self.n_obs = n_obs
        self.threshold = threshold

    def update(self, price):
        price = float(price)

        if not self.history.queue:
            self.history.put(price)
            return None

        self.last_price = self.history.queue[0]

        # Expert reward is updated each round. 
        # If current recommendation is not buy, expert's reward is the inverse of the price mvmt
        if self.buy:
            self.reward = price/self.last_price
        else:
            self.reward = self.last_price/price

        _ = self.history.get()
        self.history.put(price)

        self.avg = np.mean(list(self.history.queue))
        self.std = np.std(list(self.history.queue))
        
        if price <= self.avg - self.threshold * self.std:
            self.buy = True
        else:
            self.buy = False

        self.last_price = price

        return None



