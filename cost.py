# base class for system cost
from abc import ABCMeta, abstractmethod

class Cost(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def running_cost(self, x): 
        return

    @abstractmethod
    def terminal_cost(self, x):
        return
