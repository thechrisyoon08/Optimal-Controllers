# base class for nonlinear system 
from abc import ABCMeta, abstractmethod

class System(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.state_dim = None
        self.control_dim = None
        self.x_names = []

    @abstractmethod
    def initialize_state(self, x_init):
        self.x_init = x_init

    @abstractmethod
    def dynamics(self, x, u):
        return
