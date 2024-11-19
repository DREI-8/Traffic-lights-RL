""" Module for some RL algorithms. """

from .DQN import DQN_agent
from .fixed_time import fixed_time_agent


__all__ = ['DQN_agent', 'fixed_time_agent']