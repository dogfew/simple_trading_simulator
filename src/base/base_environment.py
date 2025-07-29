from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEnvironment(ABC):
    data: np.ndarray | Any  # Either np.ndarray or collection of np.ndarrays
    current_tick: int
    max_tick: int

    @abstractmethod
    def observe(self, tick: int):
        """
        :param tick:
        :return env_state: np.ndarray[B, N, F]
        """

    @abstractmethod
    def step(self, actions: np.ndarray):
        """
        :param actions: np.ndarray[B, N, AF]
        :return observations, rewards, is_terminated:
        """

    @abstractmethod
    def reset(self): ...
