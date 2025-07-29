from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    state: Any  # Collection[np.ndarray[B, ...], np.ndarray[B, ...]]

    @abstractmethod
    def observe(self):
        """
        Get agent state.
        """

    @abstractmethod
    def act(self, actions: np.ndarray, env_state: np.ndarray):
        """
        :param actions: np.ndarray[B, N, AF]
        :param env_state: np.ndarray[B, N, SF]
        """

    @abstractmethod
    def reset(self):
        """
        Set agent's state to default
        """
