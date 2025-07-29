from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):

    @abstractmethod
    def __call__(self, state) -> np.ndarray:
        """
        :param state: np.ndarray[B, N, SF]
        :return actions: np.ndarray[B, N, AF]
        """
