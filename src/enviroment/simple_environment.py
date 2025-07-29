import numpy as np
from numpy import ndarray

from src.agent.simple_agent import SimpleAgent


class SimpleEnvironment:
    """Stores only price dynamics"""

    def __init__(self, agent: SimpleAgent, data: np.ndarray, start_tick: int = 0):
        """
        :param data: np.ndarray[(1), N, T] | np.ndarray[B, N, T]
        """
        if data.ndim == 2:
            data = data[np.newaxis, :]
        elif data.ndim != 3:
            raise ValueError("Expected 3 dims")
        (self.batch_size, self.n_symbols, _) = data.shape
        self.agent: SimpleAgent = agent
        self.data = data
        self.max_tick = self.data.shape[-1] - 1
        self.start_tick = min(start_tick, self.max_tick)
        self.current_tick = self.start_tick

    def observe(self, tick) -> np.ndarray:
        """
        :param tick: int
        :return env_state: np.ndarray[B, N]
        """
        return self.data[:, :, tick]

    def step(self, actions: np.ndarray):
        """
        :param actions: np.ndarray[B, N, AF]
        :return:
        """
        self.current_tick += 1
        if self.current_tick > self.max_tick:
            raise StopIteration
        obs_market = self.observe(self.current_tick)  # Observe market state
        obs_agent = self.agent.observe()  # Observe agent state
        self.agent.act(actions, obs_market)
        reward = self.agent.welfare(obs_market)
        terminated = np.full(
            self.batch_size,
            fill_value=self.current_tick == self.max_tick,
            dtype=np.bool,
        )
        return (obs_market, obs_agent), reward, terminated

    def reset(self):
        self.current_tick = 0
        self.agent.reset()
