from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from src.base.base_agent import BaseAgent


@dataclass(slots=True)
class SimpleAgentState:
    cash: np.ndarray  # np.ndarray[B]
    quantities: np.ndarray  # np.ndarray[B, N]


class SimpleAgent(BaseAgent):
    def __init__(self, batch_size, n_symbols):
        self.batch_size = batch_size
        self.n_symbols = n_symbols
        self.state = SimpleAgentState(
            cash=np.zeros((batch_size,)), quantities=np.zeros((batch_size, n_symbols))
        )
        self._default_state = SimpleAgentState(
            cash=np.zeros((batch_size,)), quantities=np.zeros((batch_size, n_symbols))
        )

    def set_default_state(self, cash: int = None, quantities: np.ndarray = None):
        if cash:
            self._default_state.cash.fill(cash)
        if quantities:
            self._default_state.quantities = quantities

    def welfare(self, market_prices):
        """
        :param market_prices: np.ndarray[B, N]
        :return: np.ndarray[B]
        """
        return self.state.cash + np.sum(self.state.quantities * market_prices, axis=1)

    def net_profit(self, market_prices):
        return self.welfare(market_prices) - self._default_state.cash

    def observe(self):
        return self.state

    def act(self, actions: np.ndarray, env_state: np.ndarray):
        """
        :param actions: np.ndarray[B, N, AF]
        :param env_state: np.ndarray[B, N, (SF)]
        :return:
        """
        buy_actions = actions[:, :, 0]
        sell_actions = actions[:, :, 1]
        prices = env_state  # [B, N]
        # Sell
        quantities_outflow = self.state.quantities * sell_actions
        cost_outflow = quantities_outflow * prices
        cash_inflow = np.sum(cost_outflow, axis=1)
        self.state.cash += cash_inflow
        self.state.quantities -= quantities_outflow
        # Buy
        cost_inflow = self.state.cash[:, np.newaxis] * buy_actions
        quantities_inflow = cost_inflow / prices
        cash_outflow = np.sum(cost_inflow, axis=1)
        self.state.cash -= cash_outflow
        self.state.quantities += quantities_inflow
        # Just sanity check
        assert all(
            np.all(arr >= 0)
            for arr in (
                cash_inflow,
                quantities_outflow,
                cash_outflow,
                quantities_inflow,
                actions,
                prices,
            )
        )
        return None

    def reset(self):
        self.state = deepcopy(self._default_state)
