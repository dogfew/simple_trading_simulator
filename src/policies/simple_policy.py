import numpy as np

from src.base.base_policy import BasePolicy


class SimpleGridPolicy(BasePolicy):
    """
    If price < buy_cutoff: buy using proportion of your cash
    If price > sell_cutoff: sell using proportion of your quantity
    """

    def __init__(
        self,
        batch_size: int,
        n_symbols: int,
        step_proportion: float = 0.1,
        buy_cutoff: float = 1.0,
        sell_cutoff: float = 2.5,
    ):
        self.n_features = 2
        self.n_symbols = n_symbols
        self.batch_size = batch_size
        self.proportion = step_proportion
        self.buy_cutoff = buy_cutoff
        self.sell_cutoff = sell_cutoff

    def __call__(self, prices: np.ndarray):
        actions = np.zeros((self.batch_size, self.n_symbols, self.n_features))
        actions[:, :, 0] = np.where(
            prices < self.buy_cutoff, self.proportion, actions[:, :, 0]
        )
        actions[:, :, 1] = np.where(
            prices > self.sell_cutoff, self.proportion, actions[:, :, 1]
        )
        actions = actions / np.maximum(actions.sum(axis=1, keepdims=True), 1.0)
        return actions


class FixedPercentPolicy:
    def __init__(self, percents: np.ndarray):
        raise NotImplementedError

    def __call__(
        self,
    ):
        raise NotImplementedError
