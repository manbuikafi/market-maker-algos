from typing import Tuple, Type, Dict
import numpy as np
from gymnasium import spaces
import math

from ..data_loader import BaseDataLoader
from .market_maker_env import MarketMakerEnv


class AvellanedaStoikovEnv(MarketMakerEnv):
    """Environment for Avellaneda-Stoikov experiment
    Price path is a brownian motion with drift
    Matching mechanism is based on exponential distribution
    
    Reference:
    High-frequency trading in a limit order book, Marco Avellaneda & Sasha Stoikov
    paper url: https://www.researchgate.net/publication/24086205_High_Frequency_Trading_in_a_Limit_Order_Book
    Some model limitations, discussed: https://quant.stackexchange.com/questions/36400/avellaneda-stoikov-market-making-model
    Parameter fitting: https://quant.stackexchange.com/questions/36073/how-does-one-calibrate-lambda-in-a-avellaneda-stoikov-market-making-problem

    Args:
        data_loader (Type[BaseDataLoader]): data loader class
        init_cash (float, optional): initial cash. Defaults to 2e4.
        random_seed (int, optional): random seed. Defaults to None.
        bid_fee (float, optional): bid fee. Defaults to 0.03%.
        ask_fee (float, optional): ask fee. Defaults to 0.13%.
    """

    def __init__(
        self,
        data_loader: Type[BaseDataLoader],
        init_cash: float = 0,
        k: float = 1.5,
        risk_factor: float = 0.1,
        bid_fee: float = 0.0003,
        ask_fee: float = 0.0013,
    ):
        super().__init__(
            data_loader=data_loader,
            init_cash=init_cash,
            bid_fee=bid_fee,
            ask_fee=ask_fee,
        )

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )
        self.k = k
        self.risk_factor = risk_factor
        self.dt = self.asset_metadata["dt"]
        self.A = 1 / self.dt / math.exp(self.k * 1 / 4)

    @property
    def market_metadata(self):
        return {
            "A": self.A,
            "k": self.k,
            "risk_factor": self.risk_factor,
        }

    def _get_observation(self) -> np.ndarray:
        obs = np.asarray(
            [
                self._current_price,
                self.quantity,
                self._current_tick,
                self.market_metadata["risk_factor"],
                self.market_metadata["k"],
                self.asset_metadata["sigma"],
                self.asset_metadata["total_time"],
                self.asset_metadata["dt"],
            ]
        ).astype(np.float32)
        return obs

    def _validate_action(self, action: np.ndarray) -> Tuple[int, float, int, float]:
        """Validate action and return valid action for current environment"""
        bid_quantity, bid_price, ask_quantity, ask_price = action

        bid_quantity = int(bid_quantity)
        ask_quantity = int(ask_quantity)

        bid_price = float(bid_price)
        ask_price = float(ask_price)

        return bid_quantity, bid_price, ask_quantity, ask_price

    def _matching_order(
        self,
        bid_quantity: int,
        bid_price: float,
        ask_quantity: int,
        ask_price: float,
    ) -> Tuple[int, int]:
        delta_ask = ask_price - self._current_price
        delta_bid = self._current_price - bid_price

        # market intensities
        lambda_ask = self.A * math.exp(-self.k * delta_ask)
        lambda_bid = self.A * math.exp(-self.k * delta_bid)

        # Order consumption (can be both per time step)
        matched_bid, matched_ask = 0, 0
        prob_ask = 1 - math.exp(-lambda_ask * self.dt)
        prob_bid = 1 - math.exp(-lambda_bid * self.dt)

        if np.random.random() < prob_ask:
            matched_ask = ask_quantity
        if np.random.random() < prob_bid:
            matched_bid = bid_quantity

        return matched_bid, matched_ask

    def _calculate_reward(self) -> float:
        try:
            last_nav = self.history_info["nav"][-1]
        except:
            last_nav = self.init_cash
        return self.nav - last_nav
