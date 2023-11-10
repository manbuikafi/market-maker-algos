from typing import Tuple, Type, Dict
import numpy as np
from gymnasium import spaces
import math

from ..data_loader import BaseDataLoader
from .market_maker_env import MarketMakerEnv
from ..common import check_col

class LehalleEnv(MarketMakerEnv):
    """Environment for Lehalle expiriment.
    Price path is a brownian motion with drift
    Matching mechanism is based on high and low price

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
            high=500,
            shape=(4,),
            dtype=np.float32,
        )
        # reset to get sample ohlcdv_df
        self.data_loader.reset()
        self.asset_metadata = self.data_loader.asset_metadata
        self.dt = self.asset_metadata["dt"]
        check_col(self.data_loader.ohlcv_df, "open high low close".split())
        
        self.k = k
        self.risk_factor = risk_factor
        # self.A = 1 / self.dt / math.exp(self.k * 1 / 4)

    @property
    def market_metadata(self):
        return {
            # "A": self.A,
            "k": self.k,
            "risk_factor": self.risk_factor,
        }

    def _get_observation(self) -> np.ndarray:
        typical_price = (
            self.ohlcv_df.iloc[self._current_tick].close
            + self.ohlcv_df.iloc[self._current_tick].low
            + self.ohlcv_df.iloc[self._current_tick].high
        ) / 3
        obs = np.asarray(
            [
                typical_price,
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
        high = self.ohlcv_df.iloc[self._current_tick].high
        low = self.ohlcv_df.iloc[self._current_tick].low
        
        matched_ask, matched_bid = 0, 0
        if ask_price <= high:
            matched_ask = ask_quantity
        if bid_price >= low:
            matched_bid = bid_quantity

        return matched_bid, matched_ask

    def _calculate_reward(self) -> float:
        try:
            last_nav = self.history_info["nav"][-1]
        except:
            last_nav = self.init_cash
        return self.nav - last_nav
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        out = super().step(action)
        added_info = {
            'open': self.ohlcv_df.iloc[self._current_tick].open,
            'high': self.ohlcv_df.iloc[self._current_tick].high,
            'low': self.ohlcv_df.iloc[self._current_tick].low,
        }
        self.update_info(info=added_info)
        return out
