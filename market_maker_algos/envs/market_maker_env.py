from abc import abstractmethod
import numpy as np
import gymnasium as gym
from typing import Tuple, Type, Dict, Any
import pandas as pd

from ..data_loader import BaseDataLoader


class MarketMakerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data_loader: Type[BaseDataLoader],
        init_cash: float = 2e4,
        bid_fee: float = 0.0003,
        ask_fee: float = 0.0013,
    ):
        self.init_cash = init_cash
        self.bid_fee = bid_fee
        self.ask_fee = ask_fee
        self.data_loader = data_loader
        self.asset_metadata = data_loader.asset_metadata

        # update these variables in reset method
        self.ohlcv_df = None
        self.quantity = None
        self.cash = None
        self._current_tick = None

    @property
    def _current_price(self) -> float:
        return self.ohlcv_df.iloc[self._current_tick].close

    @property
    def nav(self) -> float:
        return self.cash + self.quantity * self._current_price

    def update_info(self, info: dict) -> None:
        if not self.history_info:
            self.history_info = {key: [] for key in info.keys()}

        for key, value in info.items():
            if key in self.history_info:
                self.history_info[key].append(value)
            else:
                self.history_info[key] = [value]
                

    def is_done(self) -> Tuple[bool, bool]:
        truncated = False
        terminated = self._current_tick == self._end_episode_tick
        # gymnaisum interface
        return terminated, truncated

    def get_history_info(self):
        return pd.DataFrame(self.history_info)

    def update_inventory(
        self, bid_quantity: int, bid_price: float, ask_quantity: int, ask_price: float
    ) -> None:
        self.quantity += bid_quantity - ask_quantity
        bid_cashflow = bid_quantity * bid_price * (1 + self.bid_fee)
        ask_cashflow = ask_quantity * ask_price * (1 - self.ask_fee)
        self.cash += ask_cashflow - bid_cashflow

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # reset data loader
        self.ohlcv_df = self.data_loader.reset()
        assert isinstance(self.ohlcv_df, pd.DataFrame)
        self._end_episode_tick = self.ohlcv_df.shape[0] - 1
        self.asset_metadata = self.data_loader.asset_metadata
        self.dt = self.asset_metadata["dt"]
        
        self.history_info = {}
        self.quantity = 0
        self.cash = self.init_cash
        self._current_tick = 0
        info = self.asset_metadata

        return self._get_observation(), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._current_tick += 1

        # validate action and modify if needed
        (
            bid_quantity,
            bid_price,
            ask_quantity,
            ask_price,
        ) = self._validate_action(action)

        # matching order
        matched_bid, matched_ask = self._matching_order(
            bid_quantity=bid_quantity,
            bid_price=bid_price,
            ask_quantity=ask_quantity,
            ask_price=ask_price,
        )
        # update inventory
        self.update_inventory(
            bid_quantity=matched_bid,
            bid_price=bid_price,
            ask_quantity=matched_ask,
            ask_price=ask_price,
        )

        step_reward = self._calculate_reward()

        # update info last
        current_info = {
            "datetime": self.ohlcv_df.iloc[self._current_tick].datetime,
            "quantity": self.quantity,
            "cash": self.cash,
            "bid_quantity": bid_quantity,
            "bid_price": bid_price,
            "ask_quantity": ask_quantity,
            "ask_price": ask_price,
            "matched_bid_quantity": matched_bid,
            "matched_ask_quantity": matched_ask,
            "close": self._current_price,
            "step_reward": step_reward,
            "nav": self.nav,
        }
        self.update_info(info=current_info)

        return self._get_observation(), step_reward, *self.is_done(), current_info

    @abstractmethod
    def _get_observation(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _calculate_reward(self, *args, **kwargs) -> Tuple[Any]:
        raise NotImplementedError
    
    @abstractmethod
    def _validate_action(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _matching_order(self, *args, **kwargs) -> Tuple[Any]:
        """Matching order mechanism. Need to be implemented in subclass

        Returns:
            Tuple[int, float, int, float]: bid_quantity, bid_price, ask_quantity, ask_price
        """
        raise NotImplementedError
