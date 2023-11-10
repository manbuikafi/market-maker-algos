import pandas as pd
import numpy as np

from ..common import brownian
from ..data_loader import BaseDataLoader


class SingleBrownianMotion(BaseDataLoader):
    def __init__(self, init_value: float, n_sample, sigma, total_time: int = 1):
        self.n_sample = n_sample
        self.sigma = sigma
        self.total_time = total_time
        self.init_value = init_value
        self.dt = total_time / n_sample

    @property
    def asset_metadata(self):
        return {
            "type": "brownian",
            "n_sample": self.n_sample,
            "sigma": self.sigma,
            "total_time": self.total_time,
            "init_value": self.init_value,
            "dt": self.dt,
        }

    def reset(self) -> pd.DataFrame:
        brownian_path = np.empty(self.n_sample)
        brownian_path[0] = self.init_value
        brownian(
            x0=self.init_value,
            n=self.n_sample - 1,
            dt=self.dt,
            delta=self.sigma,
            out=brownian_path[1:],
        )
        ohlcv_df = pd.DataFrame(
            {
                "datetime": np.arange(self.n_sample) + 1,
                "close": brownian_path,
            }
        )
        ohlcv_df["datetime"] = pd.to_datetime(ohlcv_df["datetime"], unit="s")
        return ohlcv_df
