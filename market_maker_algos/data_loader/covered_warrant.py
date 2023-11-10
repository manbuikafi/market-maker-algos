import pandas as pd
import numpy as np
import quantstats as qs

from ..data_loader import BaseDataLoader


class RandomCoveredWarrantLoader(BaseDataLoader):
    def __init__(self, path):
        self.path = path

        data = pd.read_csv(path)
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["date"] = data["datetime"].dt.date
        data["sample_id"] = data["date"].astype(str) + "_" + data["sec_cd"].astype(str)
        self.data = data

        self.sample_ids = self.data["sample_id"].unique()
        self._asset_metadata = {"type": "covered_warrant"}

    @property
    def asset_metadata(self):
        return self._asset_metadata

    def reset(self):
        sample_id = np.random.choice(self.sample_ids, size=1).item()
        sample_df = self.data[self.data.sample_id == sample_id]
        resample_df = (
            sample_df.resample("1min", on="datetime")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .ffill()
        )
        resample_df.sort_values(by="datetime", inplace=True)
        resample_df.reset_index(inplace=True)
        resample_df["open"] = resample_df["open"] / 1000
        resample_df["high"] = resample_df["high"] / 1000
        resample_df["low"] = resample_df["low"] / 1000
        resample_df["close"] = resample_df["close"] / 1000
        self.ohlcv_df = resample_df

        # update asset metadata
        date, sec_cd = sample_id.split("_")
        dt = 1 / resample_df.shape[0]
        total_time = resample_df.shape[0]
        # TODO: update volatility model
        sigma = qs.stats.volatility(resample_df["close"].pct_change())
        self._asset_metadata.update(
            {
                "date": date,
                "sec_cd": sec_cd,
                "dt": dt,
                "total_time": total_time,
                # "sigma": sigma,
                "sigma": 0.0002,
            }
        )

        return self.ohlcv_df
