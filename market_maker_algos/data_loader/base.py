from abc import abstractmethod, ABC
import pandas as pd


class BaseDataLoader(ABC):
    def __init__(self):
        self.ohlcv_df = None

    def save(self, path):
        # save csv
        if isinstance(self.ohlcv_df, pd.DataFrame):
            self.ohlcv_df.to_csv(path)
        else:
            raise NotImplementedError("Only support saving pd.DataFrame to csv file")

    def load(self, path):
        # load csv, check exception
        try:
            self.ohlcv_df = pd.read_csv(path, index_col=0)
            self.ohlcv_df.index = pd.to_datetime(self.ohlcv_df.index)
        except Exception as e:
            raise e

    @property
    @abstractmethod
    def asset_metadata(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
