import talib
import numpy as np


class TaFeatures:
    features = ['ema', 'wma', 'sma', 'sar', 'apo', 'macd',
                'macdsignal', 'macdhist', 'adosc', 'obv', 'rsi']
    default_timeperiods = [5, 10, 15, 30]

    def __init__(self, observations, timeperiods=None):
        if len(observations) == 0:
            raise ValueError("Observations is empty!")

        self.observations = observations
        if timeperiods is None:
            self.timeperiods = self.default_timeperiods
        else:
            self.timeperiods = timeperiods

        self.init_base_data()
        self.init_index()

    @property
    def feature_space(self):
        return [len(self.timeperiods), len(self.features)]

    def init_base_data(self):
        self.open = np.array([obs.open for obs in self.observations])
        self.close = np.array([obs.close for obs in self.observations])
        self.high = np.array([obs.high for obs in self.observations])
        self.low = np.array([obs.low for obs in self.observations])
        self.volume = np.array([obs.volume for obs in self.observations])

        p_n = self.normalize(self.high.max())
        self.n_open = np.array(p_n(self.open))
        self.n_close = np.array(p_n(self.close))
        self.n_high = np.array(p_n(self.high))
        self.n_low = np.array(p_n(self.low))

        v_n = self.normalize(self.volume.max())
        self.n_volume = np.array(v_n(self.volume))

    def init_index(self):
        self.ema = np.array([talib.EMA(self.n_close, timeperiod=t)
                             for t in self.timeperiods])
        self.wma = np.array([talib.WMA(self.n_close, timeperiod=t)
                             for t in self.timeperiods])
        self.sma = np.array([talib.SMA(self.n_close, timeperiod=t)
                             for t in self.timeperiods])
        self.sar = np.array([talib.SAR(self.n_high, self.n_low)
                             for t in self.timeperiods])

        self.apo = np.array([talib.APO(self.close,
                                       fastperiod=t / 2,
                                       slowperiod=t,
                                       matype=0)
                             for t in self.timeperiods])
        macd = np.array([(talib.MACD(
            self.close,
            fastperiod=t / 2,
            slowperiod=t,
            signalperiod=t / 2 - 1))
            for t in self.timeperiods])
        self.macd = macd[:, 0]
        self.macdsignal = macd[:, 1]
        self.macdhist = macd[:, 2]

        self.adosc = np.array([talib.ADOSC(self.n_high,
                                           self.n_low,
                                           self.n_close,
                                           self.n_volume,
                                           fastperiod=3, slowperiod=t)
                               for t in self.timeperiods])
        self.obv = np.array([talib.OBV(self.n_close, self.n_volume)
                             for t in self.timeperiods])
        self.rsi = np.array([talib.RSI(self.n_close, timeperiod=t)
                             for t in self.timeperiods])

    def normalize(self, base):
        def __nor(array):
            return [(x - base) / base for x in array]
        return __nor

    def get_feature(self, index):
        features = np.zeros(self.feature_space)
        for i, feature_name in enumerate(self.features):
            feature = getattr(self, feature_name)
            features[:, i] = feature[:, index]
        return np.nan_to_num(features)
