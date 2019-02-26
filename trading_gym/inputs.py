import os
import json
import types
import datetime
import random
import matplotlib.dates as mdates
import numpy as np


class Observation(object):
    __slots__ = ["open", "close", "high", "low", "index", "volume",
                 "date", "date_string", "format_string"]
    N = 6

    def __init__(self, **kwargs):
        self.index = kwargs.get('index', 0)
        self.open = kwargs.get('open', 0)
        self.close = kwargs.get('close', 0)
        self.high = kwargs.get('high', 0)
        self.low = kwargs.get('low', 0)
        self.volume = kwargs.get('volume', 0)
        self.date_string = kwargs.get('date', '')
        self.date = self._format_time(self.date_string)

    @property
    def math_date(self):
        return mdates.date2num(self.date)

    @property
    def math_hour(self):
        return (self.date.hour * 100 + self.date.minute) / 2500

    def _format_time(self, date):
        d_f = date.count('-')
        t_f = date.count(':')
        if d_f == 2:
            date_format = "%Y-%m-%d"
        elif d_f == 1:
            date_format = "%m-%d"
        else:
            date_format = ''

        if t_f == 2:
            time_format = '%H:%M:%s'
        elif t_f == 1:
            time_format = '%H:%M'
        else:
            time_format = ''
        self.format_string = " ".join([date_format, time_format]).rstrip()
        return datetime.datetime.strptime(date, self.format_string)

    def to_ochl(self):
        return [self.open, self.close, self.high, self.low]

    def to_list(self):
        return [self.index] + self.to_ochl()

    def to_array(self, extend):
        volume = self.volume / 10000
        extend = [volume] + extend
        return np.array(self.to_ochl() + extend, dtype=float)

    def __str__(self):
        return "date: {self.date}, open: {self.open}, close:{self.close}," \
            " high: {self.high}, low: {self.low}".format(self=self)

    @property
    def latest_price(self):
        return self.close


class History(object):

    def __init__(self, obs_list, history_num):
        self.obs_list = obs_list
        self.history_num = history_num

    def normalize(self, base):
        def __nor(array):
            return [(x - base) / base for x in array]
        return __nor

    def to_array(self, base, extend=[]):
        # base = self.obs_list[-1].close if self.obs_list else 1

        history = np.zeros([self.history_num + 1, Observation.N], dtype=float)
        normalize_func = self.normalize(base)
        for i, obs in enumerate(self.obs_list):
            data = obs.to_array(extend=extend)
            history[i] = normalize_func(data)
        return history


class DataManager(object):

    def __init__(self, data=None,
                 data_path=None,
                 data_func=None,
                 previous_steps=None,
                 history_num=50,
                 start_random=True,
                 use_ta=False,
                 ta_timeperiods=None):
        if data is not None:
            if isinstance(data, list):
                data = data
            elif isinstance(data, types.GeneratorType):
                data = list(data)
            else:
                raise ValueError('data must be list or generator')
        elif data_path is not None:
            data = self.load_data(data_path)
        elif data_func is not None:
            data = self.data_func()
        else:
            raise ValueError(
                "The argument `data_path` or `data_func` is required.")
        self.data = list(self._to_observations(data))
        self.index = 0
        self.max_steps = len(self.data)
        self.max_price = max([obs.high for obs in self.data])
        self.history_num = history_num
        self.feature_num = Observation.N
        self.start_random = start_random

        if use_ta:
            self._load_data_with_ta(ta_timeperiods)

    def _load_data_with_ta(self, ta_timeperiods):
        from .ta import TaFeatures
        self.ta_data = TaFeatures(self.data, timeperiods=ta_timeperiods)

    def _to_observations(self, data):
        for i, item in enumerate(data):
            item['index'] = i
            yield Observation(**item)

    def _load_json(self, path):
        with open(path) as f:
            data = json.load(f)
        return data

    def _load_pd(self, path):
        pass

    def load_data(self, path=None):
        filename, extension = os.path.splitext(path)
        if extension == '.json':
            data = self._load_json(path)
        return data

    @property
    def default_space(self):
        return [self.history_num, self.feature_num]

    @property
    def first_price(self):
        return self.data[0].close

    @property
    def current_step(self):
        return self.data[self.index]

    @property
    def history(self):
        return self.data[:self.index]

    @property
    def recent_history(self):
        return History(self.data[self.index - self.history_num:self.index + 1],
                       history_num=self.history_num)

    @property
    def total(self):
        return self.history + [self.current_step]

    @property
    def ta_features(self):
        return self.ta_data.get_feature(self.index)

    def step(self):
        observation = self.current_step
        self.index += 1
        done = self.index >= self.max_steps
        return observation, done

    def reset(self, index=None):
        if not self.start_random:
            self.index = self.history_num + 1
        elif index is None:
            self.index = random.randint(self.history_num,
                                        int(self.max_steps * 0.8))
        else:
            self.index = index


class ExtraFeature:
    ex_obs_name = ['amount', 'buy_at', 'index', 'floating_rate', 'math_hour']

    def get_extra_features(self, info):
        features = {}
        features['amount'] = self.exchange.amount
        features['buy_at'] = info['buy_at'] / self.data.max_steps
        features['index'] = info['index'] / self.data.max_steps
        features['floating_rate'] = self.exchange.floating_rate
        features['math_hour'] = self.obs.math_hour

        return [features[name] for name in self.ex_obs_name]
