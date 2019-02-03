import os
import json
import types
import datetime
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_finance import candlestick_ochl
from gym.core import GoalEnv
from gym import spaces
import logging


logger = logging.getLogger('trading-gym')

HISTORY_NUM = 20
FEATURE_NUM = 8


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

    def __init__(self, obs_list):
        self.__obs_list = obs_list

    def normalize(self, base):
        def __nor(array):
            return [(x - base) / base for x in array]
        return __nor

    def to_array(self, base, extend=[]):
        # base = self.__obs_list[-1].close if self.__obs_list else 1
        history = np.zeros([HISTORY_NUM + 1, Observation.N], dtype=float)
        normalize_func = self.normalize(base)
        for i, obs in enumerate(self.__obs_list):
            data = obs.to_array(extend=extend)
            history[i] = normalize_func(data)
        return history


class DataManager(object):

    def __init__(self, data=None,
                 data_path=None,
                 data_func=None,
                 previous_steps=None,
                 history_num=HISTORY_NUM):
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
        self.history_num = history_num

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
        return History(self.data[self.index - self.history_num:self.index + 1])

    @property
    def total(self):
        return self.history + [self.current_step]

    def step(self):
        observation = self.current_step
        self.index += 1
        done = self.index >= self.max_steps
        return observation, done

    def reset(self, index=None):
        if index is None:
            self.index = random.randint(0, int(self.max_steps * 0.8))
        else:
            self.index = index


class Exchange(object):

    class ACTION:
        PUT = -1
        HOLD = 0
        PUSH = 1

    class Positions:

        def __init__(self, symbol=''):
            self.symbol = symbol
            self.amount = 0
            self.avg_price = 0
            self.created_index = 0

        @property
        def principal(self):
            return self.avg_price * self.amount

        @property
        def is_empty(self):
            return self.amount == 0

        @property
        def is_do_long(self):
            return self.amount > 0

        @property
        def is_do_short(self):
            return self.amount < 0

        def get_profit(self, latest_price, unit):
            if self.is_empty:
                return 0
            else:
                rate = 0
                if self.is_do_short:
                    diff = abs(self.avg_price) - latest_price
                else:
                    diff = latest_price - abs(self.avg_price)
                rate = diff / self.avg_price
                profit = rate * unit
                return profit

        def update(self, action, latest_price, unit, index):
            if action != 0 and self.amount == 0:
                self.created_index = index
            if action * self.amount < 0:
                profit = self.get_profit(latest_price, unit)
            else:
                profit = 0

            amount = action * unit
            if self.amount + amount != 0:
                self.avg_price = (self.avg_price * self.amount +
                                  amount * latest_price) / (self.amount + amount)
            else:
                self.avg_price = 0
                self.created_index = 0
            self.amount += amount
            return profit

    def __init__(self, punished=False, nav=50000, end_loss=None, unit=5000):
        self.nav = nav
        self.punished = punished
        self._end_loss = end_loss
        self.unit = unit
        self.__init_data()

    def __init_data(self):
        self.position = self.Positions()
        self.fixed_profit = 0
        self.floating_profit = 0

    @property
    def start_action(self):
        return self.ACTION.HOLD

    @property
    def available_funds(self):
        return self.nav - self.position.principal

    @property
    def profit(self):
        return self.fixed_profit + self.floating_profit

    @property
    def available_actions(self):
        if self.position.is_do_short:
            return [self.ACTION.PUSH, self.ACTION.HOLD]
        elif self.position.is_do_long:
            return [self.ACTION.PUT, self.ACTION.HOLD]
        else:
            return [self.ACTION.PUT, self.ACTION.HOLD, self.ACTION.PUSH]

    @property
    def punished_action(self):
        if self.position.is_do_short:
            return self.ACTION.PUT
        elif self.position.is_do_long:
            return self.ACTION.PUSH
        return None

    @property
    def cost_action(self):
        return [self.ACTION.PUT, self.ACTION.PUSH]

    @property
    def end_loss(self):
        if self._end_loss is not None:
            return self._end_loss
        return - self.unit * 0.12

    @property
    def is_over_loss(self):
        return self.profit < self.end_loss

    @property
    def amount(self):
        return self.position.amount / self.unit

    def get_profit(self, observation, symbol='default'):
        latest_price = observation.latest_price
        positions = self.positions.values()
        positions_profite = sum([position.get_profit(
            latest_price) for position in positions])
        return self.profit + positions_profite

    def get_charge(self, action, observation):
        """
            rewrite if inneed.
        """
        return 3

    def step(self, action, observation, symbol='default'):
        self.observation = observation
        latest_price = observation.latest_price

        if action in self.available_actions:
            fixed_profit = self.position.update(
                action, latest_price, self.unit, observation.index)
            if action in self.cost_action:
                fixed_profit -= self.get_charge(action, observation)
        else:
            fixed_profit = 0
        if self.punished:
            if action == self.punished_action:
                fixed_profit -= 1  # make it different
            if action == 0:
                fixed_profit -= 0.5  # make it action
        self.floating_profit = self.position.get_profit(
            latest_price, self.unit)
        self.fixed_profit += fixed_profit

        logger.info("latest_price:{}, amount:{}, action:{}".format(
            observation.latest_price, self.position.amount, action))
        logger.info("fixed_profit: {}, floating_profit: {}".format(
            self.fixed_profit, self.floating_profit))
        return self.profit / self.unit

    def reset(self):
        self.__init_data()

    @property
    def info(self):
        return {
            'index': self.observation.index,
            'date': self.observation.date_string,
            'unit': self.unit,
            'amount': self.position.amount,
            'avg_price': self.position.avg_price,
            'profit': {
                'total': self.profit,
                'fixed': self.fixed_profit,
                'floating': self.floating_profit
            },
            'buy_at': self.position.created_index,
            'latest_price': self.observation.latest_price,
        }


class Arrow(object):

    def __init__(self, coord, color):
        self.coord = coord
        self.color = color


class Render(object):

    def __init__(self, bar_width=0.8):
        plt.ion()
        self.figure, self.ax = plt.subplots()
        self.figure.set_size_inches((12, 6))
        self.bar_width = 0.8
        self.arrow_body_len = self.bar_width / 1000
        self.arrow_head_len = self.bar_width / 20
        self.arrow_width = self.bar_width
        self.max_windows = 60

        self.arrows = []

    @property
    def arrow_len(self):
        return self.arrow_body_len + self.arrow_head_len

    def take_action(self, action, observation):
        if action == Exchange.ACTION.PUT:
            y_s = observation.high + self.arrow_body_len + self.arrow_head_len
            y_e = - self.arrow_body_len
            color = 'gray'
        elif action == Exchange.ACTION.PUSH:
            y_s = observation.high
            y_e = self.arrow_body_len
            color = 'black'
        else:
            return
        x = observation.index
        self.arrows.append(Arrow((x, y_s, 0, y_e), color))

    def draw_arrow(self, data):
        first = data[0]
        arrows = list(filter(lambda x: x.coord[0] > first[0], self.arrows))
        for arrow in arrows:
            self.ax.arrow(*arrow.coord,
                          color=arrow.color,
                          head_width=self.arrow_width,
                          head_length=self.arrow_head_len)

        self.arrows = arrows

    def draw_title(self, info):
        formatting = """
        profit: {} \t fixed-profit: {} \t floating-profit:{} \t
        amount: {} \t latest-price: {} \t time: {} \n
        """

        def r(x): return round(x, 2)
        title = formatting.format(r(info['profit']['total']),
                                  r(info['profit']['fixed']),
                                  r(info['profit']['floating']),
                                  info['amount'],
                                  r(info['latest_price']),
                                  info['date'])
        plt.suptitle(title)

    def xaxis_format(self, history):
        def formator(x, pos=None):
            for h in history:
                if h.index == x:
                    return h.date.strftime("%H:%M")
            return ""
        return formator

    def render(self, history, info, mode='human', close=False):
        """
            If it plot one by one will cache many point
            that will cause OOM and work slow.
        """
        self.ax.clear()
        history = history[-self.max_windows:]
        data = [obs.to_list() for obs in history]
        candlestick_ochl(self.ax, data, colordown='g',
                         colorup='r', width=self.bar_width)
        self.draw_arrow(data)

        self.ax.xaxis.set_major_formatter(
            FuncFormatter(self.xaxis_format(history)))
        self.ax.set_ylabel('price')
        self.figure.autofmt_xdate()

        self.draw_title(info)
        plt.show()
        plt.pause(0.0001)

    def reset(self):
        self.ax.clear()
        self.arrows = []


class TradeEnv(GoalEnv):

    def __init__(self,
                 data=None,
                 data_path=None,
                 data_func=None,
                 previous_steps=60,
                 max_episode=200,
                 punished=False,
                 unit=5000):
        self.data = DataManager(data, data_path, data_func, previous_steps)
        self.exchange = Exchange(punished=punished, unit=unit)
        self._render = Render()

        self.obs = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(HISTORY_NUM +
                                                   1, FEATURE_NUM),
                                            dtype=np.float32)

    def get_obs_with_history(self, info):
        obs = self.data.recent_history.to_array(
            base=self.data.first_price,
            extend=[info['avg_price']])
        amount = info['amount']
        index = info['buy_at'] / self.data.max_steps
        extends = [amount, index]
        amount = np.ones([HISTORY_NUM + 1, len(extends)]) * extends
        obs = np.concatenate((obs, amount), axis=1)
        return obs

    def _step(self, action):
        if self.obs is None:
            self.obs, done = self.data.step()

        print(action)
        if action in self.exchange.available_actions:
            self._render.take_action(action, self.obs)
        reward = self.exchange.step(action, self.obs)
        info = self.exchange.info

        obs = self.get_obs_with_history(info)
        self.obs, done = self.data.step()

        if self.exchange.is_over_loss:
            done = True

        return obs, reward, done, info

    def step(self, action):
        action = action - 1
        observation, reward, done, info = self._step(action)
        return observation, reward, done, info

    def reset(self, index=None):
        self.data.reset(index)
        self.exchange.reset()
        self._render.reset()
        observation, reward, done, info = self._step(
            self.exchange.start_action)
        return observation

    def render(self):
        history = self.data.history
        info = self.exchange.info
        self._render.render(history, info)

    def close(self):
        return
