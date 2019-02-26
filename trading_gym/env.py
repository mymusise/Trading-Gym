import numpy as np
from gym.core import GoalEnv
from gym import spaces
from .exchange import Exchange
from .render import Render
from .inputs import DataManager, ExtraFeature
import logging


logger = logging.getLogger('trading-gym')


class TradeEnv(GoalEnv, ExtraFeature):

    def __init__(self,
                 data=None,
                 data_path=None,
                 data_func=None,
                 previous_steps=60,
                 max_episode=200,
                 punished=False,
                 nav=5000,
                 get_obs_features_func=None,
                 ops_shape=None,
                 get_reward_func=None,
                 start_random=False,
                 use_ta=False,
                 ta_timeperiods=None,
                 add_extra=False,
                 history_num=50):
        self.data = DataManager(
            data, data_path, data_func, previous_steps,
            start_random=start_random,
            use_ta=use_ta,
            ta_timeperiods=ta_timeperiods,
            history_num=history_num)
        self.exchange = Exchange(punished=punished, nav=nav)
        self._render = Render()

        self.get_obs_features_func = get_obs_features_func
        self.get_reward_func = get_reward_func
        self.use_ta = use_ta
        self.add_extra = add_extra

        if self.get_obs_features_func is not None:
            if ops_shape is None:
                raise ValueError(
                    "ops_shape should be given if use get_obs_features_func")
            self.ops_shape = ops_shape
        elif self.use_ta:
            self.ops_shape = self.data.ta_data.feature_space
        else:
            self.ops_shape = self.data.default_space

        if len(self.ops_shape) == 2 and self.add_extra:
            self.ops_shape[-1] = self.ops_shape[-1] + len(self.ex_obs_name)

        self.obs = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=self.ops_shape,
                                            dtype=np.float32)

    def get_obs(self, info):
        history = self.data.recent_history

        if self.get_obs_features_func is not None:
            obs = self.get_obs_features_func(history, info)
        elif self.use_ta:
            obs = self.data.ta_features
        else:
            obs = self.data.recent_history.to_array(
                base=self.data.first_price,
                extend=[info['avg_price']])

        if self.add_extra:
            extra_obs = self.get_extra_features(info)
            if len(obs.shape) == 2:
                extra_obs = np.ones([obs.shape[0], len(extra_obs)]) * extra_obs
                obs = np.concatenate((obs, extra_obs), axis=1)
            else:
                obs = np.concatenate((obs, extra_obs))
        return obs

    def get_reward(self, profit):
        if self.get_reward_func is not None:
            return self.get_reward_func(self.exchange)
        else:
            return profit

    def _step(self, action):
        if self.obs is None:
            self.obs, done = self.data.step()

        if action in self.exchange.available_actions:
            self._render.take_action(action, self.obs)

        profit = self.exchange.step(action, self.obs)
        reward = self.get_reward(profit)

        info = self.exchange.info

        obs = self.get_obs(info)
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

    def render(self, mode='human', close=False):
        history = self.data.history
        info = self.exchange.info
        self._render.render(history, info, mode=mode, close=close)

    def close(self):
        return
