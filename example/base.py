import sys

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import logging
import shutil
from stable_baselines.common.vec_env import DummyVecEnv
from trading_gym.env import TradeEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy


def init_logger():
    logger = logging.getLogger('trading-gym')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Base(object):

    @property
    def best_model_file(self):
        return self.log_folder + 'best_model.pkl'

    def __init__(self):
        self.best_mean_reward, self.n_steps = -np.inf, 0
        self.log_folder = 'logs/'
        self.log_file = self.log_folder + 'model'

    def movingAverage(self, values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')

    def plot_results(self, title='Learning Curve'):
        """
        plot the results

        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(self.log_folder), 'timesteps')
        y = self.movingAverage(y, window=50)
        # Truncate x
        x = x[len(x) - len(y):]

        plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        plt.show()

    def callback(self, _locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        # Print stats every 1000 calls
        if (self.n_steps + 1) % 1000 == 0:
            # Evaluate policy performance
            x, y = ts2xy(load_results(self.log_folder), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(self.best_model_file)
        self.n_steps += 1
        # Returning False will stop training early
        return True

    def train(self, data_path,
              RLModel,
              Policy,
              retrain=False,
              render=False,
              train_steps=50000,
              save_path='model',
              env_params={},
              rl_model_params={}):
        env = TradeEnv(data_path=data_path, **env_params)
        env = Monitor(env, self.log_file, allow_early_resets=True)
        env = DummyVecEnv([lambda: env])

        if retrain:
            try:
                shutil.rmtree('./logs/model')
            except Exception:
                pass
            model = RLModel(
                Policy, env, **rl_model_params)
            model.learn(total_timesteps=train_steps, callback=self.callback)
            model.save(save_path)

        model = RLModel.load(self.best_model_file)

        obs = env.reset()
        for i in range(8000):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if render:
                env.render()
            if done:
                break

        print(info)
        return info
