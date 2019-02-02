from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN, A2C
from trading_gym.env import TradeEnv
import random
import logging
import pytest
import time


def init_logger():
    logger = logging.getLogger('trading-gym')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def test_exchange():
    init_logger()
    env = TradeEnv(data_path='./data/test_exchange.json',
                   punished=False, unit=1000)
    obs = env.reset(index=0)
    obs, reward, done, info1 = env.step(2)
    obs, reward, done, info2 = env.step(0)
    p1 = round(info2['profit']['total'], 1)
    rate1 = (info2['latest_price'] - info1['latest_price']) / info1['latest_price']
    assert p1 == round(rate1 * info2['unit'], 1) - 6
    obs, reward, done, info3 = env.step(1)
    obs, reward, done, info4 = env.step(1)

    obs, reward, done, info5 = env.step(2)
    obs, reward, done, info6 = env.step(0)
    print(info4, info5, info6)
    rate2 = (info6['latest_price'] - info5['latest_price']) / info5['latest_price']
    p2 = round(info6['profit']['total'], 1)
    assert p2 - p1 == round(rate2 * info6['unit'], 1) - 6


def test_diff_model():
    env = TradeEnv(data_path='/data/money/fake_sin_data.json')
    env = DummyVecEnv([lambda: env])

    obs = env.reset()

    model = DQN(DQNMlpPolicy, env)
    for i in range(20):
        print(model.predict(obs))

    model = A2C(MlpPolicy, env)
    for i in range(20):
        print(model.predict(obs, deterministic=True))


def test_render():
    # env = TradeEnv(data_path='/data/money/source_minute.json')
    env = TradeEnv(data_path='/data/money/fake_sin_data.json')
    action = 0
    done = False
    obs = env.reset()
    init_logger()
    for i in range(2000):
        action = random.sample([0, 1, 2], 1)[0]
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break


def train(data_path,
          RLModel,
          retrain,
          train_steps=100000,
          save_path='model',
          env_params={}):
    env = TradeEnv(data_path=data_path, **env_params)
    # env = TradeEnv(data_path='/data/money/fake_sin_data.json')
    env = DummyVecEnv([lambda: env])

    if retrain:
        model = RLModel(DQNMlpPolicy, env)
        model.learn(total_timesteps=train_steps)
        model.save(save_path)
    else:
        model = RLModel.load(save_path)

    init_logger()
    obs = env.reset()
    for i in range(5000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # env.render()
        if dones:
            break


@pytest.mark.unit
def test_fake(retrain):
    data_path = '/data/money/fake_sin_data.json'
    train(data_path, DQN, retrain, train_steps=500000,
          save_path='fake', env_params={'punished': True, 'unit': 1000})


@pytest.mark.unit
def test_train(retrain):
    data_path = '/data/money/source_minute.json'
    train(data_path, DQN, retrain, train_steps=1000000,
          env_params={'punished': True})


if __name__ == '__main__':
    # test_render()
    test_train(False)
