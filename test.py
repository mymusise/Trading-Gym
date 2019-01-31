from env import TradeEnv
import random
import logging
import pytest
import time


def init_logger():
    logger = logging.getLogger('gym-trading')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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
        print(obs)
        # env.render()
        if done:
            break


@pytest.mark.unit
def test_train(retrain):
    from stable_baselines.deepq.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import DQN

    env = TradeEnv(data_path='/data/money/source_minute.json')
    # env = TradeEnv(data_path='/data/money/fake_sin_data.json')
    env = DummyVecEnv([lambda: env])

    if retrain:
        model = DQN(MlpPolicy, env)
        model.learn(total_timesteps=5000000)
        model.save("a2c_trading")
    else:
        model = DQN.load("a2c_trading")

    init_logger()
    obs = env.reset()
    for i in range(5000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # env.render()
        if dones:
            break


if __name__ == '__main__':
    # test_render()
    test_train(False)
