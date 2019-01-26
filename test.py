from env import TradeEnv
import time
import random
import logging


def init_logger():
    logger = logging.getLogger('gym-trading')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def test_render():
    env = TradeEnv(data_path='/data/money/source_minute.json')
    action = 0
    done = False
    obs = env.reset()
    while not done:
        action = random.sample([-1, 0, 1], 1)[0]
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    time.sleep(1)


def test_train(retrain=False):
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import A2C

    env = TradeEnv(data_path='/data/money/source_minute.json')
    env = DummyVecEnv([lambda: env])
    model = A2C(MlpPolicy, env, ent_coef=0.1, verbose=1)

    if retrain:
        model.learn(total_timesteps=100000)
        model.save("a2c_trading")

    init_logger()
    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    # test_render()
    test_train(True)
