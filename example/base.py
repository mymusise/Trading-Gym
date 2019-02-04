import sys

sys.path.append("..")


import logging
from stable_baselines.common.vec_env import DummyVecEnv
from trading_gym.env import TradeEnv


def init_logger():
    logger = logging.getLogger('trading-gym')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def train(data_path,
          RLModel,
          Policy,
          retrain=False,
          render=False,
          train_steps=50000,
          save_path='model',
          env_params={},
          rl_model_params={}):
    env = TradeEnv(data_path=data_path, **env_params)
    env = DummyVecEnv([lambda: env])

    if retrain:
        model = RLModel(Policy, env, **rl_model_params)
        model.learn(total_timesteps=train_steps)
        model.save(save_path)
    else:
        model = RLModel.load(save_path)

    obs = env.reset()
    for i in range(5000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if render:
            env.render()
        if dones:
            break

    print(info)
    return info
