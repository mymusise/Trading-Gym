from base import train
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
import argparse


def test_fake(retrain, render):
    data_path = '/data/money/fake_sin_data.json'
    punished = retrain
    train(data_path, DQN, MlpPolicy,
          retrain=retrain,
          render=render,
          train_steps=50000,
          save_path='fake',
          env_params={'punished': punished, 'unit': 10},
          rl_model_params={'verbose': 1})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--retrain', action="store_true", default=False)
    parser.add_argument('--render', action="store_true", default=False)

    args = parser.parse_args()

    test_fake(args.retrain, args.render)
