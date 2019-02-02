from .base import train
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
import argparse


def test_train(retrain, render):
    data_path = '/data/money/source_minute.json'
    punished = retrain
    train(data_path, DQN, MlpPolicy,
          retrain=retrain,
          train_steps=1000000,
          env_params={'punished': punished})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--retrain', action="store_true", default=False)
    parser.add_argument('--render', action="store_true", default=False)

    args = parser.parse_args()

    test_train(args.retrain, args.render)
