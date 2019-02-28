from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
import argparse
from base import init_logger, Base


def normalize(data, base=100):
    return [x / base for x in data]


def test_fake(retrain, render):
    data_path = '/data/money/fake_sin_data.json'
    punished = retrain
    if not retrain:
        init_logger()
    times = 1 if retrain else 1
    for i in range(times):
        trainer = Base()
        info = trainer.train(data_path, DQN, MlpPolicy,
                             retrain=punished,
                             render=render,
                             train_steps=200000,
                             save_path='fake',
                             env_params={
                                 'punished': False,
                                 'nav': 50000,
                                 'data_kwargs': {
                                     'use_ta': True,
                                     'start_random': False
                                 },
                             },
                             rl_model_params={
                                 'verbose': 1,
                                 'learning_rate': 5e-5
                             })
        profit = info[0]['profit']['total']
        if profit > 10000:
            break
    # trainer.plot_results() # have some problem


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--retrain', action="store_true", default=False)
    parser.add_argument('--render', action="store_true", default=False)

    args = parser.parse_args()

    test_fake(args.retrain, args.render)
