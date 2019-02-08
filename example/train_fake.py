from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
import talib
import argparse
import numpy as np
from base import init_logger, Base


def normalize(data, base=100):
    return [x / base for x in data]


def get_technique(_open, close, high, low, volume, timeperiod):
    n_close = np.array(normalize(close))
    n_high = np.array(normalize(high))
    n_low = np.array(normalize(low))
    n_volume = np.array(normalize(volume))
    upperband, middleband, lowerband = talib.BBANDS(
        n_close, timeperiod=timeperiod, nbdevup=2, nbdevdn=2, matype=0)
    ma = talib.MA(n_close, timeperiod=timeperiod)
    ema = talib.EMA(n_close, timeperiod=timeperiod)
    dema = talib.DEMA(n_close, timeperiod=timeperiod)
    wma = talib.WMA(n_close, timeperiod=timeperiod)
    sma = talib.SMA(n_close, timeperiod=timeperiod)
    tema = talib.TEMA(n_close, timeperiod=timeperiod)
    trima = talib.TRIMA(n_close, timeperiod=timeperiod)
    sar = talib.SAR(n_high, n_low)

    apo = talib.APO(close, fastperiod=timeperiod / 2,
                    slowperiod=timeperiod, matype=0)
    macd, macdsignal, macdhist = talib.MACD(
        close,
        fastperiod=timeperiod / 2,
        slowperiod=timeperiod,
        signalperiod=timeperiod / 2 - 1)
    macdfix, macdfixsignal, macdfixhist = talib.MACDFIX(
        close, signalperiod=timeperiod - 2)

    adosc = talib.ADOSC(n_high, n_low, n_close, n_volume,
                        fastperiod=3, slowperiod=timeperiod)
    obv = talib.OBV(n_close, n_volume)
    # return np.nan_to_num([macd[-1], macdsignal[-1]])
    obs = np.nan_to_num(
        [ma[-1], ema[-1], dema[-1], wma[-1], sma[-1], tema[-1],
         trima[-1], sar[-1], apo[-1], adosc[-1], obv[-1], macd[-1],
         macdsignal[-1], macdhist[-1], macdfix[-1], macdfixsignal[-1],
         macdfixhist[-1], upperband[-1], middleband[-1], lowerband[-1]])
    return obs


def get_obs_with_talib(history, *args):
    _open = np.array(([obs.open for obs in history.obs_list]))
    close = np.array(([obs.close for obs in history.obs_list]))
    high = np.array(([obs.high for obs in history.obs_list]))
    low = np.array(([obs.low for obs in history.obs_list]))
    volume = np.array(([float(obs.volume)
                                 for obs in history.obs_list]))

    periods = [10, 15, 20]
    obs = []
    for period in periods:
        features = get_technique(_open, close, high, low, volume, period)
        obs.append(features)
    return obs


def test_fake(retrain, render):
    data_path = '/data/money/fake_sin_data.json'
    punished = retrain
    if not retrain:
        init_logger()
    times = 1 if retrain else 1
    for i in range(times):
        trainer = Base()
        info = trainer.train(data_path, DQN, MlpPolicy,
                             retrain=retrain,
                             render=render,
                             train_steps=200000,
                             save_path='fake',
                             env_params={
                                 'punished': False,
                                 'unit': 50000,
                                 'get_obs_features_func': get_obs_with_talib,
                                 'ops_shape': [3, 20],
                                 'start_random': False,
                             },
                             rl_model_params={'verbose': 1, 'learning_rate':5e-7})
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
