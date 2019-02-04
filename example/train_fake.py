from base import train
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
import talib
import argparse
import numpy as np
from base import init_logger


def get_technique(_open, close, high, low, volume, timeperiod):
    upperband, middleband, lowerband = talib.BBANDS(
        close, timeperiod=timeperiod, nbdevup=2, nbdevdn=2, matype=0)
    ma = talib.MA(close, timeperiod=timeperiod)
    ema = talib.EMA(close, timeperiod=timeperiod)
    dema = talib.DEMA(close, timeperiod=timeperiod)
    wma = talib.WMA(close, timeperiod=timeperiod)
    sma = talib.SMA(close, timeperiod=timeperiod)
    tema = talib.TEMA(close, timeperiod=timeperiod)
    trima = talib.TRIMA(close, timeperiod=timeperiod)
    sar = talib.SAR(high, low)

    adx = talib.ADX(high, low, close, timeperiod=timeperiod)
    adxr = talib.ADXR(high, low, close, timeperiod=timeperiod)
    apo = talib.APO(close, fastperiod=timeperiod / 2,
                    slowperiod=timeperiod, matype=0)
    cmo = talib.CMO(close, timeperiod=timeperiod)
    dx = talib.DX(high, low, close, timeperiod=timeperiod)
    macd, macdsignal, macdhist = talib.MACD(
        close,
        fastperiod=timeperiod / 2,
        slowperiod=timeperiod,
        signalperiod=timeperiod / 2 - 1)
    macdfix, macdfixsignal, macdfixhist = talib.MACDFIX(
        close, signalperiod=timeperiod - 2)
    mfi = talib.MFI(high, low, close, volume, timeperiod=timeperiod)
    di = talib.MINUS_DI(high, low, close, timeperiod=timeperiod)

    ad = talib.AD(high, low, close, volume)
    adosc = talib.ADOSC(high, low, close, volume,
                        fastperiod=3, slowperiod=timeperiod)
    obv = talib.OBV(close, volume)
    obs = np.nan_to_num(
        [ma[-1], ema[-1], dema[-1], wma[-1], sma[-1], tema[-1],
         trima[-1], sar[-1], apo[-1], adosc[-1], obv[-1], macd[-1],
         macdsignal[-1], macdhist[-1], macdfix[-1], macdfixsignal[-1],
         macdfixhist[-1], upperband[-1], middleband[-1], lowerband[-1]])
    return obs


def get_obs_with_talib(history, *args):
    _open = np.array([obs.open for obs in history.obs_list])
    close = np.array([obs.close for obs in history.obs_list])
    high = np.array([obs.high for obs in history.obs_list])
    low = np.array([obs.low for obs in history.obs_list])
    volume = np.array([float(obs.volume) for obs in history.obs_list])

    periods = [10, 15, 20, 30]
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
    for i in range(20):
        info = train(data_path, DQN, MlpPolicy,
                     retrain=retrain,
                     render=render,
                     train_steps=20000,
                     save_path='fake',
                     env_params={
                         'punished': punished,
                         'unit': 50000,
                         'get_obs_features_func': get_obs_with_talib,
                         'ops_shape': [4, 20],
                     },
                     rl_model_params={'verbose': 1})
        profit = info[0]['profit']['total']
        if profit > 10000:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--retrain', action="store_true", default=False)
    parser.add_argument('--render', action="store_true", default=False)

    args = parser.parse_args()

    test_fake(args.retrain, args.render)
