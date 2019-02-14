# Trading-Gym


![Build Status](https://travis-ci.org/mymusise/Trading-Gym.svg?branch=master)


## install 

```
pip install trading-gym
```

## input format

```
[
    {
        "open": 10.0,
        "close": 10.0,
        "high": 10.0,
        "low": 10.0,
        "volume": 10.0,
        "date": "2019-01-01 10:00"
    },
    {
        "open": 10.1,
        "close": 10.1,
        "high": 10.1,
        "low": 10.1,
        "volume": 10.1,
        "date": "2019-01-01 10:00"
    }
]
```


## actions

| Action | Value |
| ------ | ----- |
| PUT    | 0     |
| HOLD   | 1     |
| PUSH   | 2     |


## observation



# Examples



## quick start

```
from trading_gym.env import TradeEnv
import random


env = TradeEnv(data_path='./data/test_exchange.json')
done = False
obs = env.reset()
for i in range(500):
    action = random.sample([0, 1, 2], 1)[0]
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
```

## A sample train with stable-baselines

```
from trading_gym.env import TradeEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy


data_path = './data/fake_sin_data.json'
env = TradeEnv(data_path=data_path, unit=50000, use_ta=True)
env = DummyVecEnv([lambda: env])

model = DQN(MlpPolicy, env, verbose=2, learning_rate=1e-5)
model.learn(200000)


obs = env.reset()
for i in range(8000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if render:
        env.render()
    if done:
        break
```
