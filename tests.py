from trading_gym import ACTION, Observation, Exchange, TradeEnv
import random


def test_exchange():
    unit = 1000
    ex = Exchange(unit=unit)
    charge = ex.get_charge(ACTION.PUSH, 10)
    ex.step(ACTION.PUSH, Observation(close=10))
    assert ex.profit == -charge
    assert ex.fixed_profit == -charge

    charge += ex.get_charge(ACTION.HOLD, 11)
    ex.step(ACTION.HOLD, Observation(close=11))
    assert ex.floating_profit == unit * 0.1
    assert ex.profit == -charge + unit * 0.1

    charge += ex.get_charge(ACTION.PUT, 12)
    ex.step(ACTION.PUT, Observation(close=12))
    assert ex.profit == -charge + unit * 0.2


def test_render():
    env = TradeEnv(data_path='./data/test_exchange.json')
    action = 0
    done = False
    obs = env.reset()
    for i in range(500):
        action = random.sample([0, 1, 2], 1)[0]
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == '__main__':
    test_render()
