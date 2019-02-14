from trading_gym.env import Exchange, Observation, TradeEnv
import random


def test_exchange():
    unit = 1000
    ex = Exchange(unit=unit)
    charge = ex.get_charge(Exchange.ACTION.PUSH, 10)
    ex.step(Exchange.ACTION.PUSH, Observation(close=10))
    assert ex.profit == -charge
    assert ex.fixed_profit == -charge

    charge += ex.get_charge(Exchange.ACTION.HOLD, 11)
    ex.step(Exchange.ACTION.HOLD, Observation(close=11))
    assert ex.floating_profit == unit * 0.1
    assert ex.profit == -charge + unit * 0.1

    charge += ex.get_charge(Exchange.ACTION.PUT, 12)
    ex.step(Exchange.ACTION.PUT, Observation(close=12))
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
