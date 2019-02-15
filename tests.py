from trading_gym import ACTION, Observation, Exchange, TradeEnv
import random


def test_exchange():
    nav = 1000
    ex = Exchange(nav=nav)
    charge = ex.get_charge(ACTION.PUSH, 10)
    ex.step(ACTION.PUSH, Observation(close=10))
    assert ex.profit == -charge
    assert ex.fixed_profit == -charge

    charge += ex.get_charge(ACTION.HOLD, 11)
    ex.step(ACTION.HOLD, Observation(close=11))
    assert ex.floating_profit == nav * 0.1
    assert ex.profit == -charge + nav * 0.1

    charge += ex.get_charge(ACTION.PUT, 12)
    ex.step(ACTION.PUT, Observation(close=12))
    assert ex.profit == -charge + nav * 0.2


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


def test_custom_reward_func():
    def reward_fn(exchange):
        return 2
    env = TradeEnv(data_path='./data/test_exchange.json',
                   get_reward_func=reward_fn)
    env.reset()
    obs, reward, done, info = env.step(ACTION.PUSH)
    assert reward == 2

    def reward_fn(exchange):
        return exchange.floating_profit
    env = TradeEnv(data_path='./data/test_exchange.json',
                   get_reward_func=reward_fn)
    env.reset()
    obs, reward, done, info = env.step(ACTION.PUSH)
    obs, reward, done, info = env.step(ACTION.HOLD)
    assert reward == env.exchange.floating_profit


if __name__ == '__main__':
    test_render()
