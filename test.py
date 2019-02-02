from trading_gym.env import TradeEnv
import random


def test_exchange():
    env = TradeEnv(data_path='./data/test_exchange.json',
                   punished=False, unit=1000)
    obs = env.reset(index=0)
    obs, reward, done, info1 = env.step(2)
    obs, reward, done, info2 = env.step(0)
    p1 = round(info2['profit']['total'], 1)
    rate1 = (info2['latest_price'] - info1['latest_price']) / \
        info1['latest_price']
    assert p1 == round(rate1 * info2['unit'], 1) - 6
    obs, reward, done, info3 = env.step(1)
    obs, reward, done, info4 = env.step(1)

    obs, reward, done, info5 = env.step(2)
    obs, reward, done, info6 = env.step(0)
    print(info4, info5, info6)
    rate2 = (info6['latest_price'] - info5['latest_price']) / \
        info5['latest_price']
    p2 = round(info6['profit']['total'], 1)
    assert p2 - p1 == round(rate2 * info6['unit'], 1) - 6


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
