import matplotlib.pyplot as plt
import numpy as np
import json
import datetime


def get_sin_data():
    Fs = 200
    f = 5
    sample = 8000
    x = np.arange(sample)
    random_a = np.random.rand(sample) / 2 + 1.5
    random_b = (np.random.rand(sample) + 0.5)
    y = (np.sin(2 * np.pi * f * x / Fs) + random_a) + 50
    return x, y


def draw(x, y):
    plt.plot(x, y)
    plt.xlabel('sample(n)')
    plt.ylabel('voltage(V)')
    plt.show()


def export(y):
    data = []
    now = datetime.datetime.now()
    minute = datetime.timedelta(seconds=60)
    for i, _y in enumerate(y):
        time = now + i * minute
        item = {}
        item['open'] = _y
        item['close'] = _y
        item['high'] = _y
        item['low'] = _y
        item['volume'] = 1000
        item['date'] = time.strftime("%Y-%m-%d %H:%M")
        data.append(item)
    with open('/data/money/fake_sin_data.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    x, y = get_sin_data()
    export(y)
    draw(x, y)
