import matplotlib.pyplot as plt
import numpy as np
import json
import datetime


def get_sin_data(add_noise=True):
    Fs = 200
    f = 5
    sample = 8000
    x = np.arange(sample)
    if add_noise:
        noise = np.random.rand(sample) / 2 + 1.5
    else:
        noise = np.zeros(sample)
    y = (np.sin(2 * np.pi * f * x / Fs) + noise) + 50
    return x, y


def get_sin_linear_data():
    n = 800
    x = np.arange(n)
    noise = (np.random.rand(n) - 0.5)
    y1 = (np.sin(2 * np.pi * x / 30) + noise) / 2
    y2 = np.sin(2 * np.pi * x / 600) + 50
    y = y1 + y2
    return x, y


def draw(x, y):
    plt.plot(x, y)
    plt.xlabel('sample(n)')
    plt.ylabel('voltage(V)')
    plt.show()


def warp_json_data(y):
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
        yield item


def get_fake_json():
    x, y = get_sin_data()
    data = list(warp_json_data(y))
    return data


def export(data):
    with open('/data/money/fake_sin_data.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    x, y = get_sin_data()
    # x, y = get_sin_linear_data()
    data = list(warp_json_data(y))
    export(data)
    draw(x, y)
