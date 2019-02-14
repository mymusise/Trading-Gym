import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_finance import candlestick_ochl
from .exchange import ACTION


class Arrow(object):

    def __init__(self, coord, color):
        self.coord = coord
        self.color = color


class Render(object):

    def __init__(self, bar_width=0.8):
        plt.ion()
        self.figure, self.ax = plt.subplots()
        self.figure.set_size_inches((12, 6))
        self.bar_width = 0.8
        self.arrow_body_len = self.bar_width / 1000
        self.arrow_head_len = self.bar_width / 20
        self.arrow_width = self.bar_width
        self.max_windows = 60

        self.arrows = []

    @property
    def arrow_len(self):
        return self.arrow_body_len + self.arrow_head_len

    def take_action(self, action, observation):
        if action == ACTION.PUT:
            y_s = observation.high + self.arrow_body_len + self.arrow_head_len
            y_e = - self.arrow_body_len
            color = 'gray'
        elif action == ACTION.PUSH:
            y_s = observation.high
            y_e = self.arrow_body_len
            color = 'black'
        else:
            return
        x = observation.index
        self.arrows.append(Arrow((x, y_s, 0, y_e), color))

    def draw_arrow(self, data):
        first = data[0]
        arrows = list(filter(lambda x: x.coord[0] > first[0], self.arrows))
        for arrow in arrows:
            self.ax.arrow(*arrow.coord,
                          color=arrow.color,
                          head_width=self.arrow_width,
                          head_length=self.arrow_head_len)

        self.arrows = arrows

    def draw_title(self, info):
        formatting = """
        profit: {} \t fixed-profit: {} \t floating-profit:{} \t
        amount: {} \t latest-price: {} \t time: {} \n
        """

        def r(x): return round(x, 2)
        title = formatting.format(r(info['profit']['total']),
                                  r(info['profit']['fixed']),
                                  r(info['profit']['floating']),
                                  info['amount'],
                                  r(info['latest_price']),
                                  info['date'])
        plt.suptitle(title)

    def xaxis_format(self, history):
        def formator(x, pos=None):
            for h in history:
                if h.index == x:
                    return h.date.strftime("%H:%M")
            return ""
        return formator

    def render(self, history, info, mode='human', close=False):
        """
            If it plot one by one will cache many point
            that will cause OOM and work slow.
        """
        self.ax.clear()
        history = history[-self.max_windows:]
        data = [obs.to_list() for obs in history]
        candlestick_ochl(self.ax, data, colordown='g',
                         colorup='r', width=self.bar_width)
        self.draw_arrow(data)

        self.ax.xaxis.set_major_formatter(
            FuncFormatter(self.xaxis_format(history)))
        self.ax.set_ylabel('price')
        self.figure.autofmt_xdate()

        self.draw_title(info)
        plt.show()
        plt.pause(0.0001)

    def reset(self):
        self.ax.clear()
        self.arrows = []
