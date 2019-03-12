import logging
import pandas as pd
from collections import defaultdict

logger = logging.getLogger('trading-gym')


class ACTION:
    PUT = -1
    HOLD = 0
    PUSH = 1


class Position:

    def __init__(self, symbol=''):
        self.symbol = symbol
        self.amount = 0
        self.avg_price = 0
        self.created_index = 0

    @property
    def principal(self):
        return self.avg_price * self.amount

    @property
    def is_empty(self):
        return self.amount == 0

    @property
    def is_do_long(self):
        return self.amount > 0

    @property
    def is_do_short(self):
        return self.amount < 0

    @property
    def rate(self):
        if self.is_do_short:
            diff = abs(self.avg_price) - self.latest_price
        else:
            diff = self.latest_price - abs(self.avg_price)
        return diff / self.avg_price if self.avg_price else 0

    def get_profit(self, latest_price):
        if self.is_empty:
            return 0
        else:
            profit = self.rate * abs(self.principal)
            return profit

    def update(self, action, latest_price):
        self.latest_price = latest_price

    def step(self, action, nav, index):
        if action != 0 and self.amount == 0:
            self.created_index = index
        if action * self.amount < 0:
            profit = self.get_profit(self.latest_price)
            self.avg_price = 0 
            self.created_index = 0
            self.amount = 0
        else:
            amount = int(action * nav / self.latest_price)
            total = self.avg_price * self.amount + amount * self.latest_price
            self.amount += amount
            self.avg_price = total / self.amount if self.amount else 0
            profit = 0

        return profit


class Transaction(object):
    _columns = ["index", "time", "latest_price", "action", "amount", "profit",
                "fixed_profit", "floating_profit"]

    def init__transaction(self):
        self.transaction = pd.DataFrame(columns=self._columns)

    def add_transaction(self, obs, action, nav, symbol=''):
        current = pd.DataFrame({
            'index': [obs.index],
            'symbol': [symbol],
            'time': [obs.date],
            'month': [obs.date.strftime('%y-%m')],
            'latest_price': [obs.latest_price],
            'action': [action],
            'nav': [nav],
            'profit': [self.profit],
            'fixed_profit': [self.fixed_profit],
            'floating_profit': [self.floating_profit]
        })
        self.transaction = self.transaction.append(current, sort=False)

    def report(self):
        self.transaction = self.transaction.sort_values(['time', 'symbol'])
        months = self.transaction.groupby('month')
        month_profit = defaultdict(float)
        print(self.transaction)
        for month, cloumns in months:
            diff = cloumns.iloc[-1].fixed_profit - cloumns.iloc[0].fixed_profit
            month_profit[month] = diff / self.nav
        print("\nMonth_result:")
        for month, rate in month_profit.items():
            print(month, rate)


class Exchange(Transaction):

    """

    Attributes:
        fixed_profit (int): fixed_profit
        floating_profit (int): floating_profit
        nav (int): nav
        observation (Observation): observation
        position (Position): position
        punished (bool): punished
        unit (int): unit
    """

    def __init__(self,
                 nav=50000,
                 end_loss=None,
                 unit=5000,
                 punished=False,
                 punish_func=None):
        """

        Args:
            punished (bool, optional): Do punishe if True
            nav (int, optional): nav
            end_loss (None, optional): end_loss
            unit (int, optional): unit
        """
        self.nav = nav
        self.punished = punished
        self.punish_func = punish_func
        self._end_loss = end_loss
        self.unit = unit
        self.__init_data()

    def __init_data(self):
        self.position = Position()
        self.fixed_profit = 0
        self.init__transaction()

    @property
    def start_action(self):
        return ACTION.HOLD

    @property
    def available_funds(self):
        return self.nav - self.position.principal

    @property
    def floating_rate(self):
        return self.position.rate

    @property
    def profit(self):
        return self.fixed_profit + self.floating_profit

    @property
    def available_actions(self):
        if self.position.is_do_short:
            return [ACTION.PUSH, ACTION.HOLD]
        elif self.position.is_do_long:
            return [ACTION.PUT, ACTION.HOLD]
        else:
            return [ACTION.PUT, ACTION.HOLD, ACTION.PUSH]

    @property
    def punished_action(self):
        if self.position.is_do_short:
            return ACTION.PUT
        elif self.position.is_do_long:
            return ACTION.PUSH
        return None

    @property
    def cost_action(self):
        return [ACTION.PUT, ACTION.PUSH]

    @property
    def end_loss(self):
        if self._end_loss is not None:
            return self._end_loss
        return - self.nav * 0.2

    @property
    def is_over_loss(self):
        return self.profit < self.end_loss

    @property
    def amount(self):
        return self.position.amount / self.nav

    @property
    def floating_profit(self):
        return self.position.get_profit(self.latest_price)

    # def get_profit(self, observation, symbol='default'):
    #     latest_price = observation.latest_price
    #     positions = self.positions.values()
    #     positions_profite = sum([position.get_profit(
    #         latest_price) for position in positions])
    #     return self.profit + positions_profite

    def get_charge(self, action, latest_price):
        """
            rewrite if inneed.
        """
        if self.position.is_empty and action in self.cost_action:
            amount = self.unit / latest_price
            if amount < 100:
                return 2
            else:
                return amount * (0.0039 + 0.0039)
        else:
            return 0

    def get_punished(self, action, observation):
        if self.punish_func is None:
            return 0
        else:
            return self.punish_func(self, action, observation)

    def step(self, action, observation, symbol='default'):
        self.observation = observation
        self.latest_price = observation.latest_price
        self.position.update(action, self.latest_price)

        charge = self.get_charge(action, self.latest_price)
        if action in self.available_actions:
            fixed_profit = self.position.step(
                action, self.unit, observation.index)
            fixed_profit -= charge
            self.fixed_profit += fixed_profit

            self.add_transaction(observation, action, self.unit)
        else:
            fixed_profit = 0

        reward = self.fixed_profit / self.nav
        if self.punished:
            reward += self.get_punished(action, observation)

        logger.info(f"latest_price:{observation.latest_price}, "
                    f"amount:{self.position.amount}, "
                    f"action:{action}")
        logger.info(f"fixed_profit: {self.fixed_profit}, "
                    f"floating_profit: {self.floating_profit}")
        return reward

    def reset(self):
        self.__init_data()

    @property
    def info(self):
        return {
            'index': self.observation.index,
            'date': self.observation.date_string,
            'nav': self.nav,
            'amount': self.position.amount,
            'avg_price': self.position.avg_price,
            'profit': {
                'total': self.profit,
                'fixed': self.fixed_profit,
                'floating': self.floating_profit
            },
            'buy_at': self.position.created_index,
            'latest_price': self.observation.latest_price,
        }


class MultiExchange(Exchange):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positions = {}

    @property
    def floating_profit(self):
        return 0

    @property
    def available_funds(self):
        used_nav = sum([abs(positon.principal)
                        for positon in self.positions.values()])
        return self.nav - used_nav

    def get_position(self, symbol):
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions.get(symbol)

    def check_action_enable(self, action, position):
        if self.available_funds < self.nav * 0.8:
            if position.is_do_long:
                enable_action = [ACTION.PUT, ACTION.HOLD]
            elif position.is_do_short:
                enable_action = [ACTION.PUSH, ACTION.HOLD]
            else:
                enable_action = [ACTION.PUSH, ACTION.PUT, ACTION.HOLD]
            return action in enable_action
        else:
            return True

    def step(self, action, observation, symbol):
        self.observation = observation
        self.latest_price = observation.latest_price

        position = self.get_position(symbol)
        position.update(action, self.latest_price)

        charge = self.get_charge(action, self.latest_price)
        if self.check_action_enable(action, position):
            fixed_profit = position.step(
                action, self.unit, observation.index)
            fixed_profit -= charge
            self.fixed_profit += fixed_profit

            self.add_transaction(observation, action, self.unit, symbol)
        else:
            fixed_profit = 0

    @property
    def info(self):
        pass
