class ExtraFeature:
    ex_obs_name = ['amount', 'buy_at', 'index', 'floating_rate', 'math_hour']

    def get_extra_features(self, info):
        features = {}
        features['amount'] = self.exchange.amount
        features['buy_at'] = info['buy_at'] / self.data.max_steps
        features['index'] = info['index'] / self.data.max_steps
        features['floating_rate'] = self.exchange.floating_rate
        features['math_hour'] = self.obs.math_hour

        return [features[name] for name in self.ex_obs_name]
