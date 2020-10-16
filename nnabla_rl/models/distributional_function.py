from nnabla_rl.models.model import Model


class ValueDistributionFunction(Model):
    def probabilities(self, s):
        raise NotImplementedError


class QuantileDistributionFunction(Model):
    def quantiles(self, s):
        raise NotImplementedError


class StateActionQuantileFunction(Model):
    def quantiles(self, s, tau):
        pass
