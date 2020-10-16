class RlablaError(Exception):
    """
    Base class of all specific exceptions defined for nnabla_rl.
    """
    pass


class UnsupportedTrainingException(RlablaError):
    """
    Raised when the algorithm does not support requested training procedure.
    """
    pass


class UnsupportedEnvironmentException(RlablaError):
    """
    Raised when the algorithm does not support given environment to train the policy.
    """
    pass
