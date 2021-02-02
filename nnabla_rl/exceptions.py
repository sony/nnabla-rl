class NNablaRLError(Exception):
    '''
    Base class of all specific exceptions defined for nnabla_rl.
    '''
    pass


class UnsupportedTrainingException(NNablaRLError):
    '''
    Raised when the algorithm does not support requested training procedure.
    '''
    pass


class UnsupportedEnvironmentException(NNablaRLError):
    '''
    Raised when the algorithm does not support given environment to train the policy.
    '''
    pass
