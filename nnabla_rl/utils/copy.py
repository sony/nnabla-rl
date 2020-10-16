

def copy_network_parameters(origin_params, target_params, tau=1.0):
    
    if not ((0.0 <= tau) & (tau <= 1.0)):
            raise ValueError('tau must lie between [0.0, 1.0]')

    for key in target_params.keys():
        target_params[key].data.copy_from(
            origin_params[key].data * tau \
            + target_params[key].data * (1 - tau))