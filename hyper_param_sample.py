"""
Sampling hyper parameters
"""

import numpy as np
import random
from collections import OrderedDict

# domain range and configs
CONSTS = OrderedDict()

CONSTS['conv_layer_n'] = {
    'values': [2, 3],
}
CONSTS['fold'] = {
    'values': [0, 1], 
    'depends_on': 'conv_layer_n'
}
CONSTS['dr'] = {
    'values': [0.5], 
    'depends_on': 'conv_layer_n'
}
CONSTS['ext_ebd'] = {
    'values': [True, False], 
}
CONSTS['batch_size'] = {
    'values': [9, 10, 11, 12], 
}
CONSTS['ebd_dm'] = {
    'values': [48]
}

def coin_toss(p = 0.5):
    return np.random.binomial(n = 1, p = p, size = (1, ))


# semi-random ones
SEMI_RANDOM_PARAMS = {
    'ks': {
        2: (20, 5), 
        3: (20, 10, 5)
    }, 
    'nkerns': {
        2: (6, 12), 
        3: (5, 10, 18)
    }, 
    'filter_widths': {
        2: (10, 7), 
        3: (6, 5, 3)
    }, 
    'l2_regs': {
        2: (1e-4, 3e-5, 3e-5, 1e-4),
        3: (1e-4, 3e-5, 3e-6, 1e-5, 1e-4),
    }
}

def sample_params(n = 16, semi_random_params_key = 'conv_layer_n'):
    possibility_n = np.product([len(i['values']) for i in CONSTS.values()])
    assert n <= possibility_n, "%d <= %d ?" %(n, possibility_n)
    
    pool = set()
    samples = []
    i = 0
    
    while i <= n:
        # random hyper parameters        
        params = {}
        for key in CONSTS:
            depends_on = CONSTS[key].get('depends_on')
            value = random.choice(CONSTS[key]['values'])
            if depends_on:
                dup_times = params[depends_on]
                params[key] = tuple([value]) * dup_times
            else:
                if isinstance(value, bool): #it's bool, show or hide
                    if value:
                        params[key] = value
                else:
                    params[key] = value
        
        for key in SEMI_RANDOM_PARAMS:
            params[key] = SEMI_RANDOM_PARAMS[key][params[semi_random_params_key]]
            
        if tuple(params.values()) in pool:
            continue
        else:
            i += 1
            pool.add(tuple(params.values()))
            samples.append(params)

    return samples

def _format_value(v, tuple_sep = ' '):
    if isinstance(v, tuple):
        return tuple_sep.join(map(str, v))
    elif isinstance(v, bool):
        return ''
    else:
        return str(v)

def format_params_to_cmd(params, prefix = "python cnn4nlp.py --l2  --norm_w --ebd_delay_epoch=0 --au=tanh "):
    params_str = params2str(params)
    sig = params2str(params, cmd_sep = ',,', key_val_sep = '=', tuple_sep = ',', key_prefix = '')
    return "%s %s --img_prefix=%s"%(
        prefix, params_str, sig
    )

def params2str(params, cmd_sep = ' ',key_val_sep = ' ', tuple_sep = ' ', key_prefix = '--'):
    return cmd_sep.join(["%s%s%s%s"  %(key_prefix, 
                                       key, 
                                       key_val_sep, 
                                       _format_value(value, tuple_sep = tuple_sep))
                         for key, value in params.items()])
    
if __name__ ==  "__main__":
    possibility_n = np.product(map(len, CONSTS.values()))
    print "possibility_n = %d" %(possibility_n)
    n = 16
    for param in sample_params(n):
        print format_params_to_cmd(param)
