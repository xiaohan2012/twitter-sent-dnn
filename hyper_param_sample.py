"""
Sampling hyper parameters
"""

import numpy as np
import random
from collections import OrderedDict
import sys

from util import modify_tuple

# domain range and configs
CONSTS = OrderedDict()

# Semantic:
# - repeat: iid sampling repeat
# - values: choices to samplen from
# - values_at_position: dictionary of position to values, indicating specific treatment at specific list positions
# - default: the default value to use if `on` is false
# - on: whether to sample it or nor
# - depends_on: the array length depends on the given variable name

CONSTS['conv_layer_n'] = {
    'values': [2, 3],
    'default': 2,
    'on': False
}
CONSTS['fold'] = {
    'values': [0, 1], 
    'depends_on': 'conv_layer_n', 
    'default': 1,
    'repeat': True,
    'on': False
}
CONSTS['dr'] = {
    'values': [0.5], 
    'depends_on': 'conv_layer_n',
    'on': True
}
CONSTS['ext_ebd'] = {
    'values': [True, False],
    'default': False,
    'on': False
}
CONSTS['batch_size'] = {
    'values': [9, 10, 11, 12], 
    'default': 10,
    'on': False
}
CONSTS['ebd_dm'] = {
    'values': [48], # 60 IN PAPER
    'on': True
}

CONSTS['ks'] = {
    'values_at_position': {
        0: [25, 20, 15],
        1: [8, 5, 2],
    },
    'depends_on': 'conv_layer_n',
    'on': True
}

CONSTS['nkerns'] = {
    'values_at_position': {
        0: [7, 6, 5],
        1: [16, 14, 12],
    },
    'depends_on': 'conv_layer_n',
    'on': True
}

CONSTS['filter_widths'] = {
    'values_at_position': {
        0: [9, 8, 7],
        1: [6, 5, 4],
    },
    'depends_on': 'conv_layer_n',
    'on': True
}

# CONSTS['l2_regs'] = {
#     'values_at_position': {
#         0: [1e-6, 1e-7]
#     },
#     'values': [1e-4, 1e-5, 1e-6],
#     'depends_on': 'conv_layer_n+2',
#     'on': True
# }


def coin_toss(p = 0.5):
    return np.random.binomial(n = 1, p = p, size = (1, ))

# semi-random ones
SEMI_RANDOM_PARAMS = {
    'ks': {
        2: (20, 5), # 4 for top IN PAPER
        3: (20, 10, 5)
    }, 
    'nkerns': {
        2: (6, 12), # 6 and 14 IN PAPER
        3: (5, 10, 18)
    }, 
    'filter_widths': {
        2: (10, 7), # 7,5 IN PAPER
        3: (6, 5, 3)
    }, 
    'l2_regs': {
        2: (1e-06, 1e-06, 1e-06, 0.0001),
        # 2: (1e-6, 3e-5, 3e-5, 1e-4),
        # 3: (1e-6, 3e-5, 3e-6, 1e-5, 1e-4),
    }
}

def get_possibility_n():
    """
    Get the possibility count of the current configuration
    """
    possibility_n = 1
    params = {}
    for key in CONSTS:
        if not CONSTS[key]['on']:
            assert CONSTS[key].has_key('default'), "if ON is False, then a default must be provided"
            if CONSTS[key].has_key('default'):
                CONSTS[key]['values'] = [CONSTS[key].get('default')]
        
        depends_on = CONSTS[key].get('depends_on')
        candidates = CONSTS[key].get('values', [])
        
        if depends_on:                
            if '+' in depends_on: # extra times
                name, extra_n_str = depends_on.split('+')
                dup_times = params[name] + int(extra_n_str.strip())
            else:
                dup_times = params[depends_on]
                

            if CONSTS[key].get('repeat'):
                if CONSTS[key].has_key('values_at_position'):
                    for v in CONSTS[key]['values_at_position'].values():
                        possibility_n *= len(v)
                possibility_n *= len(candidates)
                params[key] = tuple([random.choice(CONSTS[key]['values'])]) * dup_times
            else:
                if CONSTS[key].has_key('values_at_position'):
                    value_at_positions = CONSTS[key]['values_at_position']
                    for v in value_at_positions.values():
                        possibility_n *= len(v)
                    possibility_n *= (len(candidates) ** (dup_times - len(value_at_positions)))
                else:
                    possibility_n *= (len(candidates) ** dup_times)
                # this might be unnecessary
                # params[key] = tuple([random.choice(CONSTS[key]['values']) for _ in xrange(dup_times)])
        else:
            params[key] = random.choice(CONSTS[key]['values'])
            possibility_n *= len(candidates)
            

    return possibility_n

def sample_params(n = None, semi_random_params_key = 'conv_layer_n'):
    if n is None:
        n = get_possibility_n()
    else:
        possibility_n = get_possibility_n()
        assert n <= possibility_n, "%d > %d" %(n, possibility_n)
        
    pool = set()
    samples = []
    i = 0

    sys.stderr.write('total: %d\n' %(get_possibility_n()))

    while i < n:
        # random hyper parameters        
        params = {}
        for key in CONSTS:
            if not CONSTS[key]['on']:
                if CONSTS[key].get('default'):
                    CONSTS[key]['values'] = [CONSTS[key].get('default')]
                            
            depends_on = CONSTS[key].get('depends_on')
            candidates = CONSTS[key].get('values', [])
            if candidates:
                value = random.choice(candidates)
            else:
                value = None

            if depends_on:                
                if '+' in depends_on: # extra times
                    name, extra_n_str = depends_on.split('+')
                    dup_times = params[name] + int(extra_n_str.strip())
                else:
                    dup_times = params[depends_on]

                if CONSTS[key].get('repeat'):
                    assert value is not None
                    params[key] = tuple([value]) * dup_times
                else:
                    if candidates:
                        params[key] = tuple([random.choice(candidates) for _ in xrange(dup_times)])
                    else:
                        params[key] = tuple(range(dup_times)) # fake values to be replaced
            else:
                if isinstance(value, bool): #it's bool, show or hide
                    if value:
                        params[key] = value
                else:
                    assert value is not None
                    params[key] = value
            
            # remedy step that changes the value at specific positions
            if CONSTS[key].has_key('values_at_position'):
                values = [random.choice(candidates) for candidates in CONSTS[key]['values_at_position'].values()]
                positions = CONSTS[key]['values_at_position'].keys()
                params[key] = modify_tuple(params[key], positions, values)

        for key in SEMI_RANDOM_PARAMS:
            if not (CONSTS.get(key) and CONSTS[key]['on']): #it's not used for sampling
                params[key] = SEMI_RANDOM_PARAMS[key][params[semi_random_params_key]]
            
        if tuple(params.values()) in pool:
            continue
        else:
            i += 1
            sys.stderr.write("i = %d: %r\n" %(i, params))
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

def format_params_to_cmd(name, params, 
                         prefix = "python dcnn_train.py --corpus_path=data/twitter.pkl --l2  --norm_w --ebd_delay_epoch=0 --au=tanh --n_epochs=12", 
                         more_arguments = {}):
    arg_str = ' '.join(["--%s %r" %(k, v) for k, v in more_arguments.items()])
    params_str = params2str(params)
    sig = params2str(params, cmd_sep = ',,', key_val_sep = '=', tuple_sep = ',', key_prefix = '')
    return "%s %s %s --task_signature=%s,,%s --model_path=models/%s.pkl"%(
        prefix, arg_str, params_str, name, sig, sig
    )

def params2str(params, cmd_sep = ' ',key_val_sep = ' ', tuple_sep = ' ', key_prefix = '--'):
    return cmd_sep.join(["%s%s%s%s"  %(key_prefix, 
                                       key, 
                                       key_val_sep, 
                                       _format_value(value, tuple_sep = tuple_sep))
                         for key, value in params.items()])
    
if __name__ ==  "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description = "CNN with k-max pooling for sentence classification")
    
    parser.add_argument('-n', type=int,
                        dest = "possibility_n",
                        required = False,
                        help = 'How many tasks to sample'
    )
    
    parser.add_argument('--name', type=str,
                        required = True,
                        help = 'Task name'
    )

    parser.add_argument('--output', type=str,
                        required = True,
                        help = 'Where to save the result'
    )

    args =parser.parse_args()


    for param in sample_params(args.possibility_n):
        print format_params_to_cmd(args.name, 
                                   param,
                                   more_arguments = {"output": args.output}
        )
        
