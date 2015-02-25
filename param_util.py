"""
Utility for model parameter
"""
import os
try:
    from cPickle import load
except ImportError:
    from pickle import load

class Params(object):
    pass

def load_dcnn_model_params(path, param_str = None):
    """

    >>> p = load_dcnn_model_params("models/filter_widths=8,6,,batch_size=10,,ks=20,8,,fold=1,1,,conv_layer_n=2,,ebd_dm=48,,l2_regs=1e-06,1e-06,1e-06,0.0001,,dr=0.5,0.5,,nkerns=7,12.pkl")
    >>> p.ks
    (20, 8)
    >>> len(p.W)
    2
    >>> type(p.logreg_W)
    <type 'numpy.ndarray'>
    """
    if param_str is None:
        param_str = os.path.basename(path).split('.')[0]
    p = parse_param_string(param_str)
    
    stuff = load(open(path, "r"))
    
    for name, value in stuff:
        if not hasattr(p, name):
            setattr(p, name, value)
        else:
            # if appear multiple times,
            # make it a list
            setattr(p, name, [getattr(p, name), value])
    return p

def parse_param_string(s, desired_fields = {"ks", "fold", "conv_layer_n"}):
    """
    
    >>> p = parse_param_string("twitter4,,filter_widths=8,6,,batch_size=10,,ks=20,8,,fold=1,1,,conv_layer_n=2,,ebd_dm=48,,l2_regs=1e-06,1e-06,1e-06,0.0001,,dr=0.5,0.5,,nkerns=7,12")
    >>> p.ks
    (20, 8)
    >>> p.fold
    (1, 1)
    >>> p.conv_layer_n
    2
    """
    p = Params()
    segs = s.split(',,')
    for s in segs:
        if "=" in s:
            key, value = s.split('=')
            if key in desired_fields:
                if not ',' in value:
                    setattr(p, key, int(value))
                else:
                    setattr(p, key, tuple(map(int, value.split(','))))
    return p
