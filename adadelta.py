"""
Adadelta algorithm implementation
"""
import numpy as np
import theano
import theano.tensor as T

def build_adadelta_updates(params, param_shapes, param_grads, rho=0.95, epsilon=0.001):
    # AdaDelta parameter update
    # E[g^2]
    # initialized to zero
    egs = [
        theano.shared(
            value = np.zeros(param_shape,
                             dtype = theano.config.floatX
                         ),
            borrow = True,        
            name = "Eg:" + param.name
        )
        for param_shape, param in zip(param_shapes, params)
    ]

    # E[\delta x^2], initialized to zero
    exs = [
        theano.shared(
            value = np.zeros(param_shape,
                             dtype = theano.config.floatX
                         ),
            borrow = True,        
            name = "Ex:" + param.name
        )
        for param_shape, param in zip(param_shapes, params)
    ]        

    new_egs = [
        rho * eg + (1 - rho) * g ** 2
        for eg, g in zip(egs, param_grads)
    ]
        
    delta_x = [
        -(T.sqrt(ex + epsilon) / T.sqrt(new_eg + epsilon)) * g
        for new_eg, ex, g in zip(new_egs, exs, param_grads)
    ]    

    new_exs = [
        rho * ex + (1 - rho) * (dx ** 2)
        for ex, dx in zip(exs, delta_x)
    ]    

    egs_updates = zip(egs, new_egs)
    exs_updates = zip(exs, new_exs)
    param_updates = [
        (p, p + dx)
        for dx, g, p in zip(delta_x, param_grads, params)
    ]

    updates = egs_updates + exs_updates + param_updates
    
    return updates
