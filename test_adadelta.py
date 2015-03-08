import numpy as np
import theano
import theano.tensor as T
from adadelta import build_adadelta_updates 
from test_util import assert_matrix_eq

rho=0.95
epsilon=0.01
n_iter = 10

np_params = [np.random.rand(2,2), np.random.rand(2,2)]

params = [theano.shared(value = param_val,
                        name = "param-%d" %(i),
                        borrow = True)
          for i, param_val in enumerate(np_params)]


cost = T.sum(T.dot(params[0], params[1]))# some cost function


param_shapes = [(2,2), (2,2)] 
param_grads = [T.grad(cost, param) for param in params]

assert len(np_params) == len(params) == len(param_shapes) == len(param_grads)

updates = build_adadelta_updates(params, param_shapes, param_grads, 
                                 rho = rho, epsilon = epsilon)

update = theano.function(inputs = [], 
                         outputs = params, 
                         updates = updates)


for i in xrange(n_iter):
    update()


#######################
# NUMPY IMPLEMENTATION
#######################

dummy_params = [T.dmatrix('dp1'), T.dmatrix('dp2')]
dummy_cost = T.sum(T.dot(dummy_params[0], dummy_params[1]))# some cost function

grads = theano.function(inputs = dummy_params, 
                        outputs = [T.grad(dummy_cost, param) for param in dummy_params])

def numpy_update(xs, egs, exs):
    grad_xs = grads(*xs)
    new_egs = [rho * eg + (1-rho) * (grad_x ** 2) 
              for grad_x, eg in zip(grad_xs, egs)]
    delta_xs = [- np.sqrt(ex + epsilon) / np.sqrt(new_eg + epsilon) * grad_x
               for grad_x, new_eg, ex in zip(grad_xs, new_egs, exs)]

    new_exs = [rho * ex + (1-rho) * (delta_x ** 2) 
              for delta_x, ex in zip(delta_xs, exs)]
    
    new_xs = [x + delta_x 
              for x, delta_x in zip(xs, delta_xs)]
    
    return new_xs, new_egs, new_exs

egs = [np.zeros(shape, dtype = theano.config.floatX)
       for shape in param_shapes]
exs = [np.zeros(shape, dtype = theano.config.floatX)
       for shape in param_shapes]

for i in xrange(n_iter):
    np_params, egs, exs = numpy_update(np_params, egs, exs)


# if the updates are te same
for np_param, param in zip(np_params, params):
    assert_matrix_eq(np_param, param.get_value(), param.name)
