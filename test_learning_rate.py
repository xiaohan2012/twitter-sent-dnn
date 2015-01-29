"""
Comparing adagrad, adadelta and constant learning in gradient descent(the seddle point function y^2 - x^2)

Reference:
1. comparison on several learning rate update scheme: http://ml.memect.com/archive/2014-12-12/short.html#3786866375172817
2. Saddle point, http://en.wikipedia.org/wiki/Saddle_point

"""
import numpy as np
import theano
import theano.tensor as T

rho = 0.95
epsilon = 0.00001
gamma = 0.1

const_lr = 0.01

init_x = [0.1, 0.1]
x = theano.shared(
    np.array(init_x, dtype = theano.config.floatX), 
    borrow = True,
    name = "x"
)

tolorate = 0.01

params = [x]
param_shapes = [(2,)]

# cost = 0.5 * (x[0]-2) ** 2 + (x[1]-2) ** 2
cost = x[0] ** 2 - x[1] ** 2

param_grads = [T.grad(cost, param) for param in params]

def make_func(x, cost, updates, init_x):
    x.set_value(init_x)
    f = theano.function(
        inputs = [], 
        outputs = [x, cost], 
        updates = updates
    )
    return f

def simulate(f, n_epoch_max = 100):
    epoch = 0
    used_epochs = 0
    xs = []
    print "##################"
    while epoch < n_epoch_max:
        x_val, cost_val = f()
        xs.append(x_val)
        # if abs(cost_val) < tolorate:
        #     break
        epoch += 1
        used_epochs += 1
    return xs, used_epochs


###############
# ADADELTA    #
###############
print "Using AdaDelta with rho = %f and epsilon = %f" %(rho, epsilon)
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

f = make_func(x, cost, updates, init_x)
adadelta_xs, adadelta_epochs = simulate(f)

##############
# ADAGRAD    #
##############
print "Using AdaGrad with gamma = %f and epsilon = %f" %(gamma, epsilon)
grad_hists = [
            theano.shared(
                value = np.zeros(param_shape,
                                 dtype = theano.config.floatX
                             ),
                borrow = True,        
                name = "grad_hist:" + param.name
            )
            for param_shape, param in zip(param_shapes, params)
        ]
        
new_grad_hists = [
    g_hist + g ** 2
    for g_hist, g in zip(grad_hists, param_grads)
]

param_updates = [
    (param, param - theano.printing.Print("lr")(gamma * epsilon / (T.sqrt(g_hist) + epsilon)) * param_grad)
    for param, param_grad in zip(params, param_grads)
]

grad_hist_update = zip(grad_hists, new_grad_hists)

updates = grad_hist_update + param_updates

f = make_func(x, cost, updates, init_x)
adagrad_xs, adagrad_epochs = simulate(f)

###############
# constant lr #
###############
print "Usin constant learning rate %f" %(const_lr)

updates = [
    (param, param - const_lr * param_grad)
    for param, param_grad in zip(params, param_grads)
]

f = make_func(x, cost, updates, init_x)
const_lr_xs, const_lr_epochs = simulate(f)

from matplotlib import pyplot  as plt

def myplot(data, style, title, plot_number, total):
    plt.subplot(1,total,plot_number)
    x, y = zip(*data)
    plt.plot(x, y, 'ro-')
    plt.title(title)
    plt.xlim([-10, 10]); plt.ylim([-10, 10])

myplot(adadelta_xs, 
       'ro-', 
       "AdaDelta(%d epochs)" %(adadelta_epochs), 
       1, 3)

myplot(adagrad_xs, 
       'ro-', 
       "AdaGrad(%d epochs)" %(adagrad_epochs), 
       2, 3)

myplot(const_lr_xs, 
       'ro-', 
       "ConstLR(%d epochs)" %(const_lr_epochs), 
       3, 3)

plt.show()
