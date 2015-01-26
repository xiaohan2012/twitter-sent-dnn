import numpy as np
import theano
import theano.tensor as T

from cnn4nlp import DropoutLayer


x = T.dmatrix("x")

rng = np.random.RandomState(1234)

l = DropoutLayer(x, rng, 0.5)


x_val = np.arange(6).reshape((2,3))

f = theano.function(inputs = [x],
                outputs = l.output)

print x_val
print f(x_val) 
print f(x_val) # second time should be different



