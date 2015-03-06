import numpy as np
import theano

from numpy_impl import LogisticRegression

from logreg import LogisticRegression as TheanoLogisticRegression

from test_util import assert_matrix_eq

#########################
# NUMPY PART
#########################
# 5 labels and 10 inputs

W = np.random.rand(10, 5)
b = np.random.rand(5)

x = np.random.rand(3, 10)
y = np.asarray(np.random.randint(5, size = 3), 
               dtype=np.int32
)

np_l = LogisticRegression(W, b)

#########################
# THEANO PART
#########################

x_symbol = theano.tensor.dmatrix('x')
y_symbol = theano.tensor.ivector('y')

th_l = TheanoLogisticRegression(rng = np.random.RandomState(1234), 
                                input = x_symbol, 
                                n_in = 10, 
                                n_out = 5,
                                W = theano.shared(value = W, 
                                                  name = "W"), 
                                b = theano.shared(value = b, 
                                                  name = "b")
)

f1 = theano.function(inputs = [x_symbol, y_symbol], 
                     outputs = th_l.nnl(y_symbol)
                 )

actual = np_l.nnl(x, y)
expected = f1(x, y)


assert_matrix_eq(actual, expected, "nnl")


f2 = theano.function(inputs = [x_symbol, y_symbol], 
                     outputs = th_l.errors(y_symbol)
                 )

actual = np_l.errors(x, y)
expected = f2(x, y)

assert_matrix_eq(actual, expected, "errors")
