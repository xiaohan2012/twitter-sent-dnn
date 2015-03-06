import numpy as np
from recnn_train import RNTNLayer as TheanoRNTNLayer
import theano
from test_util import assert_matrix_eq

V_val = np.asarray((np.arange(3*6*6) / 100).reshape((3,6,6)), dtype=theano.config.floatX)
W_val = np.asarray((np.arange(3*6) / 100).reshape((3,6)), dtype=theano.config.floatX)

theano_l = TheanoRNTNLayer(np.random.RandomState(1234), 3,
              V = theano.shared(value = V_val, 
                                name = "V",
                                borrow = True), 
              W = theano.shared(value = W_val, 
                                name = "W",
                                borrow = True)
)


left_input = np.asarray([[0,0,1]], dtype=theano.config.floatX)
right_input = np.asarray([[0,1,0]], dtype=theano.config.floatX)


################
# NUMPY IMPML ##
################

from recnn import RNTNLayer as NumpyRNTNLayer
numpy_l = NumpyRNTNLayer(theano_l.V.get_value(), theano_l.W.get_value())

actual = numpy_l.output(left_input, right_input)
actual1 = numpy_l.output(np.squeeze(left_input), np.squeeze(right_input)) #passing 1d array

################
# THEANO PART  #
################
left = theano.tensor.drow("left")
right = theano.tensor.drow("right")


f = theano.function(
    inputs = [left, right],
    outputs = theano_l.output(left, right)
)

expected = f(left_input, right_input)

assert_matrix_eq(actual, expected, "output")
assert_matrix_eq(actual1, expected, "output(1d passed in)")

