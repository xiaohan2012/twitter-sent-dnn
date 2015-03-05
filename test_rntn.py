import numpy as np
from recnn import RNTNLayer
import theano

V_val = np.asarray((np.arange(3*6*6) / 100).reshape((3,6,6)), dtype=theano.config.floatX)
W_val = np.asarray((np.arange(3*6) / 100).reshape((3,6)), dtype=theano.config.floatX)

l = RNTNLayer(np.random.RandomState(1234), 3,
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

concat_vec = np.concatenate(
    [left_input, right_input],
    axis = 1
)


result = np.tanh(np.dot(concat_vec, np.tensordot(V_val, np.transpose(concat_vec), [2, 0])) + np.dot(W_val, np.transpose(concat_vec)))

expected = np.squeeze(result)

print expected.shape

################
# THEANO PART  #
################
left = theano.tensor.drow("left")
right = theano.tensor.drow("right")


f = theano.function(
    inputs = [left, right],
    outputs = l.output(left, right)
)

actual = f(left_input, right_input)


assert (np.abs(actual - expected) < 1e-5).all()

