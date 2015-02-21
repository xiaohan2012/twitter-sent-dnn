import theano
import numpy as np
from dcnn import ConvFoldingPoolLayer
from test_util import assert_matrix_eq

#########################
# NUMPY PART
#########################
filter_shape = (3, 2, 2, 2)
W = np.asarray(np.random.rand(3, 2, 2, 2), 
               dtype=theano.config.floatX)
b = np.asarray(np.random.rand(3), 
               dtype=theano.config.floatX)
k = 4
fold = False
np_layer = ConvFoldingPoolLayer(k = k,
                                fold = fold,
                                W = W,
                                b = b)




#########################
## THEANO PART
#########################
from dcnn_train import ConvFoldingPoolLayer as TheanoConvFoldingPoolLayer

x_symbol = theano.tensor.dtensor4('x')
layer = TheanoConvFoldingPoolLayer(rng = np.random.RandomState(1234), 
                           input = x_symbol,
                           filter_shape = filter_shape,
                           k = k,
                           activation = "tanh",
                           norm_w = True,
                           fold = fold,
                           W = theano.shared(value = W, 
                                             borrow = True,
                                             name="W"
                                         ),
                           b = theano.shared(value = b, 
                                             borrow = True,
                                             name="b"
                                         )
)

f = theano.function(inputs = [x_symbol], 
                    outputs = layer.output)


########## Test ################

x = np.asarray(np.random.rand(2,2,3,3), 
               dtype=theano.config.floatX)

actual = np_layer.output(x)
expected = f(x)

assert_matrix_eq(actual, expected, "Conv")
