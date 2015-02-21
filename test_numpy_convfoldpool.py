import theano
import numpy as np
from dcnn import ConvFoldingPoolLayer

#########################
# NUMPY PART
#########################
W = np.asarray([
    [ #1st filter
        [[0, 0],
         [0, 1]], # 1->1
        [[0, 1],
         [0, 0]] # 1->2
    ], 
    [ #2nd filter
        [[1, 0],
         [0, 0]], # 2->1
        [[0, 0],
         [1, 0]] # 2->2
    ],
    [  #3rd filter
        [[1, 1],
         [0, 0]], # 3->1
        [[0, 0],
         [1, 1]] # 3->2
    ]
], dtype=theano.config.floatX)

filter_shape = (3, 2, 2, 2)

b = np.array([1, 1, 1])
k = 4
fold = False
layer = ConvFoldingPoolLayer(k = k,
                             fold = fold,
                             W = W,
                             b = b)

x = np.asarray([
    [
        [
            [1, 1, 1], 
            [2, 2, 2], 
            [3, 3, 3]
        ],
        [
            [1, 1, 1], 
            [2, 2, 2], 
            [3, 3, 3]
        ]
    ]
], dtype=theano.config.floatX)

actual = layer.output(x)


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

expected = f(x)

print expected
print actual

print "Test successful?", (actual == expected).all()
