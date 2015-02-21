import numpy as np

from numpy_impl import conv2d

from test_util import assert_matrix_eq

########################
# Numpy part #
########################
        
feature_n_prev = 3

input_feature_map = np.random.rand(2,feature_n_prev,2,2)

# shape: (2,3,2,2)
filters = np.random.rand(2,feature_n_prev,2,2)

numpy_output = conv2d(input_feature_map, filters)
print numpy_output

########################
# Theano part#
########################

import theano

input_feature_map_sym = theano.tensor.dtensor4("input_feature_map")
filters_sym = theano.tensor.dtensor4("filters")

f = theano.function(inputs = [input_feature_map_sym, filters_sym], 
                    outputs = theano.tensor.nnet.conv.conv2d(input_feature_map_sym, 
                                                             filters_sym, 
                                                             border_mode = "full")
)

theano_output = f(input_feature_map, filters)
print theano_output


assert_matrix_eq(numpy_output, theano_output, "Conv2d")
