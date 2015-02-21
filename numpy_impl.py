import numpy as np
from scipy.signal import convolve2d

def conv2d(input_feature_map, filters, mode = "full"):
    """
    Convolution operation that functions as same as `theano.tensor.nnet.conv.conv2d`
    
    Refer to: [the theano documentation](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)
    
    """
    assert len(input_feature_map.shape) == 4
    assert len(filters.shape) == 4
    batch_size, input_feature_n1, input_w, input_h = input_feature_map.shape
    output_feature_n, input_feature_n2, filter_w, filter_h = filters.shape

    assert input_feature_n1 == input_feature_n2, "%d != %d" %(input_feature_n1, input_feature_n2)

    output_feature_map = np.zeros((batch_size, 
                                   output_feature_n, 
                                   input_w + filter_w - 1, 
                                   input_h + filter_h - 1))

    for i in xrange(batch_size):
        # for the ith instance
        for k in xrange(output_feature_n):
            # for the kth feature map in the output
            for l in xrange(input_feature_n1):
                # for the lth feature map in the input
                output_feature_map[i, k] += convolve2d(input_feature_map[i, l], 
                                                       filters[k, l], 
                                                       mode = mode)

    return output_feature_map



def softmax(w):
    """
    w: (instances, feature values)

    >>> softmax(np.asarray([[1,2], [3,4]], dtype=np.float32)) #doctest: +SKIP
    """
    exp_w = np.exp(w)

    sums = np.sum(exp_w, axis = 1)

    return exp_w / sums[:,np.newaxis]
