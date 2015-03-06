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


class LogisticRegression(object):
    def __init__(self, W, b):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        :type W: numpy.ndarray
        :param W: (input number, output/label number)

        :type b: numpy.ndarray
        :param b:

        """
        assert W.shape[1] == b.shape[0]
        assert W.ndim == 2
        assert b.ndim == 1

        self.W = W
        self.b = b

    def _p_y_given_x(self, x):
        return softmax(np.dot(x, self.W) + self.b[np.newaxis, :])

    def nnl(self, x, y):
        """
        negative log-likelihood

        x: the input 2d array, (#instance, #feature)
        y: the correct label, (#instance)
        """
        p_y_given_x = self._p_y_given_x(x)
        return np.mean(
            -np.log(p_y_given_x[np.arange(y.shape[0]), y])
        )
        
    def errors(self, x, y):
        """
        the error rate

        x: the input 2d array, (#instance, #feature)
        y: the correct label, (#instance)
        """
        assert y.dtype == np.int32

        pred_y = self.predict(x)

        return np.sum(pred_y != y) / float(pred_y.shape[0])
        
    def predict(self, x):
        p_y_given_x = self._p_y_given_x(x)
        return np.argmax(p_y_given_x, axis = 1)
