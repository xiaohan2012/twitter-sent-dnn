"""
Numpy version of DCNN, used for prediction, instead of training
"""
import numpy as np

from numpy_impl import (conv2d, LogisticRegression)

_MODEL_PATH = "models/filter_widths=10,7,,batch_size=10,,ks=20,5,,fold=1,1,,conv_layer_n=2,,ebd_dm=48,,nkerns=6,12,,dr=0.5,0.5,,l2_regs=1e-06,0.0001,1e-05,1e-06.pkl"

class WordEmbeddingLayer(object):
    """
    Layer that takes input vectors, output the sentence matrix
    """
    def __init__(self, 
                 embeddings):
        """
        embeddings: numpy.ndarray
                    Embedding, (vocab size, embedding dimension)
        """  
        assert embeddings.ndim == 2, "Should be have 2 dimensions"
        self.embeddings = embeddings

    def output(self, x):
        """
        x: numpy.ndarray
           the input sentences consiting of word indices (number of instances, sentence word number)
        """
        sent_matrices = np.array(
            map(lambda sent: self.embeddings[sent], 
                x)
        )
        
        # equivalent to dimshuffle(0, 'x', 2, 1) in Theano
        return sent_matrices.swapaxes(1,2)[:,None,:,:]

class ConvFoldingPoolLayer(object):
    """
    Convolution, folding and k-max pooling layer
    """
    def __init__(self, 
                 k,
                 fold,
                 W,
                 b):
        """
        k: int
           the k value in the max-pooling layer

        fold: int, 0 or 1
           fold or not

        W: numpy.ndarray,
           the filter weight matrices, 
           dimension: (number of filters, num input feature maps, filter height, filter width)

        b: numpy.ndarray,
           the filter bias, 
           dimension: (number of filters, )
        """
        self.fold_flag = fold
        self.W = W
        self.b = b
        self.k = k

    def fold(self, x):
        """
        x: np.ndarray
           the input, 4d array
        """
        return (x[:, :, np.arange(0, x.shape[2], 2)] + 
                x[:, :, np.arange(1, x.shape[2], 2)]) / 2
        
    def k_max_pool(self, x, k):
        """
        perform k-max pool on the input along the rows

        x: numpy.ndarray
           the input, 4d array

        k: theano.tensor.iscalar
            the k parameter

        Returns: 
        4D numpy.ndarray
        """
        ind = np.argsort(x, axis = 3)

        sorted_ind = np.sort(ind[:,:,:, -k:], axis = 3)
        
        dim0, dim1, dim2, dim3 = sorted_ind.shape
        
        indices_dim0 = np.arange(dim0).repeat(dim1 * dim2 * dim3)
        indices_dim1 = np.transpose(np.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis=1)).flatten()
        indices_dim2 = np.transpose(np.arange(dim2).repeat(dim3).reshape((dim2*dim3, 1)).repeat(dim0 * dim1, axis = 1)).flatten()
        
        return x[indices_dim0, indices_dim1, indices_dim2, sorted_ind.flatten()].reshape(sorted_ind.shape)
        
    def output(self, x):
        # non-linear transform of the convolution output
        conv_out = conv2d(x, 
                          self.W, 
                          mode = "full")                     
        
        if self.fold_flag:
            # fold
            fold_out = self.fold(conv_out)
        else:
            fold_out = conv_out

        # k-max pool        
        pool_out = (self.k_max_pool(fold_out, self.k) + 
                    self.b[np.newaxis, :, np.newaxis, np.newaxis])
        
        return np.tanh(pool_out)
        
class DCNN(object):
    def __init__(self, params):
        self.e_layer = WordEmbeddingLayer(embeddings = params.embeddings)
        self.c_layers = []
        
        for i in xrange(params.conv_layer_n):
            self.c_layers.append(ConvFoldingPoolLayer(params.ks[i],
                                                      params.fold[i],
                                                      W = params.W[i],
                                                      b = params.b[i])
            )

        self.l_layer = LogisticRegression(
            params.logreg_W,
            params.logreg_b
        )

    def _p_y_given_x(self, x):
        output = self.e_layer.output(x)
        
        for l in self.c_layers:
            output = l.output(output)

        assert output.ndim == 4
        output = output.reshape(
            (output.shape[0], 
             np.prod(output.shape[1:]))
        )
        return self.l_layer._p_y_given_x(output)

    def predict(self, x):
        return np.argmax(self._p_y_given_x(x), axis = 1)
 
    # The following functions are 
    # FOR TESTING PURPOSE               
    #
    def _nnl(self, x, y):
        p_y_given_x = self._p_y_given_x(x)
        return np.mean(
            -np.log(p_y_given_x[np.arange(y.shape[0]), y])
        )

    def _errors(self, x, y):
        assert y.dtype == np.int32, "%r != %r" %(y.dtype, np.int32)
        pred_y = self.predict(x)
        return np.sum(pred_y != y) / float(pred_y.shape[0])


    def _c_layer_output(self, x):
        output = self.e_layer.output(x)
        
        for l in self.c_layers:
            output = l.output(output)

        return output
