"""
Numpy implementation of RecNN
"""
import numpy as np
from numpy_impl import LogisticRegression


class RNTNLayer(object):
    """ Recursive Tensor Neural Network layer
    that outputs:
    
    - combined embedding
    - score
    """
    
    def __init__(self, 
                 V,
                 W,
    ):
        """
        V: numpy.array
            tensor layer parameter

        W: numpy.array
            standard layer parameter
        """
        
        assert V.ndim == 3
        assert W.ndim == 2
        
        assert V.shape[1] == V.shape[2]
        assert V.shape[0] == W.shape[0]
        assert V.shape[1] == W.shape[1]
        
        self.V = V
        self.W = W
        
    def output(self, left_input, right_input):
        """
        Param:
        -----------

        left_input: numpy.array
            embedding for left hand side input

        right_input: numpy.array
            embedding for right hand side input

        Return:
        -----------
        The output embedding
        """
        assert left_input.ndim <= 2
        assert right_input.ndim <= 2

        # if left_input and right_input are 1d array
        # make it a 2D row 
        if left_input.ndim == 1: 
            left_input = left_input[None,0]

        if right_input.ndim == 1:
            right_input = right_input[None,0]
            
        concat_vec = np.concatenate(
            [left_input, right_input],
            axis = 1
        )
        
        result = np.tanh(np.dot(concat_vec, np.tensordot(self.V, np.transpose(concat_vec), [2, 0])) + np.dot(self.W, np.transpose(concat_vec)))
        return result.squeeze()
