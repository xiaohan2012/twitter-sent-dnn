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
        
        assert V.shape[1] == V.shape[2], "%r" %(V.shape)
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
            left_input = left_input[np.newaxis,:]

        if right_input.ndim == 1:
            right_input = right_input[np.newaxis,:]
        
        concat_vec = np.concatenate(
            [left_input, right_input],
            axis = 1
        )
        
        result = np.tanh(np.dot(concat_vec, np.tensordot(self.V, np.transpose(concat_vec), [2, 0])) + np.dot(self.W, np.transpose(concat_vec)))
        return result.squeeze()

class RNTN(object):
    def __init__(self, embedding, rntn_layer, logreg_layer, word2id):
        self.embedding = embedding
        self.rntn_layer = rntn_layer
        self.logreg_layer = logreg_layer
        self.word2id = word2id

    @classmethod
    def load_from_theano_model(cls, model, word2id):
        return RNTN(embedding = model.embedding.get_value(), 
                    rntn_layer = RNTNLayer(model.rntn_layer.V.get_value(), model.rntn_layer.W.get_value()), 
                    logreg_layer = LogisticRegression(model.logreg_layer.W.get_value(), model.logreg_layer.b.get_value()), 
                    word2id = word2id)

    def get_node_vector(self, node):
        if isinstance(node, tuple): # is internal node
            if len(node) == 3:
                left_node_vector = self.get_node_vector(node[1])
                right_node_vector = self.get_node_vector(node[2])
                return self.rntn_layer.output(left_node_vector, right_node_vector)
            elif len(node) == 2:
               return self.get_node_vector(node[1])
            else:
                raise ValueError("Invalid tuple length(should be 2 or 3)")
        else:
            assert isinstance(node, basestring)
            idx = (self.word2id[node] 
                   if node in self.word2id 
                   else self.word2id["<UNK>"])
            
            return self.embedding[idx]

    def predict_all_nodes(self, nodes):
        raise NotImplementedError

    def predict_top_node(self, node):
        vec = self.get_node_vector(node)
        return self.logreg_layer.predict(vec)[0]


