"""
Recursive Neural Tensor  Network implemetation

ALgorithm described in:

Socher, 2013, Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank

"""
import theano
import theano.tensor as T

import numpy as np

from logreg import LogisticRegression

class RNTNLayer(object):
    """ Recursive Tensor Neural Network layer
    that outputs:
    
    - combined embedding
    - score
    """
    
    def __init__(self, 
                 rng,
                 embed_dim,
                 V = None,
                 W = None,
    ):
        """
        embed_dim: int
            dimension for embedding

        label_n: int
            number of labels

        V: theano.tensor.tensor3
            tensor layer parameter

        W: theano.tensor.dmatrix
            standard layer parameter
        """
        
        if V: 
            self.V = V
        else:
            self.V = theano.shared(
                rng.normal(
                    0, 0.05, 
                    (embed_dim, 2 * embed_dim, 2 * embed_dim)
                ),
                name = "V",
                borrow = True
            )
            
        if W:
            self.W = W
        else:
            self.W = theano.shared(
                rng.normal(
                    0, 0.05, 
                    (embed_dim, 2 * embed_dim)
                ),
                name = "W",
                borrow = True
            )                
        
    def output(self, left_input, right_input):
        """
        Param:
        -----------

        left_input: theano.tensor.row
            embedding for left hand side input

        right_input: theano.tensor.row
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
            left_input = left_input.dimshuffle('x', 0) 

        if right_input.ndim == 1:
            right_input = right_input.dimshuffle('x', 0) 
            
        concat_vec = T.concatenate(
            [left_input, right_input],
            axis = 1
        )
        
        result = T.tanh(T.dot(concat_vec, T.tensordot(self.V, concat_vec.T, [2, 0])) + T.dot(self.W, concat_vec.T))
        return result.flatten()
        
class RNTN(object):
    """
    Recursive Neural Tensor Network architecture
    """
    
    def __init__(self, tree_matrix, labels, vocab_size, embed_dim, label_n):
        """
    
        tree_matrix: numpy.array, 
            the tree matrix
            row: the ith tree node
            column: length 2, the left child phrase id, the right child phrase id, 
                    value is -1 if it has no child(single word case)

        labels: numpy.array
            labels, column/row vector denoting the corresponding sentiment label

        vocab_size: int
            vocabulary size
        
        embed_dim: int
            the embedding dimension

        """
        children_indices = theano.shared(value = tree_matrix, 
                                         name = "children_indices", 
                                         borrow = True)
        
        phrase_number = tree_matrix.shape[0]

        phrase_indices = theano.shared(value = np.arange(vocab_size,
                                                         vocab_size + phrase_number),
                                       name = "phrase_indices", 
                                       borrow = True)                
        rng = np.random.RandomState(1234)     
        
        word_embedding = theano.shared(
            value = rng.normal(0, 0.05, (vocab_size, embed_dim)),
            name = 'word_embedding',
            borrow = True,
        )        
        
        phrase_embedding = theano.shared(
            value = np.asarray(
                np.zeros(
                    (phrase_number,
                     embed_dim)
                ),
                dtype = theano.config.floatX
            ),
            name = "phrase_embedding",
            borrow = True
        )        
        
        rntn_layer = RNTNLayer(rng, embed_dim)

        # forward the embedding from bottom to up
        # and get the vector for each node in each tree
        def update_embedding(child_indices, my_index, output_model):
            left_child_embedding = output_model[child_indices[0]]
            right_child_embedding = output_model[child_indices[1]]
            
            parent_embedding = rntn_layer.output(left_child_embedding, 
                                                 right_child_embedding)
            
            return T.set_subtensor(output_model[my_index], parent_embedding)
            
        final_embedding, updates = theano.scan(
            fn = update_embedding, 
            sequences = [children_indices, phrase_indices],
            outputs_info = T.concatenate([word_embedding, phrase_embedding], 
                                         axis = 0),
            n_steps = children_indices.shape[0]
        )

        # the logistic regression layer that predicts the label
        
        logreg_layer = LogisticRegression(rng, 
                                          input = final_embedding[-1], 
                                          n_in = embed_dim,
                                          n_out = label_n
        )
        
        self.final_embedding = final_embedding[-1]
        
        assert self.final_embedding.ndim == 2
        
        self.cost = logreg_layer.nnl(labels)

def main():
    # load data
    pass
    
