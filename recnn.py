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
    
    def __init__(self, x, y, vocab_size, embed_dim, label_n):
        """
        x: theano.tensor.imatrix, (minibatch size, 3)
            the tree matrix of the minibatch
            for each row, (node id, left child id, right child id)

        y: theano.tensor.ivector, (minibatch size,)
            the labels

        vocab_size: int
            vocabulary size, including both the words and phrases
        
        embed_dim: int
            the embedding dimension

        """
        assert x.ndim == 2
        assert y.ndim == 1
        
        parent_ids = x[:,0]
        children_ids = x[:,1:]
        
        rng = np.random.RandomState(1234)     

        self.embedding = theano.shared(
            value = rng.normal(0, 0.05, (vocab_size, embed_dim)),
            name = 'embedding',
            borrow = True,
        )        
        
        rntn_layer = RNTNLayer(rng, embed_dim)

        # Update the embedding by
        # forwarding the embedding from bottom to up
        # and getting the vector for each node in each tree
        
        def update_embedding(child_indices, my_index, embedding):
            assert child_indices.ndim == 1
            assert my_index.ndim == 0

            return T.switch(T.eq(child_indices[0], -1), # NOTE: not using all() because it's non-differentiable
                            embedding, # if no child, return the word embedding
                            T.set_subtensor(embedding[my_index], # otherwise, compute the embedding of RNTN layer
                                            rntn_layer.output(embedding[child_indices[0]], 
                                                              embedding[child_indices[1]])
                                            # embedding[child_indices[0]] + embedding[child_indices[1]]
                                        )
            )
            
        final_embedding, updates = theano.scan(
            fn = update_embedding, 
            sequences = [children_ids, parent_ids],
            outputs_info = self.embedding, # we should pass the whole matrix and fill in the positions if necessary
        )
                

        self.update_embedding = theano.function(inputs = [x], 
                                                updates = [(self.embedding, 
                                                            T.set_subtensor(self.embedding[parent_ids], final_embedding[-1][parent_ids]))])

        # the logistic regression layer that predicts the label
        logreg_layer = LogisticRegression(rng, 
                                          input = final_embedding[-1][parent_ids], 
                                          n_in = embed_dim,
                                          n_out = label_n
        )
        
        self.cost = logreg_layer.nnl(y)

        self.params = [logreg_layer.W, logreg_layer.b, rntn_layer.V, rntn_layer.W, self.embedding]
        self.grads = [T.grad(cost = self.cost, wrt=p) for p in self.params]

        
        # TODO: in this step, forward propagation is done again besides the one in `update_embedding`
        #       this extra computation should be avoided
        self.train = theano.function(inputs = [x, y], 
                                     updates = [(p, p - 10*g) 
                                                for p,g in zip(self.params, self.grads)])

def main():
    # load data
    pass
    
