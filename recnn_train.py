"""
Recursive Neural Tensor  Network implemetation

ALgorithm described in:

Socher, 2013, Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank

"""
import sys
import theano
import theano.tensor as T

import numpy as np

from logreg import LogisticRegression

from recnn import RNTN as NumpyRNTN
from recnn_util import (collect_nodes,
                        replace_tokens_by_condition,
                        build_input,
                        build_node_id_mapping)
from adadelta import build_adadelta_updates

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
            
        self.params = [self.V, self.W]
        self.param_shapes = [self.V.get_value().shape, self.W.get_value().shape]
        
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
        
        self.rntn_layer = RNTNLayer(rng, embed_dim)

        # Update the embedding by
        # forwarding the embedding from bottom to up
        # and getting the vector for each node in each tree
        
        def update_embedding(child_indices, my_index, embedding):

            assert child_indices.ndim == 1
            assert my_index.ndim == 0

            return T.switch(T.eq(child_indices[0], -1), # NOTE: not using all() because it's non-differentiable
                            embedding, # if no child, return the word embedding
                            T.set_subtensor(embedding[my_index], # otherwise, compute the embedding of RNTN layer
                                            self.rntn_layer.output(embedding[child_indices[0]], 
                                                                   embedding[child_indices[1]])
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
        self.logreg_layer = LogisticRegression(rng, 
                                          input = final_embedding[-1][parent_ids], 
                                          n_in = embed_dim,
                                          n_out = label_n
        )
        
        cost = self.logreg_layer.nnl(y)

        params = self.logreg_layer.params + self.rntn_layer.params + [self.embedding]
        self.params = params

        param_shapes = self.logreg_layer.param_shapes + self.rntn_layer.param_shapes + [(vocab_size, embed_dim)]
        
        grads = [T.grad(cost = cost, wrt=p) for p in params]
        
        updates = build_adadelta_updates(params, param_shapes, grads, epsilon = 0.1)
        
        # TODO: in this step, forward propagation is done again besides the one in `update_embedding`
        #       this extra computation should be avoided
        self.train = theano.function(inputs = [x, y], 
                                     updates = updates)


def main(batch_size = 3):

    import random
    from recnn_util import load_data
    
    train_trees, dev_trees, test_trees, token2id = load_data("data/stanford_sentiment_treebank.pkl")
    sys.stderr.write("Data load done")
    
    batch_number = len(train_trees) / batch_size
    
    x = T.imatrix('x')
    y = T.ivector('y')
    
    model = RNTN(
        x, y,
        vocab_size = len(token2id), 
        embed_dim = 10, 
        label_n = 5,
    )
    
    sys.stderr.write("Model compilation done\n")
    
    training_iter = 0
    validation_frequency = 10
    
    print "start training.."
    while True:
        # shuffle data
        random.shuffle(train_trees)
        # for each mini-batch in 
        for i in xrange(batch_number):
            training_iter += 1
            
            batch_trees = train_trees[i*batch_size:(i+1)*batch_size]
            batch_nodes = collect_nodes(batch_trees)
            x,y = build_input(batch_nodes, token2id)
         
            # train the model()
            model.update_embedding(x)
            model.train(x, y)
            
            print "At iter %d" %(training_iter)

            if training_iter % validation_frequency == 0:
                classifier = NumpyRNTN.load_from_theano_model(model, token2id)

                def accuracy(trees):
                    prediction = np.array([classifier.predict_top_node(tree) for tree in trees])
                    correct = np.array([tree[0] for tree in trees])
                    return np.mean(prediction == correct)

                print "At iter %d, train accuracy %.2f%%, dev accuracy %.2f%%" %(training_iter, 
                                                                                 accuracy(train_trees) * 100,
                                                                                 accuracy(dev_trees) * 100)
                
    
if __name__ == "__main__":
    main()
