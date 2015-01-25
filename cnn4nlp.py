"""
CNN for sentence modeling described in:
A Convolutional Neural Network for Modeling Sentence
"""
import sys, os, time
import pdb

import math
import numpy as np
import theano
import theano.tensor as T
import util

THEANO_COMPILE_MODE = "FAST_RUN"

from logreg import LogisticRegression
class WordEmbeddingLayer(object):
    """
    Layer that takes input vectors, output the sentence matrix
    """
    def __init__(self, rng, 
                 input,
                 vocab_size, 
                 embed_dm):
        """
        input: theano.tensor.dmatrix, (number of instances, sentence word number)
        
        vocab_size: integer, the size of vocabulary,

        embed_dm: integer, the dimension of word vector representation
        """                
        # # Note:
        # # assume the padding is the last row
        # # and it's constant to 0
        # pad_val = np.zeros((1, embed_dm))
        
        # embed_val = np.concatenate((embed_val_except_pad, pad_val), 
        #                            axis = 0)
        
        self.embeddings = theano.shared(
            np.asarray(rng.uniform(
                low = -1,
                high = 1,
                size = (vocab_size, embed_dm)
            ), 
                       dtype = theano.config.floatX),
            borrow = True,
            name = 'embeddings'
        )
        
        self.params = [self.embeddings]
        
        self.param_shapes = [(vocab_size, embed_dm)]
        
        # updated_embeddings = self.embeddings[:-1] # all rows are updated except for the last row
        
        self.normalize = theano.function(inputs = [],
                                         updates = { self.embeddings:
                                                     (self.embeddings/ 
                                                      T.sqrt((self.embeddings**2).sum(axis=1)).dimshuffle(0,'x'))
                                                 }
        )

        self.normalize() #initial normalization

        # Return:
        
        # :type, theano.tensor.tensor4
        # :param, dimension(1, 1, word embedding dimension, number of words in sentence)
        #         made to be 4D to fit into the dimension of convolution operation
        sent_embedding_list, updates = theano.map(lambda sent: self.embeddings[sent], 
                                                  input)
        sent_embedding_tensor = T.stacklists(sent_embedding_list) # make it into a 3D tensor
        
        self.output = sent_embedding_tensor.dimshuffle(0, 'x', 2, 1) # make it a 4D tensor
        
class DropoutLayer(object):
    """
    As the name suggests

    Refer to here: https://github.com/mdenil/dropout/blob/master/mlp.py
    """

    def __init__(self, input, rng, dropout_rate):

        srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
        
        # p=1-p because 1's indicate keep and p is prob of dropping
        mask = srng.binomial(n=1, 
                             p=1-dropout_rate, 
                             size=input.shape)
        
        self.output = input * T.cast(mask, theano.config.floatX)
        
    
class ConvFoldingPoolLayer(object):
    """
    Convolution, folding and k-max pooling layer
    """
    def __init__(self, 
                 rng, 
                 input,
                 filter_shape,
                 k,
                 W = None,
                 b = None):
        """
        rng: numpy random number generator
        input: theano.tensor.tensor4
               the sentence matrix, (number of instances, number of input feature maps,  embedding dimension, number of words)
        
        filter_shape: tuple of length 4, 
           dimension: (number of filters, num input feature maps, filter height, filter width)
        
        k: int or theano.tensor.iscalar,
           the k value in the max-pooling layer

        W: theano.tensor.tensor4,
           the filter weight matrices, 
           dimension: (number of filters, num input feature maps, filter height, filter width)

        b: theano.tensor.vector,
           the filter bias, 
           dimension: (filter number, )
                
        """
        self.input = input
        self.filter_shape = filter_shape

        if W is not None:
            W_val = W
        else:
            fan_in = np.prod(filter_shape[1:])
            
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / 
                       k) # it's 
            
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            
            W_val = np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(
            value = np.asarray(W_val,
                               dtype = theano.config.floatX),
            name = "W",
            borrow=True
        )
        
        # make b
        if b is not None:
            b_val = b
        else:
            b_size = (filter_shape[0], )
            b_val = rng.uniform(
                low = -.5,
                high = .5,
                size = b_size
            )
            
        self.b = theano.shared(
            value = np.asarray(
                b_val,
                dtype = theano.config.floatX
            ),
            name = "b",
            borrow = True
        )

        self.params = [self.W, self.b]
        self.param_shapes = [filter_shape,
                             b_size ]

    def fold(self, x):
        """
        :type x: theano.tensor.tensor4
        """
        return (x[:, :, T.arange(0, x.shape[2], 2)] + 
                x[:, :, T.arange(1, x.shape[2], 2)]) / 2
        
    def k_max_pool(self, x, k):
        """
        perform k-max pool on the input along the rows

        input: theano.tensor.tensor4
           
        k: theano.tensor.iscalar
            the k parameter

        Returns: 
        4D tensor
        """
        ind = T.argsort(x, axis = 3)

        sorted_ind = T.sort(ind[:,:,:, -k:], axis = 3)
        
        dim0, dim1, dim2, dim3 = sorted_ind.shape
        
        indices_dim0 = T.arange(dim0).repeat(dim1 * dim2 * dim3)
        indices_dim1 = T.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis=1).T.flatten()
        indices_dim2 = T.arange(dim2).repeat(dim3).reshape((dim2*dim3, 1)).repeat(dim0 * dim1, axis = 1).T.flatten()
        
        return x[indices_dim0, indices_dim1, indices_dim2, sorted_ind.flatten()].reshape(sorted_ind.shape)

    def output(self, k):
        # non-linear transform of the convolution output
        conv_out = T.nnet.conv.conv2d(self.input, 
                                      self.W, 
                                      border_mode = "full") 
        
        # fold
        fold_out = self.fold(conv_out)
                
        # k-max pool        
        pool_out = (self.k_max_pool(fold_out, k) + 
                    self.b.dimshuffle('x', 0, 'x', 'x'))

        # non-linearity
        return T.tanh(pool_out)

def train_and_test(
        learning_rate = 0.1,
        epsilon = 0.0001,
        rho = 0.95,
        nkerns = [6, 12],
        embed_dm = 48,
        k_top = 5,
        n_hidden = 500,
        batch_size = 500,
        n_epochs = 2000):

    ###################
    # get the data    #
    ###################
    datasets = util.stanford_sentiment('data/stanfordSentimentTreebank/trees/processed.pkl',
                                       corpus_folder = 'data/stanfordSentimentTreebank/trees/')
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    word2index = datasets[3]
    index2word = datasets[4]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    train_sent_len = train_set_x.get_value(borrow=True).shape[1]
    possible_labels =  set(train_set_y.get_value().tolist())
    
    
    ###################################
    # Symbolic variable definition    #
    ###################################
    x = T.imatrix('x') # the word indices matrix
    sent_len = x.shape[1]
    y = T.ivector('y') # the sentiment labels

    batch_index = T.iscalar('batch_index')
    
    rng = np.random.RandomState(1234)        
        
    
    ###############################
    # Construction of the network #
    ###############################

    # Layer 1, the embedding layer
    layer1 = WordEmbeddingLayer(rng, 
                                input = x, 
                                vocab_size = len(word2index), 
                                embed_dm = embed_dm)    
    
    # Layer 2: convolution&fold&pool layer
    filter_shape = (nkerns[0],
                    1, 
                    1, 10
    )
    
    layer2_k = int(max(k_top, 
                       math.ceil(.5 * train_sent_len)))

    layer2 = ConvFoldingPoolLayer(rng, 
                                  input = layer1.output, 
                                  filter_shape = filter_shape, 
                                  k = layer2_k)
    
    # Layer 3: convolution&fold&pool layer
    filter_shape = (nkerns[1], 
                    nkerns[0],
                    1, 7 
    )
    
    layer3 = ConvFoldingPoolLayer(rng, 
                                  input = layer2.output(layer2_k),
                                  filter_shape = filter_shape, 
                                  k = k_top)
    
    # Hiddne layer: dropout layer
    layer4 = DropoutLayer(
        input = layer3.output(k_top),
        rng = rng, 
        dropout_rate = 0.5
    )
    
    layer4_input = layer4.output.flatten(2) #make it into a row 
    
    
    # Softmax Layer
    model = LogisticRegression(
        input = layer4_input, 
        n_in = nkerns[1] * k_top * embed_dm / 4, # we fold twice, so divide by 4
        n_out = len(possible_labels) # five sentiment level
    )

    ############################
    # Training function and    #
    # AdaDelta learning rate   #
    ############################
    cost = model.nnl(y)
        
    params = (layer1.params + layer2.params + layer3.params + model.params)
    param_shapes=  (layer1.param_shapes + layer2.param_shapes + layer3.param_shapes + model.param_shapes)
    
    # AdaDelta parameter symbols
    # E[g^2]
    # initialized to zero
    egs = [
        theano.shared(
            value = np.zeros(param_shape,
                             dtype = theano.config.floatX
                         ),
            borrow = True,        
            name = "Eg:" + param.name
        )
        for param_shape, param in zip(param_shapes, params)
    ]
    
    # E[\delta x^2]
    # initialized to zero
    exs = [
        theano.shared(
            value = np.zeros(param_shape,
                             dtype = theano.config.floatX
                         ),
            borrow = True,        
            name = "Ex:" + param.name
        )
        for param_shape, param in zip(param_shapes, params)
    ]
    
    param_grads = [T.grad(cost, param) for param in params]
    
    # AdaDelta parameter update
    # Update E[g^2]

    # updates = [
    #     (eg, rho * eg + (1 - rho) * T.pow(param_grad, 2))
    #     for eg, param_grad, param_shape in zip(egs, param_grads, param_shapes)
    # ]
    
    # # More updates for the gradients
    # param_updates = [
    #     (param, param - (T.sqrt(ex + epsilon) / T.sqrt(eg + epsilon)) * param_grad)
    #     for eg, ex, param, param_grad in zip(egs, exs, params, param_grads)
    # ]
    
    # updates +=  param_updates
    
    # # Last, updates for E[x^2]
    # updates += [
    #     (ex, rho * ex + (1 - rho) * T.pow(param_update[1], 2))
    #     for ex, param_update in zip(exs, param_updates)
    # ]

    updates = [
        (param, param - 0.01 * param_grad)
        for param, param_grad in zip(params, param_grads)
    ]
    
    train_model = theano.function(inputs = [batch_index],
                                  outputs = cost, 
                                  updates = updates,
                                  givens = {
                                      x: train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                                      y: train_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                                  }
    )
        
    
    valid_model = theano.function(inputs = [],
                                  outputs = model.errors(y), 
                                  givens = {
                                      x: valid_set_x,
                                      y: valid_set_y
                                  }
    )
    
    #the training loop
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0

    start_time = time.clock()
    done_looping = False
    epoch = 0
    
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        print "At epoch {0}".format(epoch)
        
        for minibatch_index in xrange(n_train_batches):

            train_error = train_model(minibatch_index)
            
            layer1.normalize() # normalize the word embedding
            

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            # if (iter + 1) % validation_frequency == 0:
                
            dev_error = valid_model()
            print "At epoch %d and minibatch %d. Dev error %.2f%%" %(
                epoch, 
                minibatch_index,
                dev_error * 100
            )
        
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    
if __name__ == "__main__":
    train_and_test(learning_rate = 0.1)
