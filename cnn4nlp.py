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
                 vocab_size, 
                 embed_dm):
        """
        vocab_size: integer, the size of vocabulary,

        embed_dm: integer, the dimension of word vector representation
        """
        self.embeddings = theano.shared(
            np.asarray(rng.uniform(
                low = -0.1,
                high = 0.1,
                size = (vocab_size, embed_dm)
            ), 
                       dtype = theano.config.floatX),
            borrow = True,
            name = 'embeddings'
        )
        
        self.params = [self.embeddings]
        
        self.normalize = theano.function(inputs = [],
                                         updates = { self.embeddings:
                                                     (self.embeddings / 
                                                      T.sqrt((self.embeddings**2).sum(axis=1)).dimshuffle(0,'x'))
                                                 }
        )

        self.normalize() #initial normalization

    def output(self, word_indices):
        """
        word_indices: theano.tensor.ivector, word indices of the sentence

        Return:
        
        :type, theano.tensor.tensor4
        :param, dimension(1, 1, word embedding dimension, number of words in sentence)
                made to be 4D to fit into the dimension of convolution operation
        """
        return (self.embeddings[word_indices]).dimshuffle('x', 'x', 1, 0)
        
class ConvFoldingPoolLayer(object):
    """
    Convolution, folding and k-max pooling layer
    """
    def __init__(self, 
                 rng, 
                 input,
                 filter_shape,
                 W = None,
                 b = None):
        """
        rng: numpy random number generator
        input: theano.tensor.tensor4
               the sentence matrix, (1, number of input feature maps,  embedding dimension, number of words)
        
        filter_shape: tuple of length 4, 
           dimension: (number of filters, num input feature maps, filter height, filter width)
        
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
                       5.) # for now, assume the sentence average length is 5
            
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
            borrow = True)

        self.params = [self.W, self.b]
        
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
        
        Returns: 
        4D tensor
        """
        
        x0 = x[0] # we only use the first element
        ind = T.argsort(x0, axis = 2)

        sorted_ind = T.sort(ind[:,:, 0:k], axis = 2)

        ind = T.argsort(x0, axis = 2)

        sorted_ind = T.sort(ind[:, :, -k:], axis = 2)

        first = T.arange(x0.shape[0]).repeat(x0.shape[1] * k)
        second = T.tile(
            T.arange(x0.shape[1])
            .repeat(k)
            .dimshuffle((0, 'x')), # make it into a 2d matrix
            (self.filter_shape[0], 1)
        ).reshape((1, T.prod(sorted_ind.shape)))


        return (
            x0[first, 
               second, 
               sorted_ind.flatten()]
            .reshape(sorted_ind.shape)
            .dimshuffle('x', 0, 1, 2) # make it back to 4D
        )

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

def train_and_test(learning_rate, 
                   nkerns = [6, 14],
                   embed_dm = 48,
                   k_top = 4,
                   L1_reg=0.00, L2_reg=0.0001,
                   n_hidden = 500,
                   n_epochs = 2000):

    # get the data
    datasets = util.stanford_sentiment('data/stanfordSentimentTreebank/trees/processed.pkl.3',
                                       corpus_folder = 'data/stanfordSentimentTreebank/trees/')
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    word2index = datasets[3]
    index2word = datasets[4]

    possible_labels =  set(train_set_y.tolist())
    
    word_indices = T.ivector('word_indices') # the vector of word indices
    sent_label = T.iscalar('sent_label') # the sentence sentiment label
    
    rng = np.random.RandomState(1234)
    
    # Layer 1, the embedding layer
    layer1 = WordEmbeddingLayer(rng, len(word2index), 
                                embed_dm = embed_dm)
    
    # Layer 2: convolution&fold&pool layer
    filter_shape = (nkerns[0],
                    1, 
                    1, 7 # will 7 be too large?
    )
    
    layer2 = ConvFoldingPoolLayer(rng, 
                                  input = layer1.output(word_indices), 
                                  filter_shape = filter_shape)
    layer2_k = T.cast(T.max([k_top, 
                             T.ceil(.5 * word_indices.shape[0])]), 
                      'int32'
                  ) # to int32 for slicing
    
    # Layer 3: convolution&fold&pool layer
    filter_shape = (nkerns[1], 
                    nkerns[0],
                    1, 5 # will 5 be too large?
    )
    
    layer3 = ConvFoldingPoolLayer(rng, 
                                  input = layer2.output(layer2_k),
                                  filter_shape = filter_shape)
    
    layer4_input = (layer3
                    .output(k_top)
                    .flatten(2)) #make it into a row 
    
    # Softmax Layer
    model = LogisticRegression(
        input = layer4_input, 
        n_in = nkerns[1] * k_top * embed_dm / 4, # we fold twice, so divide by 4
        n_out = len(possible_labels) # five sentiment level
    )
           
    # Using model.nnl will cause unkown optimization error..
    # so be more direct
    cost = -T.log(model.p_y_given_x)[0, sent_label]
        
    # theano.printing.debugprint(cost)
    
    # pdb.set_trace()
    params = (layer1.params + layer2.params + layer3.params + model.params)

    updates = [(param, param - learning_rate * T.grad(cost, param))
               for param in params] # to be filled    
    
    train_model = theano.function(inputs = [word_indices, sent_label],
                                  outputs = cost, 
                                  updates = updates,
                                  mode = THEANO_COMPILE_MODE, 
                                  # mode = "FAST_RUN", 
    )
        
    # given sentence and its label, predict the sentiment label
    classify = theano.function(inputs = [word_indices],
                               outputs = model.pred_y, 
    )
    
    def accuracy(xs, ys):
        correct_n = len([1 for x, y in zip(xs, ys) if classify(x) == y])
        return correct_n / float(len(xs))
    
    # sys.exit(-1)
    #the training loop
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant

    best_validation_loss = np.inf
    best_iter = 0

    start_time = time.clock()
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        print "At epoch {epoch}".format(epoch = epoch)

        for train_instance_index in xrange(len(train_set_x)):

            train_err = train_model(
                train_set_x[train_instance_index], 
                train_set_y[train_instance_index]
            )
            
            layer1.normalize() # normalize the word embedding
            
            if train_instance_index % 100 == 0 or train_instance_index == len(train_set_x) - 1:
                print "%d / %d instances finished" %(
                    train_instance_index,
                    len(train_set_x)
                )
        
                print "validation error %.2f %%" %(
                    accuracy(
                        valid_set_x, 
                        valid_set_y
                    )
                )

                print "test error %.2f %%" %(
                    accuracy(
                        test_set_x, 
                        test_set_y
                    )
                )
            
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    
if __name__ == "__main__":
    train_and_test(0.001)
