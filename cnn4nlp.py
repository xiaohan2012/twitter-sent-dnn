"""
CNN for sentence modeling described in:
A Convolutional Neural Network for Modeling Sentence
"""
import sys, os, time
import pdb

import math, random
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
        
        embedding_val = np.asarray(
            rng.normal(0, 0.05, (vocab_size, embed_dm)), 
            dtype = theano.config.floatX
        )
        
        embedding_val[vocab_size-1,:] = 0 # the <PAD> character is intialized to 0
        
        self.embeddings = theano.shared(
            np.asarray(embedding_val, 
                       dtype = theano.config.floatX),
            borrow = True,
            name = 'embeddings'
        )

        
        self.params = [self.embeddings]
        
        self.param_shapes = [(vocab_size, embed_dm)]
        
        # Return:
        
        # :type, theano.tensor.tensor4
        # :param, dimension(1, 1, word embedding dimension, number of words in sentence)
        #         made to be 4D to fit into the dimension of convolution operation
        sent_embedding_list, updates = theano.map(lambda sent: self.embeddings[sent], 
                                                  input)
        sent_embedding_tensor = T.stacklists(sent_embedding_list) # make it into a 3D tensor
        
        self.output = sent_embedding_tensor.dimshuffle(0, 'x', 2, 1) # make it a 4D tensor
                    
class ConvFoldingPoolLayer(object):
    """
    Convolution, folding and k-max pooling layer
    """
    def __init__(self, 
                 rng, 
                 input,
                 filter_shape,
                 k,
                 activation,
                 fan_in_fan_out = True,
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

        activation: str
           the activation unit type, `tanh` or `relu`

        fan_in_fan_out: bool
           whether use fan-in fan-out initialization or not. Default, True
           If not True, use `normal(0, 0.05, size)`

        W: theano.tensor.tensor4,
           the filter weight matrices, 
           dimension: (number of filters, num input feature maps, filter height, filter width)

        b: theano.tensor.vector,
           the filter bias, 
           dimension: (filter number, )
                
        """
        
        self.input = input
        self.k = k
        self.filter_shape = filter_shape

        assert activation in ('tanh', 'relu')
        self.activation = activation
        
        if W is not None:
            self.W = W
        else:
            if fan_in_fan_out:
                # use fan-in fan-out init
                fan_in = np.prod(filter_shape[1:])
                
                fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / 
                           k) # it's 
                
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                
                W_val = np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                )
            else:
                # normal initialization
                W_val = np.asarray(
                    rng.normal(0, 0.05, size = filter_shape),
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
            b_size = b.shape
            self.b = b
        else:
            b_size = (filter_shape[0], )
            b_val = np.zeros(b_size)
            
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
        
    @property
    def output(self):
        # non-linear transform of the convolution output
        conv_out = T.nnet.conv.conv2d(self.input, 
                                      self.W, 
                                      border_mode = "full") 
        
        # fold
        fold_out = self.fold(conv_out)
                
        # k-max pool        
        pool_out = (self.k_max_pool(fold_out, self.k) + 
                    self.b.dimshuffle('x', 0, 'x', 'x'))

        
        if self.activation == "tanh":
            return T.tanh(pool_out)
        else:
            return T.switch(pool_out > 0, pool_out, 0)

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

def train_and_test(
        lr_update_method = "adadelta",
        use_L2_reg = True,
        learning_rate = 0.1,
        fan_in_fan_out = True,
        delay_embedding_learning = True,
        conv_activation_unit = "tanh", 
        epsilon = 0.000001,
        rho = 0.95,
        gamma = 0.1,
        embed_dm = 48,        
        k_top = 5,
        L2_regs= [0.00001, 0.0003, 0.0003, 0.0001],
        n_hidden = 500,
        batch_size = 500,
        n_epochs = 2000, 
        dropout_switches = [True, True, True], 
        dropout_rates = [0.2, 0.5, 0.5],
        conv_layer_n = 2,
        nkerns = [6, 12],
        conv_sizes = [10, 7],
        print_config = {}
):
    if lr_update_method:
        assert lr_update_method in ("adadelta", "adagrad")

    assert conv_layer_n == len(conv_sizes) == len(nkerns) == (len(L2_regs) - 2)

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
    
    dropout_layers = [layer1]
    layers = [layer1]
    
    for i in xrange(conv_layer_n):
        
        # for the dropout layer
        dpl = DropoutLayer(
            input = dropout_layers[-1].output,
            rng = rng, 
            dropout_rate = dropout_rates[0]
        ) 
        next_layer_dropout_input = dpl.output
        next_layer_input = layers[-1].output
        
        # for the conv layer
        filter_shape = (
            nkerns[i],
            (1 if i == 0 else nkerns[i-1]), 
            1, 
            conv_sizes[i]
        )
        
        k = int(max(k_top, 
                    math.ceil((conv_layer_n - float(i+1)) / conv_layer_n * train_sent_len)))
        
        print "For conv layer(%s) %d, filter shape = %r, k = %d, dropout_rate = %f and fan_in_fan_out: %r" %(
            conv_activation_unit, 
            i+2, 
            filter_shape, 
            k, 
            dropout_rates[i], 
            fan_in_fan_out
        )
        
        # we have two layers adding to two paths repsectively, 
        # one for training
        # the other for prediction(averaged model)

        dropout_conv_layer = ConvFoldingPoolLayer(rng, 
                                                  input = next_layer_dropout_input,
                                                  filter_shape = filter_shape, 
                                                  k = k, 
                                                  fan_in_fan_out = fan_in_fan_out,
                                                  activation = conv_activation_unit)
    
        # for prediction
        # sharing weight with dropout layer
        conv_layer = ConvFoldingPoolLayer(rng, 
                                          input = next_layer_input,
                                          filter_shape = filter_shape,
                                          k = k,
                                          activation = conv_activation_unit,
                                          W = dropout_conv_layer.W * (1 - dropout_rates[i]), # model averaging
                                          b = dropout_conv_layer.b
        )

        dropout_layers.append(dropout_conv_layer)
        layers.append(conv_layer)
    
    # last, the output layer
    # both dropout and without dropout
    n_in = nkerns[-1] * k_top * embed_dm / (len(nkerns)*2)
    print "For output layer, n_in = %d, dropout_rate = %f" %(n_in, dropout_rates[-1])
    
    dropout_output_layer = LogisticRegression(
        rng,
        input = dropout_layers[-1].output.flatten(2), 
        n_in = n_in, # divided by 2x(how many times are folded)
        n_out = len(possible_labels) # five sentiment level
    )

    output_layer = LogisticRegression(
        rng,
        input = layers[-1].output.flatten(2), 
        n_in = n_in,
        n_out = len(possible_labels),
        W = dropout_output_layer.W * (1 - dropout_rates[-1]), # sharing the parameters, don't forget
        b = dropout_output_layer.b
    )
    
    dropout_layers.append(dropout_output_layer)
    layers.append(output_layer)

    ###############################
    # Parameters to be used       #
    ##############################
    if not delay_embedding_learning:
        print "Immediate embedding learning. "
        param_layers = dropout_layers
    else:
        print "Delay embedding learning."
        param_layers = dropout_layers[1:] # exclude the embedding layer
    print "param_layers: %r" %param_layers
        
    params = [param for layer in param_layers for param in layer.params]
    param_shapes=  [param for layer in param_layers for param in layer.param_shapes]


    # cost and error come from different model!
    dropout_cost = dropout_output_layer.nnl(y)
    errors = output_layer.errors(y)
    
    ############################
    # L2 regularizer           #
    ############################
    
    L2_sqr = T.sum([
        L2_reg / 2 * ((layer.W if hasattr(layer, "W") else layer.embeddings) ** 2).sum()
        for L2_reg, layer in zip(L2_regs, param_layers)
    ])
    
    ############################
    # Training function and    #
    # AdaDelta learning rate   #
    ############################
    if use_L2_reg:
        cost = dropout_cost + L2_sqr
    else:
        cost = dropout_cost
        

    param_grads = [T.grad(cost, param) for param in params]
    
    if lr_update_method == "adadelta":
        # AdaDelta parameter update
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
    
        print "Using AdaDelta with rho = %f and epsilon = %f" %(rho, epsilon)
        
        new_egs = [
            rho * eg + (1 - rho) * g ** 2
            for eg, g in zip(egs, param_grads)
        ]
        
        delta_x = [
            -(T.sqrt(ex + epsilon) / T.sqrt(new_eg + epsilon)) * g
            for new_eg, ex, g in zip(new_egs, exs, param_grads)
        ]
        new_exs = [
            rho * ex + (1 - rho) * (dx ** 2)
            for ex, dx in zip(exs, delta_x)
        ]

        egs_updates = zip(egs, new_egs)
        exs_updates = zip(exs, new_exs)
        param_updates = [
            (p, p + dx*g)
            for dx, g, p in zip(delta_x, param_grads, params)
        ]
        
        updates = egs_updates + exs_updates + param_updates
    elif lr_update_method == "adagrad":
        print "Using AdaGrad with gamma = %f" %(gamma)
        grad_hists = [
            theano.shared(
                value = np.zeros(param_shape,
                                 dtype = theano.config.floatX
                             ),
                borrow = True,        
                name = "grad_hist:" + param.name
            )
            for param_shape, param in zip(param_shapes, params)
        ]
        
        new_grad_hists = [
            g_hist + g ** 2
            for g_hist, g in zip(grad_hists, param_grads)
        ]
        
        sqs = [
            T.sqrt(g_hist)
            for g_hist in new_grad_hists
        ]
                
        
        sqs = [
            T.set_subtensor(sq[sq != 0], gamma / sq[sq != 0])  #same as, sq[sq != 0] = gamma / sq[sq != 0]
            for sq in sqs
        ]

        param_updates = [
            (param, param - sq * param_grad)
            for sq, param, param_grad in zip(sqs, params, param_grads)
        ]

        grad_hist_update = zip(grad_hists, new_grad_hists)

        updates = grad_hist_update + param_updates
    else:
        print "Using const learning rate: %f" %(learning_rate)
        updates = [
            (param, param - learning_rate * param_grad)
            for param, param_grad in zip(params, param_grads)
        ]
        
    train_model = theano.function(inputs = [batch_index],
                                  outputs = [cost], 
                                  updates = updates,
                                  givens = {
                                      x: train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                                      y: train_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                                  },
    )

        
    train_error = theano.function(inputs = [],
                                  outputs = errors, 
                                  givens = {
                                      x: train_set_x,
                                      y: train_set_y
                                  }, 
    )

    valid_error = theano.function(inputs = [],
                                  outputs = errors, 
                                  givens = {
                                      x: valid_set_x,
                                      y: valid_set_y
                                  }, 
                                  # mode = "DebugMode"
    )
    
    #############################
    # Debugging purpose code    #
    #############################
    # : PARAMETER TUNING NOTE:
    # some demonstration of the gradient vanishing probelm
    
    if print_config["nnl"]:
        get_nnl = theano.function(
            inputs = [batch_index],
            outputs = dropout_cost,
            givens = {
                x: train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                y: train_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            }
        )
        
    if print_config["L2_sqr"]:
        get_L2_sqr = theano.function(
            inputs = [],
            outputs = L2_sqr
        )
        
    if print_config["grad_abs_mean"]:
        print_grads = theano.function(
            inputs = [], 
            outputs = [theano.printing.Print(param.name)(
                T.mean(T.abs_(param_grad))
            )
                       for param, param_grad in zip(params, param_grads)
                   ], 
            givens = {
                x: train_set_x,
                y: train_set_y
            }
        )
    if print_config["adadelta_lr_mean"]:
        print_adadelta_lr_mean = theano.function(
            inputs = [],
            outputs = [
                theano.printing.Print("adadelta mean:" +eg.name)(
                    T.mean(T.sqrt(ex + epsilon) / T.sqrt(eg + epsilon))
                )
                for eg, ex in zip(egs, exs)
            ]
        )

    if print_config["adagrad_lr_mean"]:
        print_adagrad_lr_mean = theano.function(
            inputs = [],
            outputs = [
                theano.printing.Print("adagrad mean")(
                    T.mean(sq)
                )
                for sq in sqs
            ]
        )
        
    if print_config["embeddings"]:
        print_embeddings = theano.function(
            inputs = [],
            outputs = theano.printing.Print("embeddings")(layers[0].embeddings)
        )
    
    if print_config["logreg_W"]:
        print_logreg_W = theano.function(
            inputs = [],
            outputs = theano.printing.Print(layers[-1].W.name)(layers[-1].W)
        )
        
    if print_config["logreg_b"]:
        print_logreg_b = theano.function(
            inputs = [],
            outputs = theano.printing.Print(layers[-1].b.name)(layers[-1].b)
        )

    if print_config["conv_layer1_W"]:
        print_convlayer1_W = theano.function(
            inputs = [],
            outputs = theano.printing.Print(layers[1].W.name)(layers[1].W)
        )

    if print_config["conv_layer2_W"]:
        print_convlayer2_W = theano.function(
            inputs = [],
            outputs = theano.printing.Print(layers[2].W.name)(layers[2].W)
        )

    if print_config["p_y_given_x"]:
        print_p_y_given_x = theano.function(
            inputs = [],
            outputs = theano.printing.Print("p_y_given_x")(layers[-1].p_y_given_x),
            givens = {
                x: train_set_x
            }
        )
    
    param_n = sum([1. for l in param_layers for p in l.params])
    if print_config["print_param_weight_mean"]:
        get_param_weight_mean = theano.function(
            inputs = [], 
            outputs = T.sum([T.sum(T.abs_(p)) 
                             for l in param_layers for p in l.params]) / param_n
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
    
    nnls = []
    L2_sqrs = []
    
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        print "At epoch {0}".format(epoch)

        # shuffle the training data        
        train_set_x_data = train_set_x.get_value(borrow = True)
        train_set_y_data = train_set_y.get_value(borrow = True)        
        
        permutation = np.random.permutation(train_set_x.get_value(borrow=True).shape[0])

        train_set_x.set_value(train_set_x_data[permutation])
        train_set_y.set_value(train_set_y_data[permutation])
        
        for minibatch_index in xrange(n_train_batches):
           
            train_cost = train_model(minibatch_index)

            if print_config["nnl"]:
                nnls.append(get_nnl(minibatch_index))
                
            if print_config["L2_sqr"]:
                L2_sqrs.append(get_L2_sqr())            
                
            # print_grads(minibatch_index)
            # print_learning_rates()
            # print_embeddings()
            # print_logreg_param()
            
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (minibatch_index+1) % 50 == 0 or minibatch_index == n_train_batches - 1:
                print "%d / %d minibatches completed" %(minibatch_index + 1, n_train_batches)                

            if (iter + 1) % validation_frequency == 0:
                if print_config["nnl"]:
                    print "`nnl` for the past 50 minibatches is %f" %(np.mean(np.array(nnls)))
                    nnls = []
                if print_config["L2_sqr"]:
                    print "`L2_sqr`` for the past 50 minibatches is %f" %(np.mean(np.array(L2_sqrs)))
                    L2_sqrs = []
                if print_config["print_param_weight_mean"]:
                    print "weight mean %f: " %(get_param_weight_mean())

                if print_config["conv_layer2_W"]:
                    print_convlayer2_W()

                if print_config["conv_layer1_W"]:
                    print_convlayer1_W()

                if print_config["p_y_given_x"]:
                    print_p_y_given_x()

                if print_config["adadelta_lr_mean"]:
                    print_adadelta_lr_mean()

                if print_config["adagrad_lr_mean"]:
                    print_adagrad_lr_mean()

                if print_config["grad_abs_mean"]:
                    print_grads()
                
                if print_config["logreg_W"]:
                    print_logreg_W()

                if print_config["logreg_b"]:
                    print_logreg_b()

                print "At epoch %d and minibatch %d. \nTrain error %.2f%%\nDev error %.2f%%\n" %(
                    epoch, 
                    minibatch_index,
                    train_error() * 100, 
                    valid_error() * 100
                )
    
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    
if __name__ == "__main__":
    print_config = {
        "adadelta_lr_mean": 0,
        "adagrad_lr_mean": 0,
        "logreg_W": 0,
        "logreg_b": 0,
        "conv_layer2_W": 0,
        "conv_layer1_W": 0,
        "grad_abs_mean": 0,
        "p_y_given_x": 1,
        "embeddings": 0,
        "nnl": 1,
        "L2_sqr": 1,
        "print_param_weight_mean": 0,
    }
    
    train_and_test(
        lr_update_method = "adagrad",
        use_L2_reg = True, 
        L2_regs= [0.00001, 0.0003, 0.0003, 0.0001],
        fan_in_fan_out = False,
        delay_embedding_learning = True,
        conv_activation_unit = "relu", 
        learning_rate = 0.0001, 
        batch_size = 10, 
        print_config = print_config, 
        dropout_switches = [False, False, False], 
        dropout_rates = [0.2, 0.5, 0.5]
    )
