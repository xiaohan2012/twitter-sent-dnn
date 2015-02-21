"""
CNN for sentence modeling described in paper:

A Convolutional Neural Network for Modeling Sentence

"""
import sys, os, time
import pdb

import math, random
import numpy as np
import theano
import theano.tensor as T
from util import (load_data, dump_params)

from logreg import LogisticRegression

class WordEmbeddingLayer(object):
    """
    Layer that takes input vectors, output the sentence matrix
    """
    def __init__(self, rng, 
                 input,
                 vocab_size, 
                 embed_dm, 
                 embeddings = None,
    ):
        """
        input: theano.tensor.dmatrix, (number of instances, sentence word number)
        
        vocab_size: integer, the size of vocabulary,

        embed_dm: integer, the dimension of word vector representation

        embeddings: theano.tensor.TensorType
        pretrained embeddings
        """                
        if embeddings:
            print "Use pretrained embeddings: ON"
            assert embeddings.get_value().shape == (vocab_size, embed_dm), "%r != %r" %(
                embeddings.get_value().shape, 
                (vocab_size, embed_dm)
            )
            
            self.embeddings = embeddings
        else:
            print "Use pretrained embeddings: OFF"
            embedding_val = np.asarray(
                rng.normal(0, 0.05, size = (vocab_size, embed_dm)), 
                dtype = theano.config.floatX
            )
            
            embedding_val[vocab_size-1,:] = 0 # the <PADDING> character is intialized to 0
            
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
                 activation = "tanh",
                 norm_w = True,
                 fold = 0,
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
           the activation unit type, `tanh` or `relu` or 'sigmoid'

        norm_w: bool
           whether use fan-in fan-out initialization or not. Default, True
           If not True, use `normal(0, 0.05, size)`

        fold: int, 0 or 1
           fold or not

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
        self.fold_flag = fold

        assert activation in ('tanh', 'relu', 'sigmoid')
        self.activation = activation
        
        if W is not None:
            self.W = W
        else:
            if norm_w:
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

        if self.fold_flag:
            # fold
            fold_out = self.fold(conv_out)
        else:
            fold_out = conv_out

        # k-max pool        
        pool_out = (self.k_max_pool(fold_out, self.k) + 
                    self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # around 0.
        # why tanh becomes extreme?
        
        if self.activation == "tanh":
            # return theano.printing.Print("tanh(pool_out)")(T.tanh(pool_out))
            return T.tanh(pool_out)
        elif self.activation == "sigmoid":
            return T.nnet.sigmoid(pool_out)
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

def train_and_test(args, print_config):

    assert args.conv_layer_n == len(args.filter_widths) == len(args.nkerns) == (len(args.L2_regs) - 2) == len(args.fold_flags) == len(args.ks)

    # \mod{dim, 2^{\sum fold_flags}} == 0
    assert args.embed_dm % (2 ** sum(args.fold_flags)) == 0
    
    ###################
    # get the data    #
    ###################
    datasets = load_data(args.corpus_path)
    
    train_set_x, train_set_y = datasets[0]
    dev_set_x, dev_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    word2index = datasets[3]
    index2word = datasets[4]
    pretrained_embeddings = datasets[5]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / args.batch_size
    n_dev_batches = dev_set_x.get_value(borrow=True).shape[0] / args.dev_test_batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / args.dev_test_batch_size
    
    train_sent_len = train_set_x.get_value(borrow=True).shape[1]
    possible_labels =  set(train_set_y.get_value().tolist())
    
    if args.use_pretrained_embedding:
        args.embed_dm = pretrained_embeddings.get_value().shape[1]
        
    ###################################
    # Symbolic variable definition    #
    ###################################
    x = T.imatrix('x') # the word indices matrix
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
                                embed_dm = args.embed_dm, 
                                embeddings = (
                                    pretrained_embeddings 
                                    if args.use_pretrained_embedding else None
                                )
    )
    
    dropout_layers = [layer1]
    layers = [layer1]
    
    for i in xrange(args.conv_layer_n):
        fold_flag = args.fold_flags[i]
        
        # for the dropout layer
        dpl = DropoutLayer(
            input = dropout_layers[-1].output,
            rng = rng, 
            dropout_rate = args.dropout_rates[0]
        ) 
        next_layer_dropout_input = dpl.output
        next_layer_input = layers[-1].output
        
        # for the conv layer
        filter_shape = (
            args.nkerns[i],
            (1 if i == 0 else args.nkerns[i-1]), 
            1, 
            args.filter_widths[i]
        )
        
        k = args.ks[i]
        
        print "For conv layer(%s) %d, filter shape = %r, k = %d, dropout_rate = %f and normalized weight init: %r and fold: %d" %(
            args.conv_activation_unit, 
            i+2, 
            filter_shape, 
            k, 
            args.dropout_rates[i], 
            args.norm_w, 
            fold_flag
        )
        
        # we have two layers adding to two paths repsectively, 
        # one for training
        # the other for prediction(averaged model)

        dropout_conv_layer = ConvFoldingPoolLayer(rng, 
                                                  input = next_layer_dropout_input,
                                                  filter_shape = filter_shape, 
                                                  k = k, 
                                                  norm_w = args.norm_w,
                                                  fold = fold_flag,
                                                  activation = args.conv_activation_unit)
    
        # for prediction
        # sharing weight with dropout layer
        conv_layer = ConvFoldingPoolLayer(rng, 
                                          input = next_layer_input,
                                          filter_shape = filter_shape,
                                          k = k,
                                          activation = args.conv_activation_unit,
                                          fold = fold_flag,
                                          W = dropout_conv_layer.W * (1 - args.dropout_rates[i]), # model averaging
                                          b = dropout_conv_layer.b
        )

        dropout_layers.append(dropout_conv_layer)
        layers.append(conv_layer)
    
    # last, the output layer
    # both dropout and without dropout
    if sum(args.fold_flags) > 0:
        n_in = args.nkerns[-1] * args.ks[-1] * args.embed_dm / (2**sum(args.fold_flags))
    else:
        n_in = args.nkerns[-1] * args.ks[-1] * args.embed_dm
        
    print "For output layer, n_in = %d, dropout_rate = %f" %(n_in, args.dropout_rates[-1])
    
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
        W = dropout_output_layer.W * (1 - args.dropout_rates[-1]), # sharing the parameters, don't forget
        b = dropout_output_layer.b
    )
    
    dropout_layers.append(dropout_output_layer)
    layers.append(output_layer)

    ###############################
    # Error and cost              #
    ###############################
    # cost and error come from different model!
    dropout_cost = dropout_output_layer.nnl(y)
    errors = output_layer.errors(y)
    
    def prepare_L2_sqr(param_layers, L2_regs):
        assert len(L2_regs) == len(param_layers)
        return T.sum([
            L2_reg / 2 * ((layer.W if hasattr(layer, "W") else layer.embeddings) ** 2).sum()
            for L2_reg, layer in zip(L2_regs, param_layers)
        ])
    L2_sqr = prepare_L2_sqr(dropout_layers, args.L2_regs)
    L2_sqr_no_ebd = prepare_L2_sqr(dropout_layers[1:], args.L2_regs[1:])
    
    if args.use_L2_reg:
        cost = dropout_cost + L2_sqr
        cost_no_ebd = dropout_cost + L2_sqr_no_ebd
    else:
        cost = dropout_cost
        cost_no_ebd = dropout_cost
    
    ###############################
    # Parameters to be used       #
    ###############################
    print "Delay embedding learning by %d epochs" %(args.embedding_learning_delay_epochs)
        
    print "param_layers: %r" %dropout_layers
    param_layers = dropout_layers
    
    ##############################
    # Parameter Update           #
    ##############################
    print "Using AdaDelta with rho = %f and epsilon = %f" %(args.rho, args.epsilon)
    
    params = [param for layer in param_layers for param in layer.params]
    param_shapes=  [param for layer in param_layers for param in layer.param_shapes]                                
    
    param_grads = [T.grad(cost, param) for param in params]
        
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
    
    # E[\delta x^2], initialized to zero
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
    
    new_egs = [
        args.rho * eg + (1 - args.rho) * g ** 2
        for eg, g in zip(egs, param_grads)
    ]
        
    delta_x = [
        -(T.sqrt(ex + args.epsilon) / T.sqrt(new_eg + args.epsilon)) * g
        for new_eg, ex, g in zip(new_egs, exs, param_grads)
    ]    
    
    new_exs = [
        args.rho * ex + (1 - args.rho) * (dx ** 2)
        for ex, dx in zip(exs, delta_x)
    ]    
    
    egs_updates = zip(egs, new_egs)
    exs_updates = zip(exs, new_exs)
    param_updates = [
        (p, p + dx)
        for dx, g, p in zip(delta_x, param_grads, params)
    ]

    updates = egs_updates + exs_updates + param_updates
    
    # updates WITHOUT embedding
    # exclude the embedding parameter
    egs_updates_no_ebd = zip(egs[1:], new_egs[1:])
    exs_updates_no_ebd = zip(exs[1:], new_exs[1:])
    param_updates_no_ebd = [
        (p, p + dx)
        for dx, g, p in zip(delta_x, param_grads, params)[1:]
    ]
    updates_no_emb = egs_updates_no_ebd + exs_updates_no_ebd + param_updates_no_ebd
    
    def make_train_func(cost, updates):
        return theano.function(inputs = [batch_index],
                               outputs = [cost], 
                               updates = updates,
                               givens = {
                                   x: train_set_x[batch_index * args.batch_size: (batch_index + 1) * args.batch_size],
                                   y: train_set_y[batch_index * args.batch_size: (batch_index + 1) * args.batch_size]
                               }
        )        

    train_model_no_ebd = make_train_func(cost_no_ebd, updates_no_emb)
    train_model = make_train_func(cost, updates)

    def make_error_func(x_val, y_val):
        return theano.function(inputs = [],
                               outputs = errors, 
                               givens = {
                                   x: x_val,
                                   y: y_val
                               }, 
                           )
        
    dev_error = make_error_func(dev_set_x, dev_set_y)

    test_error = make_error_func(test_set_x, test_set_y)
    

    #############################
    # Debugging purpose code    #
    #############################
    # : PARAMETER TUNING NOTE:
    # some demonstration of the gradient vanishing probelm
    
    train_data_at_index = {
        x: train_set_x[batch_index * args.batch_size: (batch_index + 1) * args.batch_size],
    }

    train_data_at_index_with_y = {
        x: train_set_x[batch_index * args.batch_size: (batch_index + 1) * args.batch_size],
        y: train_set_y[batch_index * args.batch_size: (batch_index + 1) * args.batch_size]
    }

    if print_config["nnl"]:
        get_nnl = theano.function(
            inputs = [batch_index],
            outputs = dropout_cost,
            givens = {
                x: train_set_x[batch_index * args.batch_size: (batch_index + 1) * args.batch_size],
                y: train_set_y[batch_index * args.batch_size: (batch_index + 1) * args.batch_size]
            }
        )
        
    if print_config["L2_sqr"]:
        get_L2_sqr = theano.function(
            inputs = [],
            outputs = L2_sqr
        )

        get_L2_sqr_no_ebd = theano.function(
            inputs = [],
            outputs = L2_sqr_no_ebd
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

    activations = [
        l.output
        for l in dropout_layers[1:-1]
    ]
    weight_grads = [
        T.grad(cost, l.W)
        for l in dropout_layers[1:-1]
    ]

    if print_config["activation_hist"]:
        # turn into 1D array
        get_activations = theano.function(
            inputs = [batch_index], 
            outputs = [
                val.flatten(1)
                for val in activations
            ], 
            givens = train_data_at_index
        )

    if print_config["weight_grad_hist"]:
        # turn into 1D array
        get_weight_grads = theano.function(
            inputs = [batch_index], 
            outputs = [
                val.flatten(1)
                for val in weight_grads
            ], 
            givens = train_data_at_index_with_y
        )
        
    if print_config["activation_tracking"]:
        # get the mean and variance of activations for each conv layer                
        
        get_activation_mean = theano.function(
            inputs = [batch_index], 
            outputs = [
                T.mean(val)
                for val in activations
            ], 
            givens = train_data_at_index
        )

        get_activation_std = theano.function(
            inputs = [batch_index], 
            outputs = [
                T.std(val)
                for val in activations
            ], 
            givens = train_data_at_index
        )


    if print_config["weight_grad_tracking"]:
        # get the mean and variance of activations for each conv layer
        get_weight_grad_mean = theano.function(
            inputs = [batch_index], 
            outputs = [
                T.mean(g)
                for g in weight_grads
            ], 
            givens = train_data_at_index_with_y
        )

        get_weight_grad_std = theano.function(
            inputs = [batch_index], 
            outputs = [
                T.std(g)
                for g in weight_grads
            ], 
            givens = train_data_at_index_with_y
        )        
    
    #the training loop
    patience = args.patience  # look as this many examples regardless
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
    
    activation_means = [[] for i in xrange(args.conv_layer_n)]
    activation_stds = [[] for i in xrange(args.conv_layer_n)]
    weight_grad_means = [[] for i in xrange(args.conv_layer_n)]
    weight_grad_stds = [[] for i in xrange(args.conv_layer_n)]
    activation_hist_data = [[] for i in xrange(args.conv_layer_n)]
    weight_grad_hist_data = [[] for i in xrange(args.conv_layer_n)]

    train_errors = []
    dev_errors = []
    try:
        print "validation_frequency = %d" %validation_frequency
        while (epoch < args.n_epochs):
            epoch += 1
            print "At epoch {0}".format(epoch)

            if epoch == (args.embedding_learning_delay_epochs + 1):
                print "########################"
                print "Start training embedding"
                print "########################"

            # shuffle the training data        
            train_set_x_data = train_set_x.get_value(borrow = True)
            train_set_y_data = train_set_y.get_value(borrow = True)        
            
            permutation = np.random.permutation(train_set_x.get_value(borrow=True).shape[0])

            train_set_x.set_value(train_set_x_data[permutation])
            train_set_y.set_value(train_set_y_data[permutation])
            for minibatch_index in xrange(n_train_batches):
                if epoch >= (args.embedding_learning_delay_epochs + 1):
                    train_cost = train_model(minibatch_index)
                else:
                    train_cost = train_model_no_ebd(minibatch_index)


                iter = (epoch - 1) * n_train_batches + minibatch_index
                
                if (iter + 1) % validation_frequency == 0:

                    # train_error_val = np.mean([train_error(i)
                    #                            for i in xrange(n_train_batches)])
                    dev_error_val = dev_error()
                    
                    # print "At epoch %d and minibatch %d. \nTrain error %.2f%%\nDev error %.2f%%\n" %(
                    #     epoch, 
                    #     minibatch_index,
                    #     train_error_val * 100, 
                    #     dev_error_val * 100
                    # )

                    print "At epoch %d and minibatch %d. \nDev error %.2f%%\n" %(
                        epoch, 
                        minibatch_index,
                        dev_error_val * 100
                    )
                    
                    # train_errors.append(train_error_val)
                    dev_errors.append(dev_error_val)
                    
                    if dev_error_val < best_validation_loss:
                        best_iter = iter
                        #improve patience if loss improvement is good enough
                        if dev_error_val < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = dev_error_val

                        test_error_val = test_error()

                        print(
                           (
                               '     epoch %i, minibatch %i/%i, test error of'
                                ' best dev error %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_error_val * 100.
                            )
                        )

                        print "Dumping model to %s" %(args.model_path)
                        dump_params(params, args.model_path)

                if (minibatch_index+1) % 50 == 0 or minibatch_index == n_train_batches - 1:
                    print "%d / %d minibatches completed" %(minibatch_index + 1, n_train_batches)                
                    if print_config["nnl"]:
                        print "`nnl` for the past 50 minibatches is %f" %(np.mean(np.array(nnls)))
                        nnls = []
                    if print_config["L2_sqr"]:
                        print "`L2_sqr`` for the past 50 minibatches is %f" %(np.mean(np.array(L2_sqrs)))
                        L2_sqrs = []                                                                            
                    
                ##################
                # Plotting stuff #
                ##################
                if print_config["nnl"]:
                    nnl = get_nnl(minibatch_index)
                    # print "nll for batch %d: %f" %(minibatch_index, nnl)
                    nnls.append(nnl)
                    
                if print_config["L2_sqr"]:
                    if epoch >= (args.embedding_learning_delay_epochs + 1):
                        L2_sqrs.append(get_L2_sqr())
                    else:
                        L2_sqrs.append(get_L2_sqr_no_ebd())
                    
                if print_config["activation_tracking"]:
                    layer_means = get_activation_mean(minibatch_index)
                    layer_stds = get_activation_std(minibatch_index)
                    for layer_ms, layer_ss, layer_m, layer_s in zip(activation_means, activation_stds, layer_means, layer_stds):
                        layer_ms.append(layer_m)
                        layer_ss.append(layer_s)

                if print_config["weight_grad_tracking"]:
                    layer_means = get_weight_grad_mean(minibatch_index)
                    layer_stds = get_weight_grad_std(minibatch_index)
                    
                    for layer_ms, layer_ss, layer_m, layer_s in zip(weight_grad_means, weight_grad_stds, layer_means, layer_stds):
                        layer_ms.append(layer_m)
                        layer_ss.append(layer_s)

                if print_config["activation_hist"]:
                    for layer_hist, layer_data in zip(activation_hist_data , get_activations(minibatch_index)):
                        layer_hist += layer_data.tolist()

                if print_config["weight_grad_hist"]:
                    for layer_hist, layer_data in zip(weight_grad_hist_data , get_weight_grads(minibatch_index)):
                        layer_hist += layer_data.tolist()
                                    
    except:
        import traceback
        traceback.print_exc(file = sys.stdout)
    finally:
        from plot_util import (plot_hist, 
                               plot_track, 
                               plot_error_vs_epoch, 
                               plt)

        if print_config["activation_tracking"]:
            plot_track(activation_means, 
                          activation_stds, 
                          "activation_tracking")

        if print_config["weight_grad_tracking"]:
            plot_track(weight_grad_means, 
                          weight_grad_stds,
                          "weight_grad_tracking")
            
        if print_config["activation_hist"]:        
            plot_hist(activation_hist_data, "activation_hist")

        if print_config["weight_grad_hist"]:
            plot_hist(weight_grad_hist_data, "weight_grad_hist")

        if print_config["error_vs_epoch"]:
            train_errors = [0] * len(dev_errors)
            ax = plot_error_vs_epoch(train_errors, dev_errors, 
                                     title = ('Best dev score: %f %% '
                                              ' at iter %i with test error %f %%') %(
                                                  best_validation_loss * 100., best_iter + 1, test_error_val * 100.
                                              )
            )
        if not args.task_signature:
            plt.show()
        else:
            plt.savefig("plots/" + args.task_signature + ".png")
    
    end_time = time.clock()
    
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_error_val * 100.))
    
    # save the result
    with open(args.output, "a") as f:
        f.write("%s\t%f\t%f\n" %(args.task_signature, best_validation_loss, test_error_val))
        
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
if __name__ == "__main__":
    print_config = {
        "adadelta_lr_mean": 0,
        "adagrad_lr_mean": 0,
        
        "embeddings": 0,
        "logreg_W": 0,
        "logreg_b": 0,
        
        "conv_layer1_W": 0,
        "conv_layer2_W": 0,
        
        "activation_tracking": 0, # the activation value, mean and variance
        "weight_grad_tracking": 0, # the weight gradient tracking
        "backprop_grad_tracking": 0, # the backpropagated gradient, mean and variance. In this case, grad propagated from layer 2 to layer 1
        "activation_hist": 0, # the activation value, mean and variance
        "weight_grad_hist": 0, # the weight gradient tracking
        "backprop_grad_hist": 0,
        "error_vs_epoch": 1,
        
        "l1_output": 0,
        "dropout_l1_output": 0,
        "l2_output": 0,        
        "dropout_l2_output": 0,
        "l3_output": 0,

        "p_y_given_x": 0,
        
        "grad_abs_mean": 0,
        "nnl": 1,
        "L2_sqr": 1,
        "param_weight_mean": 0,
    }
    
    import argparse, sys

    parser = argparse.ArgumentParser(description = "CNN with k-max pooling for sentence classification")
    
    parser.add_argument('--corpus_path', type=str,
                        required = True,
                       help = 'Path of preprocessed corpus'
    )

    parser.add_argument('--model_path', type=str, 
                        required = True,
                        help = 'Path of model parameters'
    )
    
    parser.add_argument("--fold", type=int, default = [1,1], nargs="+",
                        dest = "fold_flags", 
                        help = "Flags that turn on/off folding"
    )
    parser.add_argument("--ext_ebd", action = "store_true",
                        dest = "use_pretrained_embedding",
                        help = "Use external/pretrained word embedding or not. For unkown reasons, type checking does not work for this argument"
    )

    parser.add_argument("--l2", action = "store_true",
                        dest = "use_L2_reg", 
                        help = "Use L2 regularization or not"
    )
    
    parser.add_argument("--lr", type=float, default = 0.001, 
                        dest = "learning_rate", 
                        help = "Learning rate if constant learning rate is applied"
    )
    parser.add_argument("--norm_w", action = "store_true",
                        help = "Normalized initial weight as descripted in Glorot's paper"
    )
    parser.add_argument("--ebd_delay_epoch", type=int, default = 4, 
                        dest = "embedding_learning_delay_epochs", 
                        help = "Embedding learning delay epochs"
    )
    parser.add_argument("--au", type=str, default = "tanh",
                        dest = "conv_activation_unit", 
                        help = "Activation unit type for the convolution layer"
    )
    parser.add_argument("--eps", type=float, default =0.000001, 
                        dest = "epsilon", 
                        help = "Epsilon used by AdaDelta"
    )
    parser.add_argument("--rho", type=float, default = 0.95,
                        help = "Rho used by AdaDelta"
    )
    parser.add_argument("--ebd_dm", type=int, default = 48,
                        dest = "embed_dm", 
                        help = "Dimension for word embedding"
    )
    parser.add_argument("--batch_size", type=int, default = 10, 
                        dest = "batch_size", 
                        help = "Batch size in the stochastic gradient descent"
    )

    parser.add_argument("--dev_test_batch_size", type=int, default = 1000, 
                        help = "Batch size for dev/test data"
    )
    
    parser.add_argument("--n_epochs", type=int, default =20,
                        help = "Maximum number of epochs to perform during training"
    )
    parser.add_argument("--dr", type=float, default = [0.2, 0.5, 0.5], nargs="+",
                        dest = "dropout_rates", 
                        help = "Dropout rates at all layers except output layer"
    )
    parser.add_argument("--l2_regs", type = float, default = [0.00001, 0.0003, 0.0003, 0.0001], nargs="+",
                        dest = "L2_regs", 
                        help = "L2 regularization parameters at each layer. left/low->right/high"
    )
    parser.add_argument("--ks", type = int, default = [15, 6], nargs="+",
                        help = "The k values of the k-max pooling operation"
    )
    parser.add_argument("--conv_layer_n", type=int, default = 2,
                        help = "Number of convolution layers"
    )
    parser.add_argument("--nkerns", type=int, default = [6,12], nargs="+",
                        help = "Number of feature maps at each conv layer"
    )
    parser.add_argument("--filter_widths", type=int, default = [10,7], nargs="+",
                        help = "Filter width for each conv layer"
    )
    parser.add_argument("--task_signature", type=str,
                        help = "The prefix of the saved images."
    )

    parser.add_argument("--output", type=str,
                        required = True,
                        help = "The output file path to save the result"
    )

    parser.add_argument("--patience", type=int,
                        default = 5000,
                        help = "Patience parameter used for early stopping"
    )
    
    args = parser.parse_args(sys.argv[1:])

    print "Configs:\n-------------\n"
    for attr, value in vars(args).items():
        print "%s: %r" %(
            attr.ljust(25), 
            value
        )
    
    train_and_test(        
        args,
        print_config
    )
        
