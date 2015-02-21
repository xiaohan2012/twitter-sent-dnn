import theano
import numpy as np

from dcnn import DCNN

from logreg import LogisticRegression
from dcnn_train import (WordEmbeddingLayer, ConvFoldingPoolLayer)



#########################
# THEANO PART           #
#########################


x_symbol = theano.tensor.imatrix('x') # the word indices matrix
y_symbol = theano.tensor.ivector('y') # the sentiment labels

rng = np.random.RandomState(1234)

vocab_size = 10
embed_dm = 8

embeddings = np.asarray(np.random.rand(vocab_size, embed_dm), 
                        dtype = theano.config.floatX)

layer1 = WordEmbeddingLayer(rng, 
                            input = x_symbol, 
                            vocab_size = vocab_size,
                            embed_dm = embed_dm, 
                            embeddings = theano.shared(value = embeddings, 
                                                       name = "embeddings"
                                                   )
                        )

filter_shape = (3, 1, 1, 2)
W = np.asarray(np.random.rand(3, 1, 1, 2), 
               dtype = theano.config.floatX
)

b = np.asarray(np.random.rand(3), 
               dtype = theano.config.floatX
)
k = 3

layer2 = ConvFoldingPoolLayer(rng = rng, 
                              input = layer1.output,
                              filter_shape = filter_shape,
                              k = k,
                              fold = 1,
                              W = theano.shared(value = W, name = "W"),
                              b = theano.shared(value = b, name = "b")
)

n_in = filter_shape[0] * k * embed_dm / 2
n_out = 5
W_logreg = np.asarray(np.random.rand(n_in, n_out), 
                      dtype = theano.config.floatX)
b_logreg = np.asarray(np.random.rand(n_out),
                      dtype = theano.config.floatX)

layer3 = LogisticRegression(rng = rng, 
                            input = layer2.output.flatten(2), 
                            n_in = n_in, 
                            n_out = n_out,
                            W = theano.shared(value = W_logreg, name = "W_logreg"),
                            b = theano.shared(value = b_logreg, name = "b_logreg")
)

f1 = theano.function(inputs = [x_symbol, y_symbol], 
                     outputs = layer3.nnl(y_symbol)
)

f2 = theano.function(inputs = [x_symbol, y_symbol], 
                     outputs = layer3.errors(y_symbol)
)

f3 = theano.function(inputs = [x_symbol], 
                     outputs = layer3.p_y_given_x
)

f_el = theano.function(inputs = [x_symbol], 
                       outputs = layer1.output
)

f_cl = theano.function(inputs = [x_symbol], 
                       outputs = layer2.output
)

#########################
# NUMPY PART            #
#########################

class Params(object):
    pass

p = Params()
p.embeddings = embeddings
p.conv_layer_n = 1
p.ks = [3]
p.fold = [1]
p.W = [W]
p.b = [b]
p.W_logreg = W_logreg
p.b_logreg = b_logreg

dcnn = DCNN(p)

##################### Testing ####################

from test_util import (assert_matrix_eq, assert_about_eq)

x = np.asarray(np.random.randint(vocab_size, size = (3, 6)),
               dtype=np.int32
)

y = np.asarray(np.random.randint(5, size = 3), 
               dtype=np.int32
)
    
########### Embedding layer ##############
actual = f_el(x)
expected = dcnn.e_layer.output(x)
assert_matrix_eq(actual, expected, "Embedding")


########## Conv layer ###################
actual = dcnn._c_layer_output(x)
expected = f_cl(x)

assert_matrix_eq(actual, expected, "Conv")

########## Output layer ###################
actual = dcnn._p_y_given_x(x)
expected = f3(x)
assert_matrix_eq(actual, expected, "p_y_given_x")


########## errors ###########
actual = dcnn._errors(x, y)
expected = f2(x, y)
assert_about_eq(actual, expected, "errors")

########## nnl ###########
actual = dcnn._nnl(x, y)
expected = f1(x, y)
assert_about_eq(actual, expected, "nnl")
