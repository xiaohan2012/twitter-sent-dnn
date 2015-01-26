import sys
import math
import numpy as np
import theano
import theano.tensor as T

from cnn4nlp import (WordEmbeddingLayer,
                     ConvFoldingPoolLayer)

from logreg import LogisticRegression

rng = np.random.RandomState(1234)

##### Test Part One ###############
# WordEmbeddingLayer
#################################

EMB_DIM = 6
x = T.imatrix('x') # the vector of word indices

l1 = WordEmbeddingLayer(rng, 
                        x,
                        10, EMB_DIM)

print "######## All Embeddings ########"
print l1.embeddings.get_value()
print l1.embeddings.get_value().shape



get_embedding = theano.function(
    inputs = [x], 
    outputs = l1.output,
    # mode = "DebugMode"
)

print "######## Selected Embeddings ########"
selected_embedding =  get_embedding(
    np.array([
        [1,3,5], 
        [2,0,7]
    ], 
             dtype = np.int32)
)

print selected_embedding
print selected_embedding.shape
assert selected_embedding.shape == (2, 1, EMB_DIM, 3)


##### Test Part Two ###############
# ConvFoldingPoolLayer
#################################

print "############# ConvFoldingPoolLayer ##############"
k = 2
feat_map_n = 2
l2 = ConvFoldingPoolLayer(rng, 
                          input = l1.output, 
                          filter_shape = (feat_map_n, 1, 1, 2),  # two feature map, height: 1, width: 2, 
                          k = k
)

l2_output = theano.function(
    inputs = [x],
    outputs = l2.output,
)

# TODO:
# check the dimension
# input: 1 x 1 x 6 x 4
out = l2_output(
    np.array([[1, 3, 4, 5]], dtype = np.int32)
)

print out
print out.shape

expected = (1, feat_map_n, EMB_DIM / 2, k)
assert out.shape == expected, "%r != %r" %(out.shape, expected)

##### Test Part Three ###############
# LogisticRegressionLayer
#################################

print "############# LogisticRegressionLayer ##############"

l3 = LogisticRegression(
    rng, 
    input = l2.output.flatten(2), 
    n_in = feat_map_n * k * EMB_DIM / 2, # we fold once, so divide by 2
    n_out = 5 # five sentiment level
)

print "n_in = %d" %(2 * 2 * math.ceil(EMB_DIM / 2.))

y = T.ivector('y') # the sentence sentiment label

p_y_given_x = theano.function(
    inputs = [x],
    outputs = l3.p_y_given_x,
    mode = "DebugMode"
)

print "p_y_given_x = "
print p_y_given_x(
    np.array([[1, 3, 4, 5], [0, 1, 4 ,7]], dtype = np.int32)
)

cost = theano.function(
    inputs = [x, y],
    outputs = l3.nnl(y),
    mode = "DebugMode"
)

print "cost:\n", cost(
    np.array([[1, 3, 4, 5], [0, 1, 4 ,7]], dtype = np.int32), 
    np.array([1, 2], dtype = np.int32)
)

