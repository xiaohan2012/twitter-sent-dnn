import numpy as np
import theano
import theano.tensor as T

from recnn import RNTN

# tree
# (4 (2 I) (4 (4 like) (2 you)))
# word/phrase to index mapping
# I: 0, nil, nil
# like: 1, nil, nil
# you: 2, nil, nil
# like you: 3, 1, 2
# I like you: 4, 0, 3

# value definition
tree_matrix = np.asarray(
    [
        [1, 2], 
        [0, 3],
    ],
    dtype = np.int32
)

labels = np.asarray([2, 4, 2, 4, 4])

word_embedding_val = np.asarray(
    [
        [0.1,0.1,0.1], # I
        [1, 0, 0], # like
        [0.1,0.1,0.1] # you
    ],
    dtype = theano.config.floatX
)

classifier = RNTN(
    tree_matrix = tree_matrix, 
    labels = labels, 
    vocab_size = 3, 
    embed_dim = 3, 
    label_n = 5,
)


get_final_embedding = theano.function(
    inputs = [], 
    outputs = [classifier.final_embedding], 
)

print get_final_embedding()

get_cost = theano.function(
    inputs = [], 
    outputs = [classifier.cost], 
)

print get_cost()
