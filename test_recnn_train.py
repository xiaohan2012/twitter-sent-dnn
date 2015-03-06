import numpy as np
import theano
import theano.tensor as T

from recnn_train import RNTN

from test_util import assert_matrix_neq

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
        [3, 1, 2], 
        [4, 0, 3],
    ],
    dtype = np.int32
)
phrase_number = tree_matrix.shape[0]

labels = np.asarray([2, 4, 2, 4, 4], dtype=np.int32)


x = T.imatrix('x')
y = T.ivector('y')

classifier = RNTN(
    x, y,
    vocab_size = 5, 
    embed_dim = 3, 
    label_n = 5,
)

x_input = np.asarray([[1,-1,-1],
                      [2,-1,-1],
                      [3, 1, 2]],
                     dtype=np.int32)
y_input = labels[1:4]

original_embedding = classifier.embedding.get_value()

classifier.update_embedding(x_input)

new_embedding = classifier.embedding.get_value()

assert_matrix_neq(original_embedding, 
                  new_embedding,
                  "update_embeding")

original_params = [p.get_value() for p in classifier.params]

classifier.train(x_input, y_input)
updated_params = [p for p in classifier.params]

for op, up in zip(original_params, updated_params):
    assert_matrix_neq(op, up.get_value(), up.name)
