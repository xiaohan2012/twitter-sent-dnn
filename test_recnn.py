import theano
import theano.tensor as T
import numpy as np

from recnn_train import RNTN as TheanoRNTN
from recnn import RNTN as NumpyRNTN, RNTNLayer
from numpy_impl import LogisticRegression

from test_util import assert_matrix_eq

vocab_size = 6
embed_dim = 3
label_n = 5
word2id = {
    'I': 0,
    'love': 1,
    'you':2,
    '<UNK>': 5,
}

x = T.imatrix('x')
y = T.ivector('y')

th_model = TheanoRNTN(x, y, vocab_size, embed_dim, label_n)


np_model = NumpyRNTN.load_from_theano_model(th_model, word2id)# (embedding = th_model.embedding.get_value(), 
                     # rntn_layer = RNTNLayer(th_model.rntn_layer.V.get_value(), th_model.rntn_layer.W.get_value()), 
                     # logreg_layer = LogisticRegression(th_model.logreg_layer.W.get_value(), th_model.logreg_layer.b.get_value()), 
                     # word2id = word2id)

x_input = np.asarray([[4, 2, 5], 
                      [3, 1, 4]],
                     dtype=np.int32)

tree_input = (5, "love", (3, (3, "you"), (3, "bro")))
actual = np_model.get_node_vector(tree_input)

th_model.update_embedding(x_input)

expected = th_model.embedding.get_value()[3]

assert_matrix_eq(actual, expected, "node vector")

get_label = theano.function(inputs = [x], 
                            outputs = th_model.logreg_layer.pred_y)

score = np_model.predict_top_node(tree_input)

assert isinstance(score, np.int64)

assert_matrix_eq(score, get_label(x_input[1:2,:]), 'logreg.predict')
