import theano
import numpy as np
from dcnn import WordEmbeddingLayer
from dcnn_train import WordEmbeddingLayer as TheanoWordEmbeddingLayer
from test_util import assert_matrix_eq
########### NUMPY ###########

vocab_size, embed_dm = 10, 5
embeddings = np.random.rand(vocab_size, embed_dm)
sents = np.asarray(np.random.randint(10, size = (3, 6)), 
                   dtype = np.int32)


np_l = WordEmbeddingLayer(embeddings)

actual = np_l.output(sents)

########### THEANO ###########

x_symbol = theano.tensor.imatrix('x') # the word indices matrix

th_l = TheanoWordEmbeddingLayer(rng = np.random.RandomState(1234), 
                                input = x_symbol, 
                                vocab_size = vocab_size,
                                embed_dm = embed_dm, 
                                embeddings = theano.shared(value = embeddings, 
                                                           name = "embeddings"
                                                       )
                            )

f = theano.function(inputs = [x_symbol], 
                    outputs = th_l.output)

expected = f(sents)

assert_matrix_eq(actual, expected, "Embedding")

