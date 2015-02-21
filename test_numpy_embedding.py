import numpy as np
from dcnn import WordEmbeddingLayer

# 5 x 3 embedding for test
embedding = np.arange(15).reshape((5,3))
embedding_layer = WordEmbeddingLayer(embedding)
sents = np.array([
    [0, 1, 2], 
    [1, 2, 3]
])

expected = [
    [
        np.transpose(np.array(
            [
                [0,1,2], 
                [3,4,5], 
                [6,7,8]
            ]))
    ],
    [
        np.transpose(np.array(
            [
                [3,4,5], 
                [6,7,8], 
                [9, 10, 11]
            ]))
    ]
]

actual = embedding_layer.output(sents)

assert np.all(expected == actual)

