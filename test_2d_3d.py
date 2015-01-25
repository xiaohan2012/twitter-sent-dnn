import numpy as np
import theano
import theano.tensor as T

embedding = T.imatrix('embedding')
sents = T.imatrix('sents')

sents_val = np.array([[0,3], [1,2]], dtype=np.int32)

embedding_val = np.array([[1,2], [3,4], [5,6], [7,8]], dtype=np.int32)

results, updates = theano.map(lambda sent: embedding[sent], 
                              sents)
f = theano.function(inputs = [sents, embedding], 
                    outputs = T.stacklists(results)

)


print """Expected:
[
     [[1,2], [7,8]], 
     [[3,4], [5,6]]
]
"""

print "Actual:"
print type(f(sents_val, embedding_val))
