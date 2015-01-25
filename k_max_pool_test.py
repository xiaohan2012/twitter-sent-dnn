import sys, pdb
import numpy as np
import theano
import theano.tensor as T

feat_n = 2

k = T.iscalar('k')
x = T.tensor4('x')

ind = T.argsort(x, axis = 3)

sorted_ind = T.sort(ind[:,:,:, -k:], axis = 3)

dim0, dim1, dim2, dim3 = sorted_ind.shape

indices_dim0 = T.arange(dim0).repeat(dim1 * dim2 * dim3)
indices_dim1 = T.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis=1).T.flatten()
indices_dim2 = T.arange(dim2).repeat(dim3).reshape((dim2*dim3, 1)).repeat(dim0 * dim1, axis = 1).T.flatten()

k_max_pool = theano.function(
    inputs = [x,k], 
    outputs = x[indices_dim0, indices_dim1, indices_dim2, sorted_ind.flatten()].reshape(sorted_ind.shape)
)

in_x = np.random.RandomState(1).rand(2, feat_n, 4,4)
in_k = 2
print "in_x:"
print in_x

print "theano output....\n"
print k_max_pool(in_x, in_k)

print "numpy output....\n"

# k-max pooling is different from merely sorting
# it selects the k largest items at certain axis but the original order should be maintained
# this makes the problem trickier
# so there are two sort involved:
# 1. sort by the values at the last axis
# 2. sort by their original order

# the basic idea is, **flatten and reshape**
# get the k largest items out and form a 1d array
# reshape the 1d arracy into the 4d tensor

ind = np.argsort(in_x, axis = 3)

sorted_ind = np.sort(ind[:, :, :, -in_k:], axis = 3)


# for 2 data instances, 2 feature maps
# width = 4 and height = 4
# k = 2
# index at dim 0 should be: 0 x 16 and 1 x 16
# index at dim 1 should be: 0x8 1x8 0x8 1x8
# index at dim 2 should be: (00 11 22 33)x4
dim0, dim1, dim2, dim3 = sorted_ind.shape

indices_dim0 = np.arange(dim0).repeat(dim1 * dim2 * dim3)
indices_dim1 = np.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis = 1).transpose().flatten()
indices_dim2 = np.arange(dim2).repeat(dim3).reshape((dim2*dim3, 1)).repeat(dim0 * dim1, axis = 1).transpose().flatten()

print indices_dim0, indices_dim1, indices_dim2


print in_x[indices_dim0, 
           indices_dim1, 
           indices_dim2,
           sorted_ind.flatten()].reshape(sorted_ind.shape)


