# TO-DO: it's good to calculate the percentage of kept words during normalization
import os, urllib, gzip, cPickle, theano

import re
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

import pdb
from codecs import open
from collections import (OrderedDict, Counter)
from scipy import io

import theano.tensor as T
from ptb import (parse, flattened_subtrees, flatten_tree)

def load_data(dataset='mnist.pkl.gz'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
           'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an np.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #np.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                              dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                              dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def scale_to_unit_interval(ndar, eps=1e-8):
  """ Scales all values in the ndarray ndar to be between 0 and 1 """
  ndar = ndar.copy()
  ndar -= ndar.min()
  ndar *= 1.0 / (ndar.max() + eps)
  return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
  """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape = [0,0]
  # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 4
      # Create an output numpy ndarray to store the image
      if output_pixel_vals:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
      else:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

      #colors default to 0, alpha defaults to 1 (opaque)
      if output_pixel_vals:
          channel_defaults = [0, 0, 0, 255]
      else:
          channel_defaults = [0., 0., 0., 1.]

      for i in xrange(4):
          if X[i] is None:
              # if channel is None, fill it with zeros of the correct
              # dtype
              out_array[:, :, i] = np.zeros(out_shape,
                     dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
          else:
              # use a recurrent call to compute the channel and store it
              # in the output
              out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
      return out_array

  else:
      # if we are dealing with only one channel
      H, W = img_shape
      Hs, Ws = tile_spacing

      # generate a matrix to store the output
      out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


      for tile_row in xrange(tile_shape[0]):
          for tile_col in xrange(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      # if we should scale values to be between 0 and 1
                      # do this by calling the `scale_to_unit_interval`
                      # function
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                  else:
                      this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  # add the slice to the corresponding position in the
                  # output array
                  out_array[
                      tile_row * (H+Hs): tile_row * (H + Hs) + H,
                      tile_col * (W+Ws): tile_col * (W + Ws) + W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array

def process_stanford_sentiment_corpus(train_path, dev_path, test_path, 
                                      pkl_path, 
                                      unk_threshold, 
                                      unk_token= '<UNK>', 
                                      pad_token= '<PADDING>'):
    """
    Input three paths for the PTB tree file of train/validate/test data

    unk_threshold: the frequency threshold below which the word is marked as unk_token

    preproces the data and save the pickle
    
    Return the pickle path
    """
    # parse all the trees
    # and represent sentence as a list of words

    print "parsing trees.."
    with open(train_path, "r", "utf8") as train_f, \
         open(dev_path, "r", "utf8") as dev_f, \
         open(test_path, "r", "utf8") as test_f:
        #flattened subtrees for training data only
        train_sents, train_labels = zip(*[sub_sent
                                          for l in train_f
                                          for sub_sent in flattened_subtrees(parse(l))])
        dev_sents, dev_labels = zip(*[flatten_tree(parse(l))
                                      for l in dev_f])
        test_sents, test_labels = zip(*[flatten_tree(parse(l))
                                        for l in test_f])

    print "Train sent size: %d\nDev sent size: %d\nTest sent size: %d" %(
        len(train_sents), len(dev_sents), len(test_sents)
    )
    # gathering sentence length information
    sent_lens = [len(sent) 
                 for sent in train_sents]
    train_sent_max_len = max(sent_lens)
    print "train_sent_max_len: %d" %(train_sent_max_len)
    print "sent_mean_len: %f" %(np.mean(sent_lens))
    print "sent_median_len: %f" %(np.median(sent_lens))
    
    train_sent_max_len = max((len(sent) 
                              for sent in train_sents))
    dev_sent_max_len = max((len(sent) 
                              for sent in dev_sents))
    print "dev_sent_max_len: %d" %(dev_sent_max_len)

    test_sent_max_len = max((len(sent) 
                              for sent in test_sents))
    print "test_sent_max_len: %d" %(test_sent_max_len)
    
    # preprocess number to DIGIT
    # somewhat memory inefficient
    # and also to lowercase
    print "convert digits..."
    regexp =re.compile(r'\d')
    
    train_sents = [regexp.sub('DIGIT', ' '.join(sent).lower()).split()
                   for sent in train_sents]
    dev_sents = [regexp.sub('DIGIT', ' '.join(sent).lower()).split()
                 for sent in dev_sents]
    test_sents = [regexp.sub('DIGIT', ' '.join(sent).lower()).split()
                  for sent in test_sents]
    
    print "Collecting word frequency"
    # gather words in the train set
    # count their frequency
    word_freq = Counter((w 
                         for sent in train_sents
                         for w in sent))
    
    print "Building word and index mapping"
    # build the word-to-index dictionary and vice versa.
    # mark the infrequency word as unk_token
    frequent_words = [w
                      for w in word_freq 
                      if word_freq[w] > unk_threshold]

    print "Vocab size: %d" %len(frequent_words)
    
    # add the two additional words
    frequent_words.append(unk_token)
    frequent_words.append(pad_token)
    
    word2index = OrderedDict([(w, i)
                              for i, w in enumerate(frequent_words)])
    index2word = OrderedDict([(w, i)
                              for i, w in word2index.items()])
    
    
    padding_index = word2index[pad_token]
    print "padding_index = %d" %(padding_index)
    
    print "Converting sentence to numpy array.."
    
    sent2array_padded = lambda sent, max_len: (
        [word2index.get(word, word2index[unk_token])  
         for word in sent] +
        [padding_index] * (max_len - len(sent)) # add the paddings
    ) 
    
    # sent2array_unpadded = lambda sent: [word2index.get(word, word2index[unk_token])
    #                                     for word in sent]
    
    # construct the sentence data,
    # each sentence is represented by the word indices
    def create_dataset(sents, labels, sent_max_len):
        x = np.array([sent2array_padded(sent, sent_max_len)
                      for sent in sents],
                     dtype = "int32")
        
        y =np.array(labels, dtype="int32")

        return x, y

    
    train_x, train_y = create_dataset(train_sents, train_labels, train_sent_max_len)
    dev_x, dev_y = create_dataset(dev_sents, dev_labels, dev_sent_max_len)
    test_x, test_y = create_dataset(test_sents, test_labels, test_sent_max_len)
    
    
    # load the pretrained embedding
    pkl_data = (
        (train_x, train_y),
        (dev_x, dev_y),
        (test_x, test_y),
        word2index,
        index2word, 
        np.load("data/stanfordSentimentTreebank/trees/pretrained.npy")
    )
    
    print "dumping pickle to %s" %(pkl_path)
    pickle.dump(pkl_data, open(pkl_path, 'w'))

    debug = True
    if debug:
        print type(train_x)
        print train_x[0]
        print train_y[0]
        print dev_x[0]
        print dev_y[0]
        print test_x[0]
        print test_y[0]
        
    return pkl_data

def share_dataset(x, y):
    shared_x = theano.shared(
        np.asarray(x, dtype = np.int32),
        borrow = True)
    
    shared_y = theano.shared(
        np.asarray(y, dtype = np.int32),
        borrow = True
    )

    return shared_x, shared_y
    
def convert_nal_data(src_path, pkl_path):
    """convert the data in Nal's code to some format that is usable by me"""
    data = io.loadmat(src_path)


    word2index = {
        unicode(row[0][0] if row[0].size>0 else ''): i
        for i, row in enumerate(data["index"])
    }
    
    # add padding
    word2index[u"<PADDING>"] = len(word2index)

    index2word = {
        i: word
        for word, i  in word2index.items()
    }
    train = (data["train"] - 1, data["train_lbl"][:,0] - 1)
    dev = (data["valid"] - 1, data["valid_lbl"][:,0] - 1)
    test = (data["test"] - 1, data["test_lbl"][:,0] - 1)

    print "dumping result"
    pickle.dump(
        (
            train, #matlab' index starts at 1
            dev, 
            test,
            word2index,
            index2word,
            np.transpose(data['vocab_emb'])
        ), 
        open(pkl_path, "w")
    )


def load_data(pkl_path):
    """
    load the sentiment dataset, either Stanford or Twitter

    Return:
    - train
    - dev
    - test
    - word2index
    - index2word
    - pretrained embedding
    """
    data = pickle.load(open(pkl_path, 'r'))
    return (share_dataset(*data[0]), #train
            share_dataset(*data[1]), #dev
            share_dataset(*data[2]), #test
            data[3], 
            data[4], 
            theano.shared(
                value = np.asarray(
                    data[5],
                    dtype=theano.config.floatX),
                name = "embeddings", 
                borrow = True,
            )
    )

def dump_params(params, path):
    """
    params: list of thenao.tensor.TensorType
    """
    data = [(param.name, param.get_value())
            for param in params]
    pickle.dump(data, open(path, "w"))


def modify_tuple(obj, positions, new_values):
    """
    Modify the tuple object at certain positions by certain value
    
    >>> modify_tuple((1,2,3), [0,1,2], [3,2,1])
    (3, 2, 1)
    """
    assert isinstance(obj, tuple)
    assert len(positions) == len(new_values)

    alt = list(obj)
    for pos, val in zip(positions, new_values):
        alt[pos] = val
        
    return tuple(alt)

if __name__ == "__main__":
    # process_stanford_sentiment_corpus('data/stanfordSentimentTreebank/trees/train.txt', 
    #                                   'data/stanfordSentimentTreebank/trees/dev.txt', 
    #                                   'data/stanfordSentimentTreebank/trees/test.txt', 
    #                                   'data/stanfordSentimentTreebank/trees/processed.pkl', 
    #                                   unk_threshold = 3)
    convert_nal_data("data/twitter.mat", "data/twitter.pkl")    
