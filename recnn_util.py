"""
Utility for RecNN
"""
import sys
import numpy as np
import operator
from collections import OrderedDict

import ptb
from tree_stat import token_freq
import codecs 

try:
    import cPickle as pickle
except:
    import pickle

UNK_TOKEN = "<UNK>"

class CannotMergeAnyMoreException(Exception):
    pass
    
def merge_leaves(tree):
    """
    Merge the tree leaves, return the new tree.
    
    >>> from ptb import parse
    >>> t = parse("(4 (4 (2 A) (4 (3 (3 warm) (2 ,)) (3 funny))) (3 (2 ,) (3 (4 (4 engaging) (2 film)) (2 .))))")
    >>> merge_leaves(t)
    (4, (4, (2, 'A'), (4, (3, ('warm', ',')), (3, 'funny'))), (3, (2, ','), (3, (4, ('engaging', 'film')), (2, '.'))))
    >>> merge_leaves((4, (4, (2, 'A'), (4, (3, ('warm', ',')), (3, 'funny'))), (3, (2, ','), (3, (4, ('engaging', 'film')), (2, '.')))))
    (4, (4, (2, 'A'), (4, (('warm', ','), 'funny'))), (3, (2, ','), (3, (('engaging', 'film'), '.'))))
    """
    def aux(t):
        if len(t) == 3 and len(t[1]) == 2 and len(t[2]) == 2:
            return (t[0], (t[1][1], t[2][1]))
        elif len(t) == 3 and len(t[1]) == 3 and len(t[2]) == 2:
            return (t[0], aux(t[1]), t[2])
        elif len(t) == 3 and len(t[1]) == 2 and len(t[2]) == 3:
            return (t[0], t[1], aux(t[2]))
        else:
            return (t[0], aux(t[1]), aux(t[2]))

    if len(tree) == 2: #the tree is a leaf
        raise CannotMergeAnyMoreException
    else:
        return aux(tree)

def collect_nodes(trees):
    """
    Collect node information(token, left child, right child, label) of trees by starting from lower part of trees and moving to the top

    Param:
    ------

    trees: list of tree
    
    Return:
    ------
    list of tuple, (token, left child token, right child token, label)
    
    >>> from ptb import parse
    >>> t1 = parse("(4 (4 (2 A) (4 (3 (3 warm) (2 ,)) (3 funny))) (3 (2 ,) (3 (4 (4 engaging) (2 film)) (2 .))))")
    >>> t2 = parse("(0 (0 (2 A) (0 (0 (0 boring) (2 ,)) (0 bad))) (1 (2 ,) (1 (1 (1 unsatisfactory) (2 film)) (2 .))))")
    >>> t3 = parse("(2 film)") # some repeatition
    >>> data = collect_nodes([t1, t2, t3])
    >>> len(data)
    24
    >>> data[-1]
    ((('A', (('boring', ','), 'bad')), (',', (('unsatisfactory', 'film'), '.'))), ('A', (('boring', ','), 'bad')), (',', (('unsatisfactory', 'film'), '.')), 0)
    >>> data[0]
    ('funny', None, None, 3)
    >>> nodes = collect_nodes([t1])
    >>> len(nodes)
    14
    >>> nodes
    [('funny', None, None, 3), (',', None, None, 2), ('.', None, None, 2), ('engaging', None, None, 4), ('film', None, None, 2), ('warm', None, None, 3), ('A', None, None, 2), (('warm', ','), 'warm', ',', 3), (('engaging', 'film'), 'engaging', 'film', 4), ((('warm', ','), 'funny'), ('warm', ','), 'funny', 4), ((('engaging', 'film'), '.'), ('engaging', 'film'), '.', 3), (('A', (('warm', ','), 'funny')), 'A', (('warm', ','), 'funny'), 4), ((',', (('engaging', 'film'), '.')), ',', (('engaging', 'film'), '.'), 3), ((('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.'))), ('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.')), 4)]
    """
    all_tokens = []  # place to store the final result
    collected_tokens = set()
    
    while len(trees) > 0:
        shallower_trees = []
        
        # collect the leaf nodes
        for t in trees:
            tokens_with_labels = set(ptb.get_leaves_with_labels(t))

            # not all tokens are harvested
            # only the new ones
            new_tokens_with_labels = []
            for token, label in tokens_with_labels:
                if token not in collected_tokens:
                    new_tokens_with_labels.append((token, label))
            

            if new_tokens_with_labels:
                tokens, labels = zip(*new_tokens_with_labels)
            else:
                continue # nothing to add
                        
            # add new tokens, their children and their labels
            all_tokens += [
                (tok, ) + # the token
                ((tok[0], tok[1]) # children node id if has children
                  if isinstance(tok, tuple) 
                  else (None, None)) +  #for single words
                (l, ) # the label
                for tok, l in zip(tokens, labels)
            ]
            collected_tokens |= set(tokens)
            
            try:
                shallower_trees.append(merge_leaves(t))
            except CannotMergeAnyMoreException:
                pass

        trees = shallower_trees# we consider the shallower trees now
    
    return all_tokens

def replace_tokens_by_condition(nodes, condition_func, to_token = UNK_TOKEN, to_label = 3):
    """
    Replace tokens to target token by certain condition

    >>> from collections import Counter
    >>> c = Counter({'A': 10, 'funny': 10, ',': 10, '.': 10, 'engaging': 1, 'film': 10})
    >>> nodes = [('funny', None, None, 3), (',', None, None, 2), ('.', None, None, 2), ('engaging', None, None, 4), ('film', None, None, 2), ('warm', None, None, 3), ('A', None, None, 2), (('warm', ','), 'warm', ',', 3), (('engaging', 'film'), 'engaging', 'film', 4), ((('warm', ','), 'funny'), ('warm', ','), 'funny', 4), ((('engaging', 'film'), '.'), ('engaging', 'film'), '.', 3), (('A', (('warm', ','), 'funny')), 'A', (('warm', ','), 'funny'), 4), ((',', (('engaging', 'film'), '.')), ',', (('engaging', 'film'), '.'), 3), ((('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.'))), ('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.')), 4)]
    >>> condition_func = lambda w: c[w] < 5 # `engaging` and `warm` should be filtered out
    >>> replace_tokens_by_condition(nodes, condition_func, to_token = "<UNK>")
    [('<UNK>', None, None, 3), ('funny', None, None, 3), (',', None, None, 2), ('.', None, None, 2), ('film', None, None, 2), ('A', None, None, 2), (('warm', ','), '<UNK>', ',', 3), (('engaging', 'film'), '<UNK>', 'film', 4), ((('warm', ','), 'funny'), ('warm', ','), 'funny', 4), ((('engaging', 'film'), '.'), ('engaging', 'film'), '.', 3), (('A', (('warm', ','), 'funny')), 'A', (('warm', ','), 'funny'), 4), ((',', (('engaging', 'film'), '.')), ',', (('engaging', 'film'), '.'), 3), ((('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.'))), ('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.')), 4)]
    """
    new_nodes = [(to_token, None, None, to_label)] # to_token should be added also
    for node in nodes:
        parent, lchild, rchild, label = node

        # ignore leaf node satisfying condition
        if lchild is None and rchild is None:
            assert isinstance(parent, basestring)
            if condition_func(parent):
                continue

        # replace internal node children(if is string and satisfy condition)
        if isinstance(lchild, basestring):
            if condition_func(lchild):
                lchild = to_token

        if isinstance(rchild, basestring):
            if condition_func(rchild):
                rchild = to_token

        new_nodes.append((parent, lchild, rchild, label))

    return new_nodes
    
def build_node_id_mapping(nodes):
    """
    Build the mapping from tree node to array index
    
    >>> nodes = [('funny', None, None, 3), (',', None, None, 2), ('.', None, None, 2), ('engaging', None, None, 4), ('film', None, None, 2), ('warm', None, None, 3), ('A', None, None, 2), (('warm', ','), 'warm', ',', 3), (('engaging', 'film'), 'engaging', 'film', 4), ((('warm', ','), 'funny'), ('warm', ','), 'funny', 4), ((('engaging', 'film'), '.'), ('engaging', 'film'), '.', 3), (('A', (('warm', ','), 'funny')), 'A', (('warm', ','), 'funny'), 4), ((',', (('engaging', 'film'), '.')), ',', (('engaging', 'film'), '.'), 3), ((('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.'))), ('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.')), 4)]
    >>> token2id = build_node_id_mapping(nodes)
    >>> token2id # doctest: +ELLIPSIS
    OrderedDict([('funny', 0), (',', 1), ('.', 2)...((('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.'))), 13)])

    """
    tokens = map(operator.itemgetter(0), nodes)
    mapping = OrderedDict()
    for i, token in enumerate(tokens):
        mapping[token] = i
    return mapping

def build_input(nodes, token2id):
    """
    Param:
    ----------
    the tree nodes and token to index mapping
    
    Return
    ----------
    1. tree matrix: numpy.array, Nx3, (token id, left child id, right child id)
    2. labels: numpy.array, 1xN or Nx1

    >>> token2id = OrderedDict([('<UNK>', 14), ('funny', 0), (',', 1), ('.', 2), ('engaging', 3), ('film', 4), ('warm', 5), ('A', 6), (('warm', ','), 7), (('engaging', 'film'), 8), ((('warm', ','), 'funny'), 9), ((('engaging', 'film'), '.'), 10), (('A', (('warm', ','), 'funny')), 11), ((',', (('engaging', 'film'), '.')), 12), ((('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.'))), 13)])
    >>> nodes = [('balhword', None, None, 3), ('funny', None, None, 3), (',', None, None, 2), ('.', None, None, 2), ('engaging', None, None, 4), ('film', None, None, 2), ('warm', None, None, 3), ('A', None, None, 2), (('warm', ','), 'warm', ',', 3), (('engaging', 'film'), 'engaging', 'film', 4), ((('warm', ','), 'funny'), ('warm', ','), 'funny', 4), ((('engaging', 'film'), '.'), ('engaging', 'film'), '.', 3), (('A', (('warm', ','), 'funny')), 'A', (('warm', ','), 'funny'), 4), ((',', (('engaging', 'film'), '.')), ',', (('engaging', 'film'), '.'), 3), ((('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.'))), ('A', (('warm', ','), 'funny')), (',', (('engaging', 'film'), '.')), 4)]
    >>> x, y = build_input(nodes, token2id)
    >>> x # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[14, -1, -1],
           [ 0, -1, -1], 
           [ 1, -1, -1], 
    ...
           [13, 11, 12]], dtype=int32)
    >>> y # doctest: +ELLIPSIS
    array([3, 3, 2, 2,..., 4], dtype=int32)
    """
    x_array = []
    for t1, t2, t3, _ in nodes:
        if t1 in token2id:
            x_array.append([token2id[t1], token2id.get(t2, -1), token2id.get(t3, -1)])
        else: # cope with unknown words
            x_array.append([token2id[UNK_TOKEN], token2id.get(t2, -1), token2id.get(t3, -1)])

    x = np.asarray(x_array, dtype=np.int32)
    y = np.asarray([y for _,_,_,y in nodes], dtype=np.int32)

    return x, y

def dump_data(train_path, dev_path, test_path, output_path = "data/stanford_sentiment_treebank.pkl"):
    sys.stderr.write("loading trees..\n")
    train_trees = ptb.load_trees(codecs.open(train_path, "r", "utf8"))
    dev_trees = ptb.load_trees(codecs.open(dev_path, "r", "utf8"))
    test_trees = ptb.load_trees(codecs.open(test_path, "r", "utf8"))
    
    nodes = collect_nodes(train_trees)
    freq_table = token_freq(train_trees)
    rare_condition = lambda w: freq_table[w] < 5
    
    sys.stderr.write("preprocessing trees..\n")
    nodes = replace_tokens_by_condition(nodes, rare_condition)
    
    sys.stderr.write("get vocabulary size\n")
    word_number = len(filter(lambda node: node[1] is None, nodes))
    sys.stderr.write("word_number = %d\n" %(word_number))
    
    token2id = build_node_id_mapping(nodes)        
    
    assert "<UNK>" in token2id, "<UNK> should be in `token2id`"


    data = (train_trees, dev_trees, test_trees, token2id)
    
    pickle.dump(data, open(output_path, "w"))

    return data

def load_data(path = "data/stanford_sentiment_treebank.pkl"):
    """
    >>> data1 = dump_data("data/unittest_data/train.txt", \
    "data/unittest_data/dev.txt", \
    "data/unittest_data/test.txt",\
    "data/unittest_data/dump.pkl")
    >>> data2 = load_data("data/unittest_data/dump.pkl")
    >>> data1 == data2
    True
    """
    return pickle.load(open(path, "r"))

if __name__ == "__main__":
    dump_data("data/stanfordSentimentTreebank/trees/train.txt", 
              "data/stanfordSentimentTreebank/trees/dev.txt",
              "data/stanfordSentimentTreebank/trees/test.txt")
