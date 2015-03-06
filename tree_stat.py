from ptb import get_leaves_with_labels
from collections import Counter

def token_freq(trees):
    """
    Collect token frequency statistics from trees

    >>> trees = [(1, (1, (2, 'a'), (3, 'b')), (1, (1, 'c'), (2, 'd'))), \
    (1, (1, (2, 'b'), (3, 'b')), (1, (1, 'c'), (2, 'a')))]
    >>> token_freq(trees)
    Counter({'b': 3, 'a': 2, 'c': 2, 'd': 1})
    """
    counter = Counter()
    for tree in trees:
        leaves = get_leaves_with_labels(tree)
        counter += Counter([token for token, label in  leaves])

    return counter
