"""
Utility for RecNN
"""
def get_leaves_with_labels(tree):
    """return leaves in the tree, as well as their labels
    >>> from ptb import parse
    >>> t = parse("(4 (4 (2 A) (4 (3 (3 warm) (2 ,)) (3 funny))) (3 (2 ,) (3 (4 (4 engaging) (2 film)) (2 .))))")
    >>> get_leaves_with_labels(t)
    [('A', 2), ('warm', 3), (',', 2), ('funny', 3), (',', 2), ('engaging', 4), ('film', 2), ('.', 2)]
    """
    def aux(t):
        if len(t) == 2: # leaf
            return [(t[1], t[0])]
        else:
            return aux(t[1]) + aux(t[2])

    return aux(tree)

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

def build_tree_matrix(trees):
    """
    Given binary parse trees

    Return:
    - a list of (node_id, (left_child_node_id, right_child_node_id), label)
    - phrase2id mapping 
    - id2phrase mapping
    
    >>> from ptb import parse
    >>> t1 = parse("(4 (4 (2 A) (4 (3 (3 warm) (2 ,)) (3 funny))) (3 (2 ,) (3 (4 (4 engaging) (2 film)) (2 .))))")
    >>> t2 = parse("(0 (0 (2 A) (0 (0 (0 boring) (2 ,)) (0 bad))) (1 (2 ,) (1 (1 (1 unsatisfactory) (2 film)) (2 .))))")
    >>> data, token2id, id2token = build_tree_matrix([t1, t2])
    >>> data[-1]
    ((('A', (('boring', ','), 'bad')), (',', (('unsatisfactory', 'film'), '.'))), (20, 21), 0)
    >>> id2token # doctest: +ELLIPSIS
    {0: 'funny', 1: ',', 2: '.', 3: 'engaging', 4: 'film', 5: 'warm', 6: 'A', 7: 'bad', 8: 'boring', 9: 'unsatisfactory'...
    """
    # assigning row index to each word/phrase, more basic words/phrases should have smaller indices
    # group each word/phrase by the depth, 0 from the leaf and increases as going up

    # Note:
    # be careful if the labels for the same phrase is NOT consistent

    all_tokens = []  # place to store the final result
    collected_tokens = set()
    all_labels = []

    token2id = {}
    id2token = {}
    
    while len(trees) > 0:
        shallower_trees = []
        
        # collect the leaf nodes
        for t in trees:
            tokens_with_labels = set(
                get_leaves_with_labels(t)
            )

            # not all tokens are harvested
            # only the new ones
            new_tokens_with_labels = []
            for token, label in tokens_with_labels:
                if token not in collected_tokens:
                    new_tokens_with_labels.append((token, label))
                
            tokens, labels = zip(*new_tokens_with_labels)
            
            # update the id/token mapping
            token2id.update({
                token: i+len(all_tokens)
                for i, token in enumerate(tokens)
            })

            id2token.update({
                i+len(all_tokens): token
                for i, token in enumerate(tokens)
            })
            
            # add new tokens, their children and their labels
            all_tokens += [
                (tok, # the token
                 ((token2id[tok[0]], token2id[tok[1]]) # children node id if has children
                  if isinstance(tok, tuple) 
                  else (-1, -1)), #for single words
                 l) # the label
                for tok, l in zip(tokens, labels)
            ]
            collected_tokens |= set(tokens)
            
            try:
                shallower_trees.append(merge_leaves(t))
            except CannotMergeAnyMoreException:
                pass

        trees = shallower_trees# we consider the shallower trees now
    
    return all_tokens, token2id, id2token

if __name__ == "__main__":
    import doctest
    doctest.testmod()
