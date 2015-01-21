import pdb
def matching_paren_position(s, left_paren_location):
    """
    find the position of the parenthsis that matched the given one
    >>> matching_paren_position('(())', 0)
    3
    >>> matching_paren_position('(())', 1)
    2
    >>> matching_paren_position('(() (()()) )', 4)
    9
    """
    i = left_paren_location
    depth = 0
    while True:
        i += 1
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            if depth == 0:
                return i
            depth -= 1
    raise ValueError("Cannot find matching parenthesis")


def parse(s):
    """
    Given string in PTB format

    parse it into tuple of tuple

    >>> parse('(2 (2 The) (2 Rock))')
    (2, (2, 'The'), (2, 'Rock'))
    >>> parse('(4 (3 gorgeously) (3 (2 elaborate) (2 continuation)))')
    (4, (3, 'gorgeously'), (3, (2, 'elaborate'), (2, 'continuation')))
    >>> parse('(1 (1 (2 a) (3 b)) (1 (1 c) (2 d)))')
    (1, (1, (2, 'a'), (3, 'b')), (1, (1, 'c'), (2, 'd')))
    """        
    # start and end position of the first pair of parenthesis
    s1 = s.find('(', 1)
    if s1 > 0: # we found it
        e1 = matching_paren_position(s, s1)
        assert e1 > 0 # then e1 should be found
        
        # start and end position of the second pair of parenthesis
        s2 = s.find('(', e1+1)
        assert s2 > 0, "find '(' from position %d in string '%s'" %(s1+1, s)
        
        e2 = matching_paren_position(s, s2)
        assert e2 > 0, "find ')' from position %d in string '%s'" %(e1+1, s)
        
        first_tuple_str = s[s1: e1+1]
        second_tuple_str = s[s2: e2+1]
    
        return (int(s[1]), 
                parse(first_tuple_str),
                parse(second_tuple_str),
            )
    else: #fail to find it
        return (int(s[1]), s[2:-1].strip())


def flatten_tree(t):
    """
    flattena a PTB tree, return:
    (a list of the text values, top node label)
    
    >>> t = parse('(4 (3 gorgeously) (3 (2 elaborate) (2 continuation)))')
    >>> flatten_tree(t)
    (['gorgeously', 'elaborate', 'continuation'], 4)

    >>> t = parse('(3 gorgeously)')
    >>> flatten_tree(t)
    (['gorgeously'], 3)

    >>> t = parse('(1 (1 (2 a) (3 b)) (1 (1 c) (2 d)))')
    >>> flatten_tree(t)
    (['a', 'b', 'c', 'd'], 1)

    >>> t = parse('(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 \\'s)) (2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 \\'\\')) (2 and)) (3 (2 that) (3 (2 he) (3 (2 \\'s) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) (2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) (2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .)))')
    >>> flatten_tree(t)
    (['The', 'Rock', 'is', 'destined', 'to', 'be', 'the', '21st', 'Century', "'s", 'new', '``', 'Conan', "''", 'and', 'that', 'he', "'s", 'going', 'to', 'make', 'a', 'splash', 'even', 'greater', 'than', 'Arnold', 'Schwarzenegger', ',', 'Jean-Claud', 'Van', 'Damme', 'or', 'Steven', 'Segal', '.'], 3)
    """

    def collect_words(tree):
        if len(tree) == 2: # leaf node
            return [tree[1]]
        else:
            return collect_words(tree[1]) + collect_words(tree[2])

    if len(t) == 2: # just a node, not actually a tree
        return ([t[1]], t[0])
    else:
        return (
            collect_words(t[1]) + collect_words(t[2]),
            t[0]
        )
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
        
