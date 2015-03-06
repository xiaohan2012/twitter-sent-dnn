import numpy as np

def assert_matrix_eq(actual, expected, name):
    try:
        assert (np.abs(actual - expected) < 1e-5).all()
        print "`eq` of `%s`: OK" %(name)
    except AssertionError:
        print "`eq` of `%s`: Fail" %(name)
        print "Actual:"
        print actual
        print "Expected:"
        print expected
        print "Equality matrix:"
        print np.abs(actual - expected) < 1e-5

def assert_matrix_neq(actual, expected, name, verbose = False):
    def print_verbose():
        print "First:"
        print actual
        print "Second:"
        print expected
        print "Difference matrix:"
        print np.abs(actual - expected)

    try:
        assert (np.abs(actual - expected) > 1e-5).any()
        print "`neq` of %s: OK" %(name)
        if verbose:
            print_verbose()

    except AssertionError:
        print "`neq` of `%s`: Fail" %(name)
        print_verbose()

def assert_about_eq(actual, expected, name):
    try:
        assert np.abs(actual - expected) < 1e-5
        print "%s: OK" %(name)
    except AssertionError:
        print "%s: Fail" %(name)
        print "Actual:"
        print actual
        print "Expected:"
        print expected
