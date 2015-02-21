import numpy as np

def assert_matrix_eq(actual, expected, name):
    try:
        assert (np.abs(actual - expected) < 1e-5).all()
        print "%s: OK" %(name)
    except AssertionError:
        print "%s: Fail" %(name)
        print "Actual:"
        print actual
        print "Expected:"
        print expected
        print "Equality matrix:"
        print np.abs(actual - expected) < 1e-5
