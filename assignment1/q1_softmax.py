import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        #Added by Jiarong
#softmax(x) = softmax(x + c)
#Note: In practice, we make use of this property and choose c = -maxi xi when computing softmax
#probabilities for numerical stability (i.e., subtracting its maximum element from all elements of x)
#so here: softmax(x) = softmax(x-maxi xi)

        #axis = 1 means add by rows
        #keepdims = True, explaination below
        c = - np.max(x, axis = 1, keepdims=True)
        x = x + c
        
        x = np.exp(x) / (np.sum(np.exp(x), axis = 1)).reshape(-1,1)

        #raise NotImplementedError
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
                
        c = - np.max(x)
        x = x + c
        x = np.exp(x) / np.sum(np.exp(x))
 
        #raise NotImplementedError
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()

    
  
'''
Added by Jiarong
explain: why use keepdims=True
https://stackoverflow.com/questions/40927156/what-the-role-of-keepdims-in-python

Consider a small 2d array:

In [180]: A=np.arange(12).reshape(3,4)
In [181]: A
Out[181]: 
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
Sum across rows; the result is a (3,) array

In [182]: A.sum(axis=1)
Out[182]: array([ 6, 22, 38])
But to sum (or divide) A by the sum requires reshaping

In [183]: A-A.sum(axis=1)
...
ValueError: operands could not be broadcast together with shapes (3,4) (3,) 
In [184]: A-A.sum(axis=1)[:,None]   # turn sum into (3,1)
Out[184]: 
array([[ -6,  -5,  -4,  -3],
       [-18, -17, -16, -15],
       [-30, -29, -28, -27]])
If I use keepdims, "the result will broadcast correctly against" A.

In [185]: A.sum(axis=1, keepdims=True)   # (3,1) array
Out[185]: 
array([[ 6],
       [22],
       [38]])
In [186]: A-A.sum(axis=1, keepdims=True)
Out[186]: 
array([[ -6,  -5,  -4,  -3],
       [-18, -17, -16, -15],
       [-30, -29, -28, -27]])

'''