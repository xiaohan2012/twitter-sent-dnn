import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    def __init__(self, 
                 rng, 
                 input, 
                 n_in, 
                 n_out,
                 W = None,
                 b = None
    ):
        """ Initialize the parameters of the logistic regression

        :type: rng: 
        :param: rng: the random number generator
        
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        W and b: theano.tensor.TensorType
        """        
        if W:
            self.W = W
        else:
            self.W = theano.shared(value = np.asarray(rng.normal(0, 0.05, (n_in, n_out)),
                                                      dtype = theano.config.floatX
                                                  ), 
                                   name = 'logreg_W',
                                   borrow = True
                               )
        if b:
            self.b = b
        else:

            self.b = theano.shared(value = np.asarray(
                np.zeros((n_out, )),
                dtype = theano.config.floatX
            ),
                                   name = 'logreg_b',
                                   borrow = True)

        # the probability of labels given the data
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        # the predicted labels
        self.pred_y = T.argmax(self.p_y_given_x, axis = 1)

        self.params = [self.W, self.b]
        self.param_shapes = [(n_in, n_out), (n_out, )]

    def nnl(self, y):
        """
        negative log-likelihood
        y, the correct label
        """
        return T.mean(
            -T.log(self.p_y_given_x[T.arange(y.shape[0]), y])
        )
        
    def errors(self, y):
        """
        the error rate

        :type y: theano.tensor.ivector
        :param y: the class labels to be compared with
        """
        assert y.ndim == self.pred_y.ndim
        assert y.dtype.startswith('int')

        return T.mean(T.neq(self.pred_y, y))

from util import load_data

def train_and_test(learning_rate, batch_size,
                   n_epochs=1000):
    # get the data
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # define test error function
    x = T.dmatrix('x')
    y = T.ivector('y')

    index = T.lscalar('index') # the batch index

    model = LogisticRegression(x, 28*28, 10)
    cost = model.nnl(y)
    
    g_W = T.grad(cost = cost, wrt = model.W)
    g_b = T.grad(cost = cost, wrt = model.b)

    test_model = theano.function(inputs = [index], 
                                 outputs = model.errors(y),
                                 givens = {
                                     x: test_set_x[index * batch_size: (index+1) * batch_size],
                                     y: test_set_y[index * batch_size: (index+1) * batch_size],
                                 })

    validate_model = theano.function(inputs = [index], 
                                 outputs = model.errors(y),
                                 givens = {
                                     x: valid_set_x[index * batch_size: (index+1) * batch_size],
                                     y: valid_set_y[index * batch_size: (index+1) * batch_size],
                                 })

    train_model = theano.function(inputs = [index], 
                                  outputs = cost, 
                                  updates = [
                                      (model.W, model.W - learning_rate * g_W),
                                      (model.b, model.b - learning_rate * g_b)
                                  ],
                                  givens = {
                                      x: train_set_x[index * batch_size: (index+1) * batch_size],
                                      y: train_set_y[index * batch_size: (index+1) * batch_size]
                                  }
    )
    
    #the training loop
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        print "At epoch {epoch}".format(epoch = epoch)
        for minibatch_index in xrange(n_train_batches):
            train_err = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                   'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                       epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                       (
                           '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                           epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break
        

if __name__ == "__main__":
    train_and_test(0.1, 600)            
