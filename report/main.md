# Twitter Sentiment Prediction using Convolutional Neural Network

Author: **Han Xiao** and **Yao Lu**  
Date: Febaurary, 16, 2015

## Objective

In this project, we built up a Web application that:

1. predicts sentiment labels(positive or negative) for tweets
2. allows the user to see the result

The Web application is implemented using Tornado and a pre-trained Convolutional Neural Network(CNN), implemented in Theano, is used as the sentiment prediction engine.

## System architecture
<!-- Yao -->
<!-- Technical choices and basic introduction to the UI -->

## Data

Two data sources are used to build up this system:

1. Twitter140: used to train the CNN
2. live data: crawled from Twitter's newest posts used for prediction

### [Twitter140](http://help.sentiment140.com/for-students/)
This data source contains roughly 1.6 million, 872 and 1821 labelled tweets for training, validation and testing respectively. Tweets are givens labels, either *positive* or *negative*. For training data, labelling is done in a automatic way based on emoticons. For example, if the tweet contains a "happy" icon, e.g, *:)*, then it's positive, while in case it contains a "unhappy" one, e.g, *:(*, it's negative. Finally, the emoticons are removed from the tweets. Validation and testing data are manually labelled.

### Live Twitter data
We provide a script which can crawl tweets which contain sentiment emoticons in real-time. As in the training set, tweets which contain positive emoticons such as :), : ), =), :D and :-) are labeled as positive samples, which contain negative emoticons :(, : (  and :-( are labeled as negative samples and which contain both are removed. The data crawled with this script are in the same format as in the training set.

## Algorithm

Dynamic Convolution Neural Network, as described in [Kalchbrenner, 2014](http://nal.co/papers/Kalchbrenner_DCNN_ACL14), is used as sentiment prediction algorithm. The network layers are set as follows:

1. Embedding layer: set as the lowest layer, to get the vector representation for individual words to form the matrix representation of the sentence
2. Convolutional Layer: extracts local features  by performing 2d convoltuion on the sentence matrix
3. K-max pooling layer: extracts kth strongest signals on a per-feature basis
4. Folding layer: adds interaction among features by "folding" the input matrix
5. Logistic regression layer: the final layer that makes the prediction by assigning scores to each output label

Input data is fed into the first layer and propagate forward to the last layer thorough intermediate layers. Layer 1 and Layer 5 are always used as the lowest and highest layer respectively. Layer 2 to 4 can be repeated as a whole to form a neural network that contains multiple rounds of convolution-pooling-folding opetations.

## Model training

### Basic
[Backpropagation](http://en.wikipedia.org/wiki/Backpropagation) is the basic framework for training this model. Stochastic gradient descent is optimization method.

### Training techniques
As training deep neural architecture can be difficult, several techniques are used to facilitate this process:

1. *Dropoutd*, [Hinto, 2012](http://arxiv.org/pdf/1207.0580.pdf): as a way to model averaging
2. *Adadelta*, [Zeiler, 2012](http://arxiv.org/abs/1212.5701): to automically adjust the learning rate on a per-parameter basis 

### Parameter tuning

As a set of hyper-parameters, such as layer number, batch size, etc, need to be tuned, random search in combination with grid search, ([Begio, 2012](http://arxiv.org/abs/1206.5533)) is applied.

### Final result

The final set of hyper-parameters achieving the empirical best results are as follow:

1. *batch size*: 10
2. *convoltuion layer number*: 2
3. *k value*:
4. *filter width*:
5. *L2 regularizer parameter*:
6. *feature map number*:
7. *dropout rates*: 

Under this parameter setting, the model achieves validation accuracy of % and test accuracy of %.


