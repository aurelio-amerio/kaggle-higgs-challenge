"""
  Aurelio_Amerio_MLPClassifier_lib.py

  MLP classification library

  Features implemented:
  -batch training
  -dropout
  -wights and biases regularization
  -esponential decay of the learning rate
  -leaky relu activation function
  -validation set comparison
  -timing of the fit function
  -fit, predict and score methods


  ----------------------------------------------------------------------
  author:       Aurelio Amerio (aurelio.amerio@edu.unito.it)
  Student ID:   QT08313
  Date:         03/08/2018
  ----------------------------------------------------------------------
"""

import tensorflow as tf
import numpy as np
from time import time
from sklearn.model_selection import train_test_split


class MLPClassifier:
    """
    Class to approximate functions using a Multi Layer Perceptron Neural Network
    """

    def __init__(self, learning_rate=0.001, random_seed=None, n_layers=2, n_nodes=(50, 50), n_features=28 * 28,
                 n_classes=10):
        """
        :param learning_rate: learning rate for the Adam optimizer
        :param random_seed: set the seed for the initialization of the weights and biases matrices
        :param n_layers: int, number of hidden layers, must be >= 1
        :param n_nodes: tuple of ints, number of nodes per hidden layer
        :param n_features: number of features of the training set
        :param n_classes: number of outputs of the training set

        """

        # define various quantities
        self.random_seed = random_seed
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_classes = n_classes
        self.starting_learning_rate = learning_rate
        self._index_in_epoch = 0
        self._num_samples = 0
        self._epochs_completed = 0

        # I create a tensorflow graph
        self.g = tf.Graph()
        with self.g.as_default():
            # set random seed
            if random_seed is not None:
                tf.set_random_seed(self.random_seed)

            # call build function which creates all the variables and placeholders I need
            self._build()

        # create session with selected graph
        self.sess = tf.Session(graph=self.g)

        self.training_costs = []
        self.test_costs = []
        self.training_accuracy = []
        self.test_accuracy = []

    def _build(self):
        """
        support function for the constructor
        """
        # arrays necessary to define a layer
        self.w_layers = []
        self.bias_layers = []
        self.layers = []

        # placeholders
        self.X = tf.placeholder(shape=[None, self.n_features], dtype=tf.float32, name="X_input")
        self.y = tf.placeholder(shape=[None, self.n_classes], dtype=tf.float32, name="y_input")
        # placeholders fow w and b regularization
        self.beta_w = tf.placeholder(shape=(), dtype=tf.float32, name="beta_w")
        self.beta_b = tf.placeholder(shape=(), dtype=tf.float32, name="beta_b")

        # keep probability for dropout
        self.keep_probs_array_ph = tf.placeholder(shape=[None], dtype=tf.float32, name="dropout_keep_prob")

        # sets w and b for the first layer based on the input
        self.w_layers.append(tf.Variable(tf.random_normal([self.n_features, self.n_nodes[0]])))
        self.bias_layers.append(tf.Variable(tf.random_normal([self.n_nodes[0]])))

        self.layers.append(
            tf.nn.dropout(tf.nn.leaky_relu(tf.add(tf.matmul(self.X, self.w_layers[0]), self.bias_layers[0])),
                          self.keep_probs_array_ph[0]))

        # sets w and b for all the other layers
        if self.n_layers > 1:
            for i in range(1, self.n_layers):
                self.w_layers.append(tf.Variable(tf.random_normal([self.n_nodes[i - 1], self.n_nodes[i]])))
                self.bias_layers.append(tf.Variable(tf.random_normal([self.n_nodes[i]])))
                self.layers.append(
                    tf.nn.dropout(
                        tf.nn.leaky_relu(tf.add(tf.matmul(self.layers[i - 1], self.w_layers[i]), self.bias_layers[i])),
                        self.keep_probs_array_ph[i]))

        # create the output layer which needs to be optimized

        self.output = tf.Variable(tf.random_normal([self.n_nodes[self.n_layers - 1], self.n_classes]))
        self.bias_output = tf.Variable(tf.random_normal([self.n_classes]))

        self.output_layer = tf.matmul(self.layers[self.n_layers - 1], self.output) + self.bias_output

        # create loss function

        # regularize
        summ_of_reg = 0
        # self.regularizer = tf.nn.l2_loss(self.w_layers)

        for i in range(len(self.w_layers)):
            summ_of_reg += self.beta_w * tf.nn.l2_loss(self.w_layers[i])
            summ_of_reg += self.beta_b * tf.nn.l2_loss(self.bias_layers[i])

        # add the output layer to the reguralizer
        summ_of_reg += self.beta_w * tf.nn.l2_loss(self.output)
        summ_of_reg += self.beta_b * tf.nn.l2_loss(self.bias_output)

        self.l2_regularizer = summ_of_reg
        self.dloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.output_layer))
        # self.dloss = tf.nn.l2_loss(self.output_layer - self.y)
        self.dloss_reg = tf.reduce_mean(self.dloss + self.l2_regularizer)

        # array which contains 0, 1 depending on whether the prediction is right or wrong
        self.correct_prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.y, 1),
                                           name="correct_prediction")
        # n_correct_predictions/n_samples
        self.tf_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="tf_accuracy")

    # ------------------------------------#
    def _next_batch(self, batch_size):
        """
        Return the next `batch_size` examples from the training dataset.

        Adapted from tensorflow.contrib.learn.python.learn.datasets.mnist
        """

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_samples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_samples)
            np.random.shuffle(perm)
            self.X_train = self.X_train[perm]
            self.y_train = self.y_train[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_samples
        end = self._index_in_epoch
        return self.X_train[start:end], self.y_train[start:end]

    def fit(self, X_train, y_train, max_epochs=100, keep_probs=None, batch_size=None, display_step=20, beta_w=0,
            beta_b=0, validation_split=None, decay_rate=None, decay_steps=100000, verbose=True):
        """
        Method which finds the best fit for given X_train, y_train and selected number of iterations.

        :param X_train: A vector of shape [N,1] containing the data to be fitted
        :param y_train: A vector of shape [N,1] containing the labels for the data to be fitted
        :param max_epochs: max number of iterations for the fit method
        :param keep_probs:  keep probability on dropout, if 1. no dropout will be performed
        :param validation_split:  relative size of the validation set, if None no validation set will be created
        :param batch_size:  size of the batch for batch training
        :param display_step:  number of iterations after which the algorithm should store the cost function value and
                              the accuracy
        :param beta_w: L2 regularization parameter for the weights
        :param beta_b: L2 regularization parameter for the bias
        :param decay_rate:  If set, apply exponential decay to the learning rate
        :param decay_steps: number of decay steps
        :param verbose: if true, print optimization progress and results
        :return: self
        """
        # if validation set is required, split the data
        if validation_split is not None:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train,
                                                                                  test_size=validation_split)
        else:
            self.X_train = X_train
            self.X_val = None
            self.y_train = y_train
            self.y_val = None

        self._num_samples = len(self.X_train)

        max_epochs = max_epochs
        beta_w = beta_w
        beta_b = beta_b

        # set batch size
        if batch_size is None:
            self.batch_size = len(self.X_train) - 1
        else:
            self.batch_size = batch_size

        self.training_costs = []
        self.val_costs = []
        self.training_accuracy = []
        self.val_accuracy = []

        # initialize keep prob array
        if keep_probs is not None:
            self.keep_probs_array = keep_probs
            # fill one array with ones, same shape as keep_probs_array, for prediction and cost evaluation
            self.keep_probs_array_one = np.ones(shape=self.n_layers)
        else:
            self.keep_probs_array = np.ones(shape=self.n_layers)
            self.keep_probs_array_one = np.ones(shape=self.n_layers)

        # start variable initialization
        with self.g.as_default():
            # setup exponential learning rate
            global_step = tf.Variable(0, trainable=False)
            if decay_rate is not None:
                self.learning_rate = tf.train.exponential_decay(self.starting_learning_rate, global_step,
                                                                decay_steps=decay_steps, decay_rate=decay_rate,
                                                                staircase=True)

                # Passing global_step to minimize() will increment it at each step.
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        name="AdamOptimizer").minimize(
                    self.dloss_reg, global_step=global_step)
            else:
                self.learning_rate = self.starting_learning_rate

                # Passing global_step to minimize() will increment it at each step.
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        name="AdamOptimizer").minimize(
                    self.dloss_reg)

            init_op = tf.global_variables_initializer()

            self.sess.run(init_op)

        print("starting the fit")
        t1 = time()
        n_batches = np.ceil(len(self.X_train) / self.batch_size)

        # optimization loop
        for i in np.arange(max_epochs):
            for j in np.arange(n_batches):
                X_batch, y_batch = self._next_batch(self.batch_size)

                self.sess.run(self.optimizer,
                              feed_dict={self.X: X_batch, self.y: y_batch,
                                         self.keep_probs_array_ph: self.keep_probs_array, self.beta_w: beta_w,
                                         self.beta_b: beta_b})

            if not i % display_step:
                cost = self.sess.run(self.dloss,
                                     feed_dict={self.X: self.X_train, self.y: self.y_train,
                                                self.keep_probs_array_ph: self.keep_probs_array_one})

                self.training_costs.append(cost)

                accuracy = self.sess.run(self.tf_accuracy,
                                         feed_dict={self.X: self.X_train, self.y: self.y_train,
                                                    self.keep_probs_array_ph: self.keep_probs_array_one})

                self.training_accuracy.append(accuracy)

                # if validation_split is required, calculate the cost function for the validation set

                if validation_split is not None:
                    cost = self.sess.run(self.dloss,
                                         feed_dict={self.X: self.X_val, self.y: self.y_val,
                                                    self.keep_probs_array_ph: self.keep_probs_array_one})
                    self.val_costs.append(cost)

                    accuracy = self.sess.run(self.tf_accuracy,
                                             feed_dict={self.X: self.X_val, self.y: self.y_val,
                                                        self.keep_probs_array_ph: self.keep_probs_array_one})

                    self.val_accuracy.append(accuracy)
                    if verbose is True:
                        print("progress: {:{width}.2f}%\ttrain acc={:.4f}\tval acc={:.4f}".format(i / max_epochs * 100,
                                                                                                  self.training_accuracy[
                                                                                                      int(
                                                                                                          i / display_step)],
                                                                                                  self.val_accuracy[
                                                                                                      int(
                                                                                                          i / display_step)],
                                                                                                  width=6))
                else:
                    if verbose is True:
                        print("progress: {:{width}.2f}%\ttrain acc={:.4f}".format(i / max_epochs * 100,
                                                                                  self.training_accuracy[
                                                                                      int(i / display_step)], width=6))

        t2 = time()
        execution_time = t2 - t1

        print("fit routine completed, execution time: {:.2f}s".format(execution_time))

        return self

    def predict(self, X_predict):
        """
        Method to compute the expected value for X of shape [N,1], given the model created by fit
        Must call "fit" first
        :param X_predict: matrix of values to predict of shape [N,1]
        :return: predicted y value
        """
        y_pred = self.sess.run(tf.argmax(self.output_layer, 1),
                               feed_dict={self.X: X_predict, self.keep_probs_array_ph: self.keep_probs_array_one})
        return y_pred

    def score(self, X_test, y_test):
        """
        returns the accuracy of the model on the given validation set
        :param X_test: validation set
        :param y_test: test labels
        :return: accuracy of the model on the validation set
        """
        X_test = X_test
        y_test = y_test

        accuracy = self.sess.run(self.tf_accuracy,
                                 feed_dict={self.X: X_test, self.y: y_test,
                                            self.keep_probs_array_ph: self.keep_probs_array_one})
        return accuracy

    def __del__(self):
        self.sess.close()
