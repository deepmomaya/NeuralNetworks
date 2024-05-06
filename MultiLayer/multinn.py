import tensorflow as tf
import numpy as np

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize a multi-layer neural network.

        :param input_dimension: The number of dimensions for each input data sample.
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activation_function = []
        self.loss_function = None
        # Initialize the multi-layer neural network

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
        Add a dense layer to the neural network.

        :param num_nodes: Number of nodes in the layer.
        :param transfer_function: Activation function for the layer.
                                  Possible values are: "Linear", "Relu", "Sigmoid".
        :return: None
        """
        w = self.weights
        b = self.biases
        dim = self.input_dimension
        if not self.weights:
            w.append(tf.Variable(np.random.randn(dim, num_nodes), trainable=True))
            b.append(tf.Variable(np.random.randn(num_nodes,), trainable=True))
        else:
            w.append(tf.Variable(np.random.randn(w[-1].shape[1], num_nodes), trainable=True))
            b.append(tf.Variable(np.random.randn(num_nodes,), trainable=True))
        self.activation_function.append(transfer_function.lower())
        # Add a dense layer to the neural network

    def get_weights_without_biases(self, layer_number):
        """
        Get the weight matrix (without biases) for a given layer.

        :param layer_number: Layer number starting from layer 0.
        :return: Weight matrix for the given layer (not including the biases).
                 The shape of the weight matrix should be [input_dimensions][number of nodes].
        """
        return self.weights[layer_number]
        # Return the weight matrix for the given layer

    def get_biases(self, layer_number):
        """
        Get the biases for a given layer.

        :param layer_number: Layer number starting from layer 0.
        :return: Biases for the given layer.
                 The shape of the biases should be [1][number_of_nodes].
        """
        return self.biases[layer_number]
        # Return the biases for the given layer

    def set_weights_without_biases(self, weights, layer_number):
        """
        Set the weight matrix (without biases) for a given layer.

        :param weights: Weight matrix (without biases).
                        The shape of the weight matrix should be [input_dimensions][number of nodes].
        :param layer_number: Layer number starting from layer 0.
        :return: None
        """
        self.weights[layer_number] = weights
        # Set the weight matrix for the given layer

    def set_biases(self, biases, layer_number):
        """
        Set the biases for a given layer.

        :param biases: Biases.
                       The shape of the biases should be [1][number_of_nodes].
        :param layer_number: Layer number starting from layer 0.
        :return: None
        """
        self.biases[layer_number] = biases
        # Set the biases for the given layer

    def calculate_loss(self, y, y_hat):
        """
        Calculate the sparse softmax cross-entropy loss.

        :param y: Array of desired (target) outputs [n_samples].
                  This array includes the indexes of the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: Loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        # Calculate the sparse softmax cross-entropy loss

    def predict(self, X):
        """
        Predict the output of the multi-layer network for given input samples.

        :param X: Array of input [n_samples, input_dimensions].
        :return: Array of outputs [n_samples, number_of_classes].
                 This array is a numerical array.
        """
        w = self.weights
        z = tf.Variable(X)
        for i in range(len(w)):
            m = tf.matmul(z,self.get_weights_without_biases(i))
            prediction = tf.add(m,self.get_biases(i))
            if self.activation_function[i] == "linear":
                z = prediction
            elif self.activation_function[i] == "sigmoid":
                z = tf.nn.sigmoid(prediction)
            elif self.activation_function[i] == "relu":
                z = tf.nn.relu(prediction)
        return z
        # Predict the output for the neural network

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
        Train the neural network by adjusting the weights and biases of all the layers.

        :param X_train: Array of input [n_samples, input_dimensions].
        :param y_train: Array of desired (target) outputs [n_samples].
                         This array includes the indexes of the desired (true) class.
        :param batch_size: Number of samples in a batch.
        :param num_epochs: Number of times training should be repeated over all input data.
        :param alpha: Learning rate.
        :return: None
        """
        w = self.weights
        b = self.biases
        for e in range(num_epochs):
            for i in range(0,X_train.shape[0],batch_size):
                b = i + batch_size
                if b > X_train.shape[0]:
                    b = X_train.shape[0]
                batchX = tf.Variable(X_train[i:b, :])
                batchy = tf.Variable(y_train[i:b])
                with tf.GradientTape(persistent=True) as gt:
                    gt.watch(w)
                    gt.watch(self.biases)
                    y_hat = self.predict(batchX)
                    loss = self.calculate_loss(batchy,y_hat)
                for y in range(len(w)):
                    ww = tf.scalar_mul(alpha,gt.gradient(loss, self.get_weights_without_biases(y)))
                    bb = tf.scalar_mul(alpha,gt.gradient(loss,self.get_biases(y)))
                    sw = tf.subtract(self.get_weights_without_biases(y),ww)
                    sb = tf.subtract(self.get_biases(y),bb)
                    self.set_weights_without_biases(sw,y)
                    self.set_biases(sb,y)         
        # Train the neural network

    def calculate_percent_error(self, X, y):
        """
        Calculate the percent error.

        :param X: Array of input [n_samples, input_dimensions].
        :param y: Array of desired (target) outputs [n_samples].
                  This array includes the indexes of the desired (true) class.
        :return: Percent error
        """
        err = 0
        pred = self.predict(X)
        predout = np.argmax(pred,axis=1)
        for i in range(pred.shape[0]):
            if predout[i] != y[i]:
                err += 1
        percent_error = err/pred.shape[0]
        return percent_error
        # Calculate the percent error

    def calculate_confusion_matrix(self, X, y):
        """
        Calculate the confusion matrix.

        :param X: Array of input [n_samples, input_dimensions].
        :param y: Array of desired (target) outputs [n_samples].
                  This array includes the indexes of the desired (true) class.
        :return: Confusion matrix[number_of_classes, number_of_classes].
                 Confusion matrix should be shown as the number of times that
                 an image of class n is classified as class m.
        """
        pred = self.predict(X)
        predout = np.argmax(pred,axis=1)
        confusion_matrix = np.zeros((len(np.unique(y)),len(np.unique(y))))
        for i in range(len(y)):
            m = y[i]
            n = predout[i]
            confusion_matrix[m.astype(int)][n.astype(int)] = confusion_matrix[m.astype(int)][n.astype(int)] + 1
        return confusion_matrix
        # Calculate the confusion matrix
