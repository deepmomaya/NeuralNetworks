import numpy as np

class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        self.initialize_weights()
        # Initialize the linear associator neural network with given parameters

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initialize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        dim = self.input_dimensions
        nn = self.number_of_nodes 
        if seed is not None:
            np.random.seed(seed)
            self.weights = np.random.randn(nn, dim)
        else:
            self.weights = np.random.randn(nn, dim)
        # Initialize weights using random numbers and seed

    def set_weights(self, W):
        """
        This function sets the weight matrix.
        :param W: Weight matrix
        :return: None if the input matrix, W, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        dim = self.input_dimensions
        nn = self.number_of_nodes
        self.weights = W
        if W.shape != (nn, dim):
            return -1
        else:
            return None
        # Set weight matrix and check for correct shape

    def get_weights(self):
        """
        This function should return the weight matrix.
        :return: Weight matrix
        """
        return self.weights
        # Return the weight matrix

    def predict(self, X):
        """
        Make a prediction on an array of inputs.
        :param X: Array of input [input_dimensions, n_samples].
        :return: Array of model outputs [number_of_nodes, n_samples]. This array is a numerical array.
        """
        tf = self.transfer_function
        w = self.weights
        function = np.dot(w, X)
        if tf == "Hard_limit":
            prediction = function
            prediction[function >= 0] = 1
            prediction[function < 0] = 0
            return prediction
        elif tf == "Linear":
            return function
        # Predict the output after the model gets trained

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data and the targets,
        this function adjusts the weights using the pseudo-inverse rule.
        :param X: Array of input [input_dimensions, n_samples].
        :param y: Array of desired (target) outputs [number_of_nodes, n_samples].
        """
        z = np.linalg.pinv(X)
        self.weights = np.dot(y, z)
        # Adjust the weights using the pseudo-inverse rule

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions, n_samples].
        :param y: Array of desired (target) outputs [number_of_nodes, n_samples].
        :param num_epochs: Number of times training should be repeated over all input data.
        :param batch_size: Number of samples in a batch.
        :param alpha: Learning rate.
        :param gamma: Controls the decay.
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb".
        :return: None
        """
        for m in range(num_epochs):
            for n in range(0, X.shape[1], batch_size):
                batch_data = n + batch_size
                if batch_data > X.shape[1]:
                    batch_data = X.shape[1]
                xx = X[:, n:batch_data]
                yy = y[:, n:batch_data]
                out = yy - self.predict(xx)
                self.weights += alpha * (np.dot(out, xx.T))
        # Train the weights for the linear associator neural network

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions, n_samples].
        :param y: Array of desired (target) outputs [number_of_nodes, n_samples].
        :return: Mean squared error.
        """
        return np.square(np.subtract(y, self.predict(X))).mean()
        # Calculate the mean squared error
