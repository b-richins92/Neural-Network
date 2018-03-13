import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
			Represents the constructor for the class object Network where
			we will be implementing our neural network code. sizes represents
			a list of parameters defining the number of layers in the network
			as well as the number of neurons in each layer. Initialize the biases
			and weights randomly using numpy randon number generator. Note we use
			zip which returns an iterator of n-tuples based off of the iterable object.
		"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
			function to return the output of our network given a specific
			input 'a'. Notice all we are doing is moving from layer to layer.
			If we choose a 4 layer network, the process of moving from layer 2
			to 3 would involve a'=(wa+b) where a' would be the new parameter to
			move from layer 3 o 4.
		"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def gradDescent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Gradient descent designed to train the data/network. Makes use of
			stochastic processes to average the test data. 'training_data' is a list of
			tuples ``(x, y)`` representing the training inputs and the desired outputs. Epochs
			refers to each successive batch of training data and will be how we evaluate
			whether our network is learning. Eta is simply the parameter involved with gradient
			descent that moves it in a negative direction.
		"""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
			Update the network's weights and biases by applying
        	gradient descent using backpropagation to a single mini batch.
        """
        var_b = [np.zeros(b.shape) for b in self.biases]
        var_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_var_b, delta_var_w = self.backprop(x, y)
            var_b = [nb+dnb for nb, dnb in zip(var_b, delta_var_b)]
            var_w = [nw+dnw for nw, dnw in zip(var_w, delta_var_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, var_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, var_b)]

    def backprop(self, x, y):
        """Return a tuple ``(var_b, var_w)`` representing the
        gradient for the cost function C(x).  `var_b' and
        `var_w` are layer-by-layer lists of arrays, similar
        to 'self.biases` and `self.weights`."""
        var_b = [np.zeros(b.shape) for b in self.biases]
        var_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        var_b[-1] = delta
        var_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            var_b[-l] = delta
            var_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (var_b, var_w)

    def evaluate(self, test_data):
        """
			Return the number of test inputs for which the neural
        	network outputs the correct result.
		"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activations."""
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
