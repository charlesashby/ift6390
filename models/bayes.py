from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np


class AbstractDensityEstimator(object):
    """ Abstract density estimator
        - Contains the signature for the 'train' and 'compute_predictions' method
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, training_data):
        """ Performs the training of the density estimator
            :param training_data: A numpy matrix of shape (n, d).
            :return: Nothing
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_predictions(self, test_data):
        """ Computes the estimated density for each test sample point
            :param test_data: A numpy matrix of shape (n, d).
            :return: A numpy vector of shape (n,) representing the estimated log density of each test point.
        """
        return NotImplementedError()


class DiagonalGaussian(AbstractDensityEstimator):
    """ Diagonal Gaussian Density Estimator """
    def __init__(self, nb_dims):
        """ Constructor
            :param nb_dims: The number of features each point in the dataset has.
        """
        self.d = nb_dims
        self.mu = np.zeros((self.d,))                   # Vector of dimension d
        self.sigma_2 = np.ones((self.d,))               # Vector of dimension d

    def train(self, training_data):
        """ Performs the training of the density estimator
            :param training_data: A numpy matrix of shape (n, d).
            :return: Nothing
        """
        n_train, d = training_data.shape
        assert d == self.d
        self.mu = np.mean(training_data, axis=0)
        self.sigma_2 = 1./n_train * np.sum((training_data - self.mu) ** 2., axis=0)
        self.sigma_2[self.sigma_2 == 0.0] = 1e-10

    def compute_predictions(self, test_data):
        """ Computes the estimated density for each test sample point
            :param test_data: A numpy matrix of shape (n, d).
            :return: A numpy vector of shape (n,) representing the estimated log density of each test point.
        """
        n_test, d = test_data.shape
        assert d == self.d
        if np.prod(self.sigma_2) != 0.0:
            log_prob = -self.d * np.log(2.*np.pi)/2. - np.log(np.prod(self.sigma_2))/2.
        else:
            log_prob = -self.d * np.log(2. * np.pi) / 2. - np.log(1e-10) / 2.
        log_prob += -np.sum((test_data - self.mu)**2. / (2. * self.sigma_2), axis=1)
        return log_prob


class ParzenWindows(AbstractDensityEstimator):
    """ Soft Parsen Windows Density Estimator """
    def __init__(self, sigma):
        """ Constructor
            :param sigma: The std deviation to use for the Gaussian kernel
        """
        self.d = 1
        self.sigma = sigma                          # Vector of dimension d
        self.training_data = np.zeros((1, self.d))  # Matrix of size (n, d)

    def train(self, training_data):
        """ Performs the training of the density estimator
            :param training_data: A numpy matrix of shape (n_train, d).
            :return: Nothing
        """
        n_train, self.d = training_data.shape
        self.training_data = training_data

    def compute_predictions(self, test_data):
        """ Computes the estimated density for each test sample point
            :param test_data: A numpy matrix of shape (n_test, d).
            :return: A numpy vector of shape (n_test,)
            representing the estimated log density of each test point.
        """
        n_test, d = test_data.shape
        assert d == self.d
        kernelized = self._kernel(test_data)                                    # Matrix of size (n_test, n_train)
        log_prob = np.log(np.mean(kernelized, axis=-1))
        return log_prob

    def _kernel(self, test_data):
        """ Computes the kernel function between each training point and each test point
            :param test_data: A numpy matrix of shape (n_test, d)
            :return: A matrix of size (n_test, n_train) representing the kernel results between each
                     test point and each training_point
        """
        training_points = self.training_data[None, :, :]                        # Matrix of size (1, n_train, d)
        test_points = test_data[:, None, :]                                     # Matrix of size (n_test, 1, d)
        dist = np.sum(np.abs(training_points - test_points) ** 2., axis=-1)     # Matrix (n_test, n_train)
        kernelized = 1./((2. * np.pi)**(self.d/2.) * self.sigma**self.d)
        kernelized *= np.exp(-0.5 * dist / (self.sigma ** 2))
        return kernelized


class BayesClassifier(object):
    """ Bayes Classifier """

    def __init__(self, density_estimator):
        """ Constructor
            :param density_estimator: The instantiated density estimator
        """
        self.density_models = []
        self.log_priors = []
        self.nb_classes = 0
        self.density_estimator = density_estimator

    def train(self, training_data, training_labels):
        """ Performs the training of the density estimator
            :param training_data: A numpy matrix of shape (n, d).
            :param training_labels: A numpy vector of shape (n,)
            :return: Nothing
        """
        unique_labels = sorted(np.unique(training_labels))
        nb_train = len(training_labels)
        self.nb_classes = len(unique_labels)
        self.density_models = [deepcopy(self.density_estimator) for _ in range(self.nb_classes)]
        self.log_priors = []

        # Training density models and log priors
        for class_ix, class_label in enumerate(unique_labels):
            class_training_data = training_data[training_labels == class_label]
            self.density_models[class_ix].train(class_training_data)
            self.log_priors += [np.log(np.sum(training_labels == class_label) * 1. / nb_train)]

    def compute_predictions(self, test_data):
        """ Computes the estimated density for each test sample point
            :param test_data: A numpy matrix of shape (n, d).
            :return: A numpy vector of shape (n,) representing the index of the predicted class.
        """
        posterior = [self._posterior(class_ix, test_data) for class_ix in range(self.nb_classes)]   # Shape: (c, n)
        posterior = np.array(posterior).T                                                           # Shape: (n, c)
        return np.argmax(posterior, axis=-1)

    def _posterior(self, class_ix, test_data):
        """ Computes the unnormalized log posterior probability of a class given the data
            :param class_ix: The class ix (0 to nb_classes - 1)
            :param test_data: The test data to use to conditional the probability
            :return: The unnormalized log posterior probability
        """
        return self.density_models[class_ix].compute_predictions(test_data) + self.log_priors[class_ix][None]

