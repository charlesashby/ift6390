import numpy as np
import pickle
import random as rnd
from sklearn.metrics import confusion_matrix

SAVE_PATH = "C:\\Users\\Charles\\PycharmProjects\\fraud_detection\\checkpoints\\mlp"
LOGGING_PATH = "C:\\Users\\Charles\\PycharmProjects\\fraud_detection\\data\\log.txt"
data_dir = 'C:/Users/Charles/PycharmProjects/fraud_detection/'

# ---------------- #
# CONFUSION MATRIX #
#                  #
#
#
#                  #
# ---------------- #
"""
with open("C:\\Users\\Charles\\PycharmProjects\\fraud_detection\\data\\data_set.pickle", 'rb') as handle:
    train_x, train_y, test_x, test_y = pickle.load(handle)

    # CREATE TRAINING & VALIDATION SET

    idx_fraud_train = np.where(train_y[:, 0] == '0')[0]
    idx_non_fraud_train = np.where(train_y[:, 0] == '1')[0][:len(idx_fraud_train)]

    # idx_fraud_valid = np.where(train_y[:, 0] == '0')[0][:50]
    # idx_non_fraud_valid = np.where(train_y[:, 0] == '1')[0][:50]

    idx_fraud_test = np.where(test_y[:, 0] == '0')[0]
    idx_non_fraud_test = np.where(test_y[:, 0] == '1')[0][:len(idx_fraud_test)]

    # under_sampled_idx_valid = np.concatenate((idx_non_fraud_valid, idx_fraud_valid))
    under_sampled_idx_train = np.concatenate((idx_non_fraud_train, idx_fraud_train))
    under_sampled_idx_test = np.concatenate((idx_non_fraud_test, idx_fraud_test))

    # Shuffle the indexes
    np.random.shuffle(under_sampled_idx_train)
    np.random.shuffle(under_sampled_idx_test)
    # np.random.shuffle(under_sampled_idx_valid)

    trunc_train_x, trunc_train_y = train_x[under_sampled_idx_train, :], \
        train_y[under_sampled_idx_train, :].astype('float32')
    trunc_train_y = np.array([1 if trunc_train_y[i][0] == 0 else 0 for i in range(trunc_train_y.shape[0])])

    # trunc_valid_x, trunc_valid_y = train_x[under_sampled_idx_valid, :], \
    #     train_y[under_sampled_idx_valid, :].astype('float32')

    trunc_test_x, trunc_test_y = test_x[under_sampled_idx_test, :], \
        test_y[under_sampled_idx_test, :].astype('float32')
    trunc_test_y = np.array([1 if trunc_test_y[i][0] == 0 else 0 for i in range(trunc_test_y.shape[0])])

"""


class SVM():
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """

    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001):

        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):

        # Initialization
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernels[self.kernel_type]
        count = 0

        while True:
            count += 1
            alpha_prev = np.copy(alpha)

            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)

                if k_ij == 0:
                    continue

                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return



        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)

        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)

        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        return np.dot(alpha * y, X)

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if isinstance(y_i, np.ndarray):
            test = any(y_i != y_j)
        else:
            test = y_i !=y_j
        if test:
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_rnd_int(self, a,b,z):
        i = z
        while i == z:
            i = rnd.randint(a,b)
        return i

    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)

"""
if __name__ == '__main__':
    model = SVM(max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001)
    model.fit(trunc_train_x, trunc_train_y)
    predictions = np.int64(model.predict(trunc_test_x) >= 0)
    conf_matrix = confusion_matrix(y_true=trunc_test_y, y_pred=predictions)

    # >>>  conf_matrix
    # >>>  [[92,  1],
    #       [25, 68]]

    # on the complete data set
    predictions_full = np.int64(model.predict(test_x) >= 0)
    test_y = np.array([1 if test_y[i][0] == '0' else 0 for i in range(test_y.shape[0])])
    conf_matrix_full = confusion_matrix(y_true=test_y, y_pred=predictions_full)

    # >>> conf_matrix_full
    # >>> [[49965,    35],
    #      [   25,    68]]
"""