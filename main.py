from models.bayes import *
from models.neural_net import *
from models.svm import *
from utils.data import *
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve


if __name__ == '__main__':
    """
    # classification on MNIST
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_mnist()
    
    # Bayes Classifier
    gaussian_estimator = DiagonalGaussian(nb_dims=X_train.shape[1])
    gaussian_estimator.train(X_train)

    bayes_net = BayesClassifier(gaussian_estimator)
    bayes_net.train(training_data=X_train, training_labels=Y_train)
    predicted_classes = bayes_net.compute_predictions(X_test)
    correct_classes = Y_test
    error_rate = 1. - np.mean(predicted_classes == correct_classes)
    
    # >>> error_rate
    # >>> 0.183
    


    # Feed-Forward Neural Network
    nn = FeedForwardNN([X_train, Y_train, X_valid, Y_valid, X_test, Y_test])
    nn.build()
    nn.train()
    nn.compute_confusion_matrix()
    # >>> accuracy: 0.9800
    
 
    # SVM Classifier (linear kernel)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_mnist(one_hot=False)
    classification = svm.SVC(C=5.0, gamma=0.05, verbose=True)
    train_scores, valid_scores = validation_curve(classification)
    # classification.fit(X_train, Y_train)
    """
    # -----------------
    # FRAUD DETECTION
    # -----------------

    # Under sample data sets
    with open("data/data_set.pickle", 'rb') as handle:
        train_x, train_y, test_x, test_y = pickle.load(handle)

        # CREATE TRAINING & VALIDATION SET

        idx_fraud_train = np.where(train_y[:, 0] == '0')[0]
        idx_non_fraud_train = np.where(train_y[:, 0] == '1')[0][:len(idx_fraud_train)]

        idx_fraud_valid = np.where(train_y[:, 0] == '0')[0][:50]
        idx_non_fraud_valid = np.where(train_y[:, 0] == '1')[0][:50]

        idx_fraud_test = np.where(test_y[:, 0] == '0')[0]
        idx_non_fraud_test = np.where(test_y[:, 0] == '1')[0][:len(idx_fraud_test)]

        under_sampled_idx_valid = np.concatenate((idx_non_fraud_valid, idx_fraud_valid))
        under_sampled_idx_train = np.concatenate((idx_non_fraud_train, idx_fraud_train))
        under_sampled_idx_test = np.concatenate((idx_non_fraud_test, idx_fraud_test))

        # Shuffle the indexes
        np.random.shuffle(under_sampled_idx_train)
        np.random.shuffle(under_sampled_idx_test)
        np.random.shuffle(under_sampled_idx_valid)

        trunc_train_x, trunc_train_y = train_x[under_sampled_idx_train, :], \
                                       train_y[under_sampled_idx_train, :].astype('float32')
        trunc_train_y = np.array([1 if trunc_train_y[i][0] == 0 else 0 for i in range(trunc_train_y.shape[0])])

        trunc_valid_x, trunc_valid_y = train_x[under_sampled_idx_valid, :], \
             train_y[under_sampled_idx_valid, :].astype('float32')

        trunc_test_x, trunc_test_y = test_x[under_sampled_idx_test, :], \
                                     test_y[under_sampled_idx_test, :].astype('float32')
        trunc_test_y = np.array([1 if trunc_test_y[i][0] == 0 else 0 for i in range(trunc_test_y.shape[0])])


    # SVM
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

    # Neural Network
    neural_net = FeedForwardNN([trunc_train_x, trunc_train_y, trunc_valid_x,
                               trunc_valid_y, trunc_test_x, trunc_test_y])
    neural_net.build()
    neural_net.train()
    neural_net.compute_confusion_matrix()

    # ---------------- #
    # CONFUSION MATRIX #
    #                  #
    # [[47499  2501]   #
    # [    9    84]]   #
    #                  #
    # ---------------- #


    # Bayes here

    