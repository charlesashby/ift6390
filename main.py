from models.bayes import *
from models.neural_net import *
from models.svm import *
from utils.data import *
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve


if __name__ == '__main__':

    # MNIST Start ---------------------------------------------------------
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_mnist(one_hot=False)

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

    # Feed-Forward Neural Network for MNIST
    nn = FeedForwardNN([X_train, Y_train, X_valid, Y_valid, X_test, Y_test])
    nn.build()
    nn.train()
    nn.compute_confusion_matrix()

    # >>> accuracy: 0.9800

    # Support Vector Classifier for MNIST
    # WARNING: Do not start this unless you have a lot of RAM

    param_C = 5
    param_gamma = 0.05
    classifier = svm.SVC(C=param_C,gamma=param_gamma)

    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    classifier.fit(X_train, Y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    # Class prediction on the test set
    expected = Y_test
    predicted = classifier.predict(X_test)
    cm = metrics.confusion_matrix(expected, predicted)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % cm)
    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

    with open('log.txt', 'w') as f:
        f.write(str(metrics.accuracy_score(expected, predicted)))
        f.write(str(metrics.confusion_matrix(expected, predicted)))
        f.write(str(metrics.classification_report(expected, predicted)))

        #
        #              precision    recall  f1-score   support
        #
        #          0       0.98      0.99      0.99       980
        #          1       0.99      0.99      0.99      1135
        #          2       0.98      0.98      0.98      1032
        #          3       0.98      0.99      0.98      1010
        #          4       0.99      0.98      0.99       982
        #          5       0.99      0.98      0.98       892
        #          6       0.99      0.99      0.99       958
        #          7       0.98      0.98      0.98      1028
        #          8       0.97      0.98      0.98       974
        #          9       0.98      0.96      0.97      1009
        #
        #         avg      0.98      0.98      0.98     10000
        #
        #
        #    Confusion matrix:
        #    [[ 974    0    1    0    0    1    1    1    2    0]
        #     [   0 1128    3    1    0    1    0    1    1    0]
        #     [   4    0 1015    1    1    0    0    6    5    0]
        #     [   0    0    1  996    0    4    0    5    4    0]
        #     [   0    1    3    0  965    0    4    0    2    7]
        #     [   2    0    1    7    1  872    3    1    4    1]
        #     [   5    2    0    0    2    3  945    0    1    0]
        #     [   0    3    9    1    1    0    0 1004    2    8]
        #     [   2    0    1    6    1    2    0    2  958    2]
        #     [   4    4    2    8    6    2    0    6    6  971]]
        #
        #    Accuracy=0.9828

    # MNIST End ---------------------------------------------------------

    # Fraud Detection Start ---------------------------------------------

    # Under sample data sets for fraud detection
    # to solve the class imbalance problem

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

    # Support Vector Machine (linear kernel)

    model = SVM(max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001)
    model.fit(trunc_train_x, trunc_train_y)
    predictions = np.int64(model.predict(trunc_test_x) >= 0)
    conf_matrix = confusion_matrix(y_true=trunc_test_y, y_pred=predictions)

    # On the under-sampled test set
    # >>>  conf_matrix
    # >>>  [[92,  1],
    #       [25, 68]]

    # on the complete data set
    predictions_full = np.int64(model.predict(test_x) >= 0)
    test_y = np.array([1 if test_y[i][0] == '0' else 0 for i in range(test_y.shape[0])])
    conf_matrix_full = confusion_matrix(y_true=test_y, y_pred=predictions_full)

    # On the full test set
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

    # Bayes Classifier
    gaussian_estimator = DiagonalGaussian(nb_dims=trunc_train_x.shape[1])
    gaussian_estimator.train(trunc_train_x)

    bayes_net = BayesClassifier(gaussian_estimator)
    bayes_net.train(training_data=trunc_train_x, training_labels=trunc_train_y)
    predicted_classes = bayes_net.compute_predictions(trunc_test_x)
    correct_classes = trunc_test_y
    error_rate = 1. - np.mean(predicted_classes == correct_classes)

    # Results for Bayes Classifier goes here...

    # Fraud Detection End ------------------------------------------------------------

    # Sentiment Analysis Start -------------------------------------------------------

    # WARNING: Do not run this unless you Have a lot of RAM

    with open('sentiment_analysis_data_sets.pkl', 'rb') as f:
        train, valid, test = pickle.load(f)

    X_train, Y_train = load_to_ram(train)

    # Support Vector Classifier

    param_C = 5
    param_gamma = 0.05
    classifier = svm.SVC(C=param_C, gamma=param_gamma)
    classifier.fit(X_train, Y_train)

    X_test, Y_test = load_to_ram(test)
    expected_classes = Y_test
    predicted_classes = classifier.predict(X_test)
    cm = metrics.confusion_matrix(expected_classes, predicted_classes)

    # Waiting on results... Best accuracy so far: batch_size: 25,000 12 processes
    # accuracy: 0.713375
    # 
    #   [[20939 19101]
    #    [ 3829 36131]]             
    # 
    #            precision    recall  f1-score   support
    # 
    #       0       0.85      0.52      0.65     40040
    #       1       0.65      0.90      0.76     39960
    # 
    #     avg       0.75      0.71      0.70     80000

    # Bayes Classifier

    param_C = 5
    param_gamma = 0.05
    classifier = svm.SVC(C=param_C, gamma=param_gamma)
    classifier.fit(X_train, Y_train)
    expected = Y_test
    predicted = classifier.predict(X_test)
    cm = metrics.confusion_matrix(expected, predicted)

    # accuracy: 0.7073875

    # Neural Net

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_valid = np.array(load_to_ram(valid))
    Y_valid = np.array(load_to_ram(valid))
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    neural_net = FeedForwardNN([X_train, Y_train, X_valid, Y_valid, X_test, Y_test])
    neural_net.build()
    neural_net.train()
    tt = neural_net.compute_confusion_matrix()

    # accuracy: 0.78036249

    # Sentiment Analysis End ---------------------------------------------------------


    
