from models.bayes import *
from models.neural_net import *
from models.svm import *
from utils.data import *
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
 from sklearn.naive_bayes import GaussianNB

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
    # accuracy for hp [64, 128, 256, 512]: [0.973, 0.9782, 0.98, ]
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
    # precision: 0.66019
    # recall: 0.731182

    # Support Vector Machine (RBF kernel)
    model = SVM(max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001)
    model.fit(trunc_train_x, trunc_train_y)
    predictions = np.int64(model.predict(trunc_test_x) >= 0)
    conf_matrix = confusion_matrix(y_true=trunc_test_y, y_pred=predictions)
    
    # quadratic kernel results
    # [[48956,  1044],
    #  [   17,    76]]
    # precision: 0.06785
    # recall: 0.81720
    
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
    # precision: 0.032495
    # recall: 0.90322

    # Bayes Classifier
   
    gnb = GaussianNB()
    y_pred = gnb.fit(trunc_train_x, trunc_train_y).predict(test_x)

    # confusion matrix
    # [[48631,  1369],
    #  [   15,    78]]
    # recall: 0.83870
    # precision: 0.05390
    
    # Fraud Detection End ------------------------------------------------------------

    # Sentiment Analysis Start -------------------------------------------------------

    # WARNING: Do not run this unless you Have a lot of RAM

    with open('sentiment_analysis_data_sets.pkl', 'rb') as f:
        train, valid, test = pickle.load(f)

    X_train, Y_train = load_to_ram(train)

    # Support Vector Classifier - The data set is too big for only one support vector machine,
    # We split it into smaller batch (in our case 35k samples) and run multiple SVMs in 
    # parallel (here we run 12 on a Xeon E5-2643 v2 @ 3.50 Ghz) - We also perform grid search
    # on the parameter c (best value was 0.1)

    param_C = 5
    param_gamma = 0.05
    classifier = svm.SVC(C=param_C, gamma=param_gamma)
    classifier.fit(X_train, Y_train)
    q = queue.Queue()
    parallel_size = int(multiprocessing.cpu_count() / 2)
    workers = []
    for c in [0.05, 0.1, 1, 5]:
        for i in range(parallel_size):
            workers.append(threading.Thread(target=svm_classifier_worker,
                                            args=(X_train[i * 35000: (i+1) * 35000],
                                                  Y_train[i * 35000: (i+1) * 35000],
                                                  i, q, c=c)))
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
    
    print("Training finished")
    
    classifiers = []
    while not q.empty():
        classifiers.append(q.get())
       
    X_test, Y_test = load_to_ram(test)
    predictions = [classifier.predict(X_test) for classifier in classifiers]
    predictions = np.array(predictions)
    pred = np.argmax(predictions, axis=0)
    predicted_classes = [int(predictions[pred[i]][i]) for i in range(pred.shape[0])]
    expected_classes = Y_test
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


    
