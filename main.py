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
