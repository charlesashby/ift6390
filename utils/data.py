import gzip, os, codecs
import math
import numpy as np
import pickle
import random
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


def load_mnist(one_hot=True):
    """ Loads the MNIST data set and returns X_train,
        Y_train, X_valid, Y_valid, X_test, Y_test
    """
    with gzip.open('data/mnist.pkl.gz') as mnist:
        pickler = pickle._Unpickler(mnist)
        pickler.encoding = 'latin1'
        data = pickler.load()

    X_train, Y_train = data[0]
    Y_train_one_hot = np.zeros((Y_train.shape[0], Y_train.max() + 1))
    Y_train_one_hot[np.arange(Y_train.shape[0]), Y_train] = 1

    X_valid, Y_valid = data[1]
    Y_valid_one_hot = np.zeros((Y_valid.shape[0], Y_valid.max() + 1))
    Y_valid_one_hot[np.arange(Y_valid.shape[0]), Y_valid] = 1

    X_test, Y_test = data[2]
    Y_test_one_hot = np.zeros((Y_test.shape[0], Y_test.max() + 1))
    Y_test_one_hot[np.arange(Y_test.shape[0]), Y_test] = 1
    if one_hot:
        return X_train, np.array(Y_train_one_hot, dtype=np.int32), \
               X_valid, np.array(Y_valid_one_hot, dtype=np.int32), \
               X_test,  np.array(Y_test_one_hot, dtype=np.int32)
    else:
        return X_train, np.array(Y_train, dtype=np.int32), \
               X_valid, np.array(Y_valid, dtype=np.int32), \
               X_test,  np.array(Y_test, dtype=np.int32)


def load_two_moons():
    """ Loads the two moons datasets and returns X"""
    np_state = np.random.get_state()
    np.random.seed(123)
    data = np.loadtxt('data/2moons.txt')
    np.random.shuffle(data)
    np.random.set_state(np_state)

    # Splitting into train, valid, test
    nb_train = math.floor(0.8 * len(data))
    nb_valid = math.floor(0.1 * len(data))
    nb_test = len(data) - nb_train - nb_valid
    X_train, Y_train = data[:nb_train, :2], data[:nb_train, 2]
    X_valid, Y_valid = data[nb_train:nb_train + nb_valid, :2], data[nb_train:nb_train + nb_valid, 2]
    X_test, Y_test = data[-nb_test:, :2], data[-nb_test:, 2]
    return X_train, np.array(Y_train, dtype=np.int32), \
           X_valid, np.array(Y_valid, dtype=np.int32), \
           X_test,  np.array(Y_test, dtype=np.int32)


def create_fraud_detection_data_set():
    """ Create the data sets for fraud
        detection
    """
    ones = []
    zeros = []

    with open('data/creditcard.csv', 'r') as f:
        data = csv.reader(f, delimiter=',')
        for line in data:
            if line[-1] == '0':
                zeros.append(line[:-1] + [1, 0])
            elif line[-1] == '1':
                ones.append(line[:-1] + [0, 1])

    training_set = np.array(ones[:-93] + zeros[:-50000])
    test_set = np.array(ones[-93:] + zeros[-50000:])

    test_set_x = test_set[:, :-2]
    test_set_y = test_set[:, -2:]

    training_set_x = training_set[:, :-2]
    training_set_y = training_set[:, -2:]

    data_set = [training_set_x, training_set_y,
                test_set_x, test_set_y]

    with open('data/data_set_fraud.pickle', 'wb') as handle:
        pickle.dump(data_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def shuffle_datasets(valid_perc=0.05):
    """ Shuffle the dataset """

    def reshape_lines(lines):
        data = []
        for l in lines:
            split = l.split('","')
            data.append((split[0][1:], split[-1][:-2]))

        return data


    TRAIN_SET = 'data/training.1600000.processed.noemoticon.csv'
    TEST_SET = 'data/testdata.manual.2009.06.14.csv'

    assert os.path.exists(TRAIN_SET), 'Download the training set at http://help.sentiment140.com/for-students/'
    assert os.path.exists(TEST_SET), 'Download the testing set at http://help.sentiment140.com/for-students/'

    # Create training and validation set
    print('Creating training & validation set...')

    with open(TRAIN_SET, encoding = "ISO-8859-1") as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines_train = lines[:int(len(lines) * (1 - valid_perc))]
        lines_valid = lines[int(len(lines) * (1 - valid_perc)):]

    valid = reshape_lines(lines_valid)
    train = reshape_lines(lines_train)

    print('Creating testing set...')

    with open(TEST_SET, encoding="ISO-8859-1") as f:
        lines = f.readlines()
        random.shuffle(lines)
        test = reshape_lines(lines)
    print('All datasets have been created!')
    data_sets = [train, valid, test]
    all_words = []

    # Creating the dictionary
    for line in train:
        words = line[1]
        word_tokens = word_tokenize(words)
        for word in word_tokens:
            w = word.lower()
            if w not in stopwords.words('english'):
                all_words.append(w)

    words = nltk.FreqDist(all_words)
    dict = words.most_common(20000)
    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(dict, f)


def load_to_ram(data):
    """
    Load some data to RAM and convert
    sentences to one-hot vectors

    :param data:
        the data set you want
        to load
        (list of (target, sentence)
         tuple)
    :return:
        features, targets
    """
    X = []
    Y = []

    with open('utils/dictionary.pkl', 'rb') as f:
        word_features = pickle.load(f)[:3000]

    for i, line in enumerate(data):
        if i % 10000 == 0:
            print(i)
        data = line[1]
        Y.append(int(line[0]))
        zeros = np.zeros(shape=3001)
        for w in word_tokenize(data):
            if w in word_features:
                zeros[word_features.index(w)] += 1
            else:
                zeros[-1] += 1
            X.append(zeros)

    return X, Y


def svm_classifier_worker(X_train, Y_train, i, q, c=0.1):
    """ Worker method for training SVMs on
        the sentiment analysis data set
        
    :param q:
        Queue for multiprocessing
    :param i:
        i-th worker
    :param X_train, Y_train:
        Data
    """

    classifier = svm.SVC(C=c, kernel='linear')
    classifier.fit(X_train, Y_train)
    #print('picking %i' % i)
    #svm_sentiment_analysis_pkl = open('pickled_models/svm_sentiment_analysis_%i.pkl' % i, 'wb')
    #pickle.dump(classifier, svm_sentiment_analysis_pkl)
    #print('done')
    q.put(classifier)























