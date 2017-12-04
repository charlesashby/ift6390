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


def create_data_set_sentiment():
    with open('word_features.pkl', 'rb') as f:
        word_features = pickle.load(f)

    with open('data_sets.pkl', 'r') as f:
        train, valid, test = pickle.load(f)
    X_train = []
    Y_train = []
    for i, line in enumerate(train):
        if i % 10000 == 0:
            print(i)
        data = line[1]
        Y_train.append(int(line[0]))
        zeros = np.zeros(shape=3001)
        for w in word_tokenize(data):
            if w in word_features:
                zeros[word_features.index(w)] += 1
            else:
                zeros[-1] += 1
            X_train.append(zeros)
























