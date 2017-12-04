import numpy as np
import pickle
import tensorflow as tf
from utils.ops import MLP
import os

dir = os.getcwd()

SAVE_PATH = "checkpoints/mlp"
LOGGING_PATH = "checkpoints/log.txt"
data_dir = 'C:/Users/Charles/PycharmProjects/fraud_detection/'

# ---------------- #
# CONFUSION MATRIX #
#                  #
# [[47499  2501]   #
# [    9    84]]   #
#                  #
# ---------------- #


# ---------------------- #
# UNDERSAMPLE DATA SET   #
# ---------------------- #
"""
with open("C:\\Users\\Charles\\PycharmProjects\\fraud_detection\\data\\data_set.pickle", 'rb') as handle:
    train_x, train_y, test_x, test_y = pickle.load(handle)

    # CREATE TRAINING & VALIDATION SET

    idx_fraud_train = np.where(train_y[:, 0] == '0')[0][50:]
    idx_non_fraud_train = np.where(train_y[:, 0] == '1')[0][50:len(idx_fraud_train) + 50]

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

    trunc_valid_x, trunc_valid_y = train_x[under_sampled_idx_valid, :], \
        train_y[under_sampled_idx_valid, :].astype('float32')

    trunc_test_x, trunc_test_y = test_x[under_sampled_idx_test, :], \
        test_y[under_sampled_idx_test, :].astype('float32')

"""


class FeedForwardNN(object):
    """ Feed Forward Neural Network Implementation """

    def __init__(self, data):
        self.hparams = self.get_hparams()
        self.X_train, self.Y_train, self.X_valid, \
            self.Y_valid, self.X_test, self.Y_test = data

        self.X = tf.placeholder('float32',
                                shape=[None, self.X_train.shape[1]], name='X')
        self.Y = tf.placeholder('float32',
                                shape=[None, self.Y_train.shape[1]], name='Y')

    def build(self):
        """ Build the Network """

        mlp = MLP(self.X, out_dim=self.Y_train.shape[1], size=256, scope='mlp')
        self.pred = tf.nn.softmax(mlp)
        self.cost = - tf.reduce_sum(self.Y * tf.log(
            tf.clip_by_value(self.pred, 1e-10, 1.0)))

        labels = tf.argmax(self.Y, 1)
        predictions = tf.argmax(self.pred, 1)
        self.predictions = tf.equal(predictions, labels)
        self.confusion_matrix = tf.confusion_matrix(labels, predictions)
        self.acc = tf.reduce_mean(tf.cast(self.predictions, 'float32'))

    def train(self):
        """ Train the Network """
        BATCH_SIZE = self.hparams['BATCH_SIZE']
        EPOCHS = self.hparams['EPOCHS']
        learning_rate = self.hparams['learning_rate']
        patience = self.hparams['patience']

        cost = self.cost
        acc = self.acc

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # parameters for saving and early stopping
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            best_acc = 0.0
            DONE = False
            epoch = 0

            while epoch <= EPOCHS and not DONE:
                loss = 0.0
                batch = 1
                epoch += 1

                n_batch_train = int(self.X_train.shape[0] / BATCH_SIZE)

                for i in range(n_batch_train):
                    batch_x, batch_y = self.X_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], \
                                       self.Y_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

                    _, c, a = sess.run([optimizer, cost, acc], feed_dict={self.X: batch_x, self.Y: batch_y})

                    loss += c
                    if batch % 500 == 0:
                        # Compute Accuracy on the Training set and print some info
                        print('Epoch: %5d/%5d -- batch: %5d/%5d -- Loss: %.4f -- Train Accuracy: %.4f' %
                              (epoch, EPOCHS, batch, n_batch_train, loss/batch, a))

                        # Write loss and accuracy to some file
                        log = open(LOGGING_PATH, 'a')
                        log.write('%s, %6d, %.5f, %.5f \n' % ('train', epoch * batch, loss/batch, a))
                        log.close()

                    # --------------
                    # EARLY STOPPING
                    # --------------

                    # Compute Accuracy on the Validation set, check if validation has improved, save model, etc
                    if batch % 50 == 0:
                        accuracy = []

                        # Compute accuracy on validation set
                        a = sess.run([acc], feed_dict={self.X: self.X_valid, self.Y: self.Y_valid})
                        accuracy.append(a)
                        mean_acc = np.mean(accuracy)

                        # if accuracy has improved, save model and boost patience
                        if mean_acc > best_acc:
                            best_acc = mean_acc
                            save_path = saver.save(sess, SAVE_PATH)
                            patience = self.hparams['patience']
                            print('Model saved in file: %s' % save_path)
                            print('Epoch: %5d/%5d -- batch: %5d/%5d -- Valid Accuracy: %.4f' %
                                    (epoch, EPOCHS, batch, n_batch_train, mean_acc))
                        # else reduce patience and break loop if necessary
                        else:
                            patience -= 50
                            if patience <= 0:
                                DONE = True
                                break



                        # Write validation accuracy to log file
                        log = open(LOGGING_PATH, 'a')
                        log.write('%s, %6d, %.5f \n' % ('valid', epoch * batch, mean_acc))
                        log.close()

                    batch += 1

    def compute_confusion_matrix(self):

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, SAVE_PATH)
            pred = self.predictions

            p, a, confusion_matrix = sess.run([pred, self.acc, self.confusion_matrix],
                                           feed_dict={self.X: self.X_test, self.Y: self.Y_test})
            print('accuracy: %.4f' % a)

            print(confusion_matrix)
            return confusion_matrix

    def get_hparams(self):
        """ Get Hyper-Parameters """
        return {
            'BATCH_SIZE':       64,
            'EPOCHS':           500,
            'learning_rate':    0.005,
            'patience':         50000
        }




