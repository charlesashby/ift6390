import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score as auc
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics.ranking import _binary_clf_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --------------------
# CONFUSION MATRIX
#
# [[49047,   953],
#  [   20,    73]]
#
# ---------------------


# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
#data_dir = os.getcwd()
data_dir = 'C:\\Users\\Charles\\PycharmProjects\\fraud_detection\\'


with open(data_dir + 'data\\data_set.pickle', 'rb') as handle:
    train_x, train_y, test_x, test_y = pickle.load(handle)

train_y = np.array([0 if train_y[i][0] == '1' else 1 for i in range(train_y.shape[0])])
test_y = np.array([0 if test_y[i][0] == '1' else 1 for i in range(test_y.shape[0])])

# Network Parameters
n_hidden_1 = 15 # 1st layer num features
#n_hidden_2 = 15 # 2nd layer num features
n_input = train_x.shape[1] # MNIST data input (img shape: 28*28)

X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    # 'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    # 'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    # 'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    # 'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
    # biases['encoder_b2']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
    # biases['decoder_b2']))
    return layer_1


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define batch mse
batch_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1)

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# TRAIN StARTS
save_model = os.path.join(data_dir, 'temp_saved_model_1layer.ckpt')
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    now = datetime.now()
    sess.run(init)
    total_batch = int(train_x.shape[0] / batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_idx = np.random.choice(train_x.shape[0], batch_size)
            batch_xs = train_x[batch_idx]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        # Display logs per epoch step
        if epoch % display_step == 0:
            train_batch_mse = sess.run(batch_mse, feed_dict={X: train_x})
            #import pdb; pdb.set_trace()

            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c),
                  "Train auc=", "{:.6f}".format(auc(train_y, train_batch_mse)),
                  "Time elapsed=", "{}".format(datetime.now() - now))

    print("Optimization Finished!")

    save_path = saver.save(sess, save_model)
    print("Model saved in file: %s" % save_path)

save_model = os.path.join(data_dir, 'temp_saved_model_1layer.ckpt')
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # ----------------------------------------- #
    # FPS / TPS TERMINOLOGY   (A = Fraud)       #
    #                                           #
    # TPS: test says A and sample is A          #
    # FPS: test says A and sample is not A      #
    # TNS: test says not A and sample is not A  #
    # FNS: test says not A and sample is A      #
    # ----------------------------------------- #

    now = datetime.now()

    saver.restore(sess, save_model)

    test_batch_mse = sess.run(batch_mse, feed_dict={X: test_x})
    fps, tps, thresholds = _binary_clf_curve(test_y, test_batch_mse)
    fpr, tpr, threshold = roc_curve(test_y, test_batch_mse)
    print("Test auc score: {:.6f}".format(auc(test_y, test_batch_mse)))


max = 0
for i in range(thresholds.shape[0]):
    ratio = fps[i] / tps[i]
    if ratio > max:
        max = ratio



# FINDING THE THRESHOLD (TRAINING SET)

with tf.Session() as sess:

    # Build the graph and restore weights here ...

    # ----

    train_batch_mse = sess.run(batch_mse, feed_dict={X: train_x})

    # Compute false positives and true positives
    fps_train, tps_train, thresholds_train = _binary_clf_curve(train_y, train_batch_mse)

    # Plot the TPS/FPS here...

    plt.plot(fps_train, tps_train)

    # Find the threshold associated to the point (4394, 330)
    t_idx = np.where(tps_train == 330)[0][0]
    optimal_threshold = thresholds_train[t_idx]

    # Compute confusion matrix on the test set
    test_mse = sess.run(batch_mse, feed_dict={X: test_x})
    pred = np.int64(test_mse > optimal_threshold)
    confusion_matrix = confusion_matrix(y_true=test_y, y_pred=pred)