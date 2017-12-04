import tensorflow as tf


def MLP(input_, out_dim, size=128, scope=None):
    ''' MLP Implementation '''
    assert len(input_.shape) == 2, "MLP takes input of dimension 2 only"

    with tf.variable_scope(scope or "MLP"):
        W_h = tf.get_variable("W_hidden", [input_.get_shape()[1], size], dtype='float32')
        b_h = tf.get_variable("b_hidden", [size], dtype='float32')
        W_out = tf.get_variable("W_out", [size, out_dim], dtype='float32')
        b_out = tf.get_variable("b_out", [out_dim], dtype='float32')

    h = tf.nn.relu(tf.matmul(input_, W_h) + b_h)
    out = tf.matmul(h, W_out) + b_out
    return out



def ResBlock(input_, out_dim, size=128, scope=None):
    ''' Residual Block Implementation '''

    with tf.variable_scope(scope or "MLP"):
        W_h = tf.get_variable("W_hidden", [input_.get_shape()[1], size], dtype='float32')
        b_h = tf.get_variable("b_hidden", [size], dtype='float32')
        W_h_res = tf.get_variable("W_hidden_res", [input_.get_shape()[1], size], dtype='float32')
        b_h_res = tf.get_variable("b_hidden_res", [size], dtype='float32')

        W_out = tf.get_variable("W_out", [size, out_dim], dtype='float32')
        b_out = tf.get_variable("b_out", [out_dim], dtype='float32')

    h = tf.nn.relu(tf.matmul(input_, W_h) + b_h)
    h_res = tf.nn.relu(tf.matmul(input_, W_h_res) + b_h_res) + h
    out = tf.matmul(h_res, W_out) + b_out

    return out