import tensorflow as tf
import numpy as np
import csv
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

labels_array = []
testing_labels_array = []

with open('breast-cancer-wisconsin-training-labels.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter='\n')

    for row in readCSV:
      labels_array.append(row)

with open('breast-cancer-wisconsin-testing-labels.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter='\n')

    for row in readCSV:
        testing_labels_array.append(row)

filename = 'breast-cancer-wisconsin-training-data.csv'
filename2 = 'breast-cancer-wisconsin-testing-features.csv'
data = np.genfromtxt(filename, delimiter=",").astype(np.float32)
testing_features = np.genfromtxt(filename2, delimiter=",").astype(np.float32)

data_tf = tf.convert_to_tensor(data, np.float32)
testing_tf = tf.convert_to_tensor(testing_features, np.float32)

labels_array_numpy = np.array(labels_array)
testing_labels_array_numpy = np.array(testing_labels_array)

data2 = labels_array_numpy
values = array(data2)

n_nodes_hl1 = 90
n_nodes_hl2 = 90
n_nodes_hl3 = 90

n_classes = 1
batch_size = 10
num_input = 9
learning_rate = 0.000009

x = tf.placeholder('float32', shape=[None, num_input], name='x')
y = tf.placeholder('float', shape=[None, n_classes], name='y')

TRAIN_DATASIZE, _ = data.shape
PERIOD = int(len(data) / batch_size)

def neural_network_model(data_tf):

    weight1 = tf.Variable(tf.random_normal([num_input, n_nodes_hl1]), name='w1')
    bias1 = tf.Variable(tf.random_normal([n_nodes_hl1]), name='b1')

    weight2 = tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name='w2')
    bias2 = tf.Variable(tf.random_normal([n_nodes_hl2]), name='b2')

    weight3 = tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]), name='w3')
    bias3 = tf.Variable(tf.random_normal([n_nodes_hl3]), name='b3')

    output_weight = tf.Variable(tf.random_normal([n_nodes_hl1, n_classes]), name='output_w')
    output_bias = tf.Variable(tf.random_normal([n_classes]), name='output_b')

    l1 = tf.add(tf.matmul(data_tf, weight1), bias1)
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, weight2), bias2)
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, weight3), bias3)
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_weight), output_bias)

    return output

def next_batch(num, data, labels):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

global save_path

def train_neural_network(x):

    with tf.name_scope("model"):
        prediction = neural_network_model(x)

    with tf.name_scope("cost"):
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    tf.summary.scalar("loss", loss_op)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

    hm_epochs = 40

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(PERIOD):
                epoch_x, epoch_y = next_batch(batch_size, data, data2)
                _, c = sess.run([optimizer, loss_op], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.greater(tf.nn.sigmoid(prediction), [0.5]), tf.cast(y, 'bool'))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name="accuracy")
        print('Accuracy:', accuracy.eval({x: testing_features, y: testing_labels_array_numpy}))

        save_path = saver.save(sess, "Desktop/breast_cancer_neural_net/model.ckpt")
        print("Model saved in path: %s" % save_path)

train_neural_network(x)
'''
tf.reset_default_graph()

weight1 = tf.Variable(tf.zeros([9,90]), name='model/w1')
weight2 = tf.Variable(tf.zeros([90,90]), name='model/w2')
weight3 = tf.Variable(tf.zeros([90,90]), name='model/w3')
output_weight = tf.Variable(tf.zeros([90,1]), name='model/output_w')

bias1 = tf.Variable(tf.zeros([90]), name='model/b1')
bias2 = tf.Variable(tf.zeros([90]), name='model/b2')
bias3 = tf.Variable(tf.zeros([90]), name='model/b3')
output_bias = tf.Variable(tf.zeros([1]), name='model/output_b')

saver = tf.train.Saver()

#print_tensors_in_checkpoint_file(file_name='Desktop/breast_cancer_neural_net/model.ckpt', tensor_name='',
#                                 all_tensors=False)

with tf.Session() as sess:

    saver.restore(sess, "Desktop/breast_cancer_neural_net/model.ckpt")
    print("Model restored.")

    def saved_neural_network_model(x):

        l1 = tf.add(tf.matmul(x, weight1), bias1)
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, weight2), bias2)
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, weight3), bias3)
        l3 = tf.nn.relu(l3)

        output = tf.add(tf.matmul(l3, output_weight), output_bias)

        output_sigmoid = tf.sigmoid(output)

        return output_sigmoid

    print(sess.run(weight1))

    neural_net_test = (saved_neural_network_model(testing_features))
    predicted_value = (sess.run(neural_net_test))
    print(predicted_value)

'''