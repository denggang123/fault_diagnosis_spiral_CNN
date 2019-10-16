# -*- encoding: utf-8 -*-
"""
@Time    : 2019/8/6 13:52
@Author  : gang.deng01
"""
import tensorflow as tf
import numpy as np
from random import choice
import os


def text2vec(text):
    vector = np.zeros(1 * 10)

    def char2pos(c):     # ord()它以一个字符（长度为1的字符串）作为参数，返回对应的ASCII数值，或者Unicode数值
        k = ord(str(c))
        if 48 <= k <= 57:    # ord("0")的值为48
            return k - 48
    idx = char2pos(text)
    vector[idx] = 1
    return vector


def normalization_0_1(data):
    d_max = data.max()
    d_min = data.min()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j] /= (d_max - d_min)
    return data


def get_one_image(root_dir, f):
    data = np.loadtxt(os.path.join(root_dir, f))
    data = np.expand_dims(normalization_0_1(data), axis=2)  # data.shape = (31,31,1)
    label = text2vec(f[-5])
    return data, label


def get_image_files(root_dir):
    """返回包含路径内所有文件名的一个列表，列表的元素只包含文件名而不含路径"""
    img_list = list()
    files = os.listdir(root_dir)
    for f in files:
        if os.path.isfile(os.path.join(root_dir, f)):
            img_list.append(f)
    return img_list

w = 31
h = 31
label_vector_size = 10
train_dir = "./processed_samples/for_train/"
test_dir = "./processed_samples/for_test/"
train_files = get_image_files(train_dir)
test_files = get_image_files(test_dir)
# for i in range(20):
#     print(train_files[i][-5])

# 占位符
x_image = tf.placeholder(shape=[None, h, w, 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, label_vector_size], dtype=tf.float32)
keep_prob = tf.placeholder(dtype=tf.float32)

# convolution layer 1
conv1_w = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.random_normal(shape=[32], stddev=0.1))
conv1_out = tf.nn.conv2d(input=x_image, filter=conv1_w, strides=[1, 1, 1, 1], padding='SAME')
conv1_relu = tf.nn.relu(tf.add(conv1_out, conv1_bias))

# max pooling 1
maxpooling_1 = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# convolution layer 2
conv2_w = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.random_normal(shape=[64], stddev=0.1))
conv2_out = tf.nn.conv2d(input=maxpooling_1, filter=conv2_w, strides=[1, 1, 1, 1], padding='SAME')
conv2_relu = tf.nn.relu(tf.add(conv2_out, conv2_bias))

# max pooling 2
maxpooling_2 = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# convolution layer 3
conv3_w = tf.Variable(tf.random_normal(shape=[3, 3, 64, 64], stddev=0.1, dtype=tf.float32))
conv3_bias = tf.Variable(tf.random_normal(shape=[64], stddev=0.1))
conv3_out = tf.nn.conv2d(input=maxpooling_2, filter=conv3_w, strides=[1, 1, 1, 1], padding='SAME')
conv3_relu = tf.nn.relu(tf.add(conv3_out, conv3_bias))

# max pooling 3
maxpooling_3 = tf.nn.max_pool(conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# fc-1
w_fc1 = tf.Variable(tf.random_normal(shape=[4*4*64, 200], stddev=0.1, dtype=tf.float32))
# w_fc1 = tf.Variable(xavier_init(4*4*64, 200))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[200]))
h_pool2 = tf.reshape(maxpooling_3, [-1, 4*4*64])
output_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2, w_fc1), b_fc1))

# dropout
h2 = tf.nn.dropout(output_fc1, keep_prob=keep_prob)

# fc-2
w_fc2 = tf.Variable(tf.random_normal(shape=[200, 10], stddev=0.1, dtype=tf.float32))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.add(tf.matmul(output_fc1, w_fc2), b_fc2)

# loss
cross_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y)
loss = tf.reduce_mean(cross_loss)
step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# accuracy
predict = tf.reshape(y_conv, [-1, 1, 10])
max_idx_p = tf.argmax(predict, 2)
max_idx_l = tf.argmax(tf.reshape(y, [-1, 1, 10]), 2)
correct_pred = tf.equal(max_idx_p, max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_train_batch(files, batch_size):
    images = []
    labels = []
    for f in range(batch_size):
        image, label = get_one_image(train_dir, choice(files))
        images.append(image)
        labels.append(label)
    return images, labels


def get_batch(root_dir, files):
    images = []
    labels = []
    for f in files:
        image, label = get_one_image(root_dir, f)
        images.append(image)
        labels.append(label)
    return images, labels


test_images, test_labels = get_batch(test_dir, test_files)
# print(np.array(test_images[2]).shape)
# saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = list()
    acc_list = list()
    for i in range(10000):
        batch_xs, batch_ys = get_train_batch(train_files, 100)
        curr_loss, curr_ = sess.run([loss, step], feed_dict={x_image: batch_xs, y: batch_ys, keep_prob: 0.5})
        if (i + 1) % 100 == 0:
            print("run step (%d) ..., loss : (%f)" % (i+1, curr_loss))
            loss_list.append(curr_loss)
            curr_acc = sess.run(accuracy, feed_dict={x_image: test_images, y: test_labels, keep_prob: 1.0})
            print("current test Accuracy : %f" % (curr_acc))
            acc_list.append(curr_acc)
    np.savetxt("loss_list.txt",loss_list)
    np.savetxt("acc_list.txt", acc_list)

    # saver.save(sess, "./code_break.ckpt", global_step=10000)
