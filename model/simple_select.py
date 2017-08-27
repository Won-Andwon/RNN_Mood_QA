"""
简单型挑选程序
"""
import os
import math
import random
import time

import tensorflow as tf
import numpy as np

import data_utils

from six.moves import xrange


class CNNSelect(object):
    def __init__(self,
                 neo_size=512,
                 input_size=200,
                 vocabulary_size=40000,
                 save_path=None):
        if save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path)

        self.global_step = tf.Variable(0, trainable=False)
        
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="sentence_inputs")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name="next_word_id")
        
        w1 = tf.Variable(tf.random_normal([200, neo_size], stddev=0.1), name="weights-1")
        b1 = tf.Variable(tf.random_normal([neo_size], stddev=0.1), name="bias-1")
        
        out1 = tf.matmul(self.inputs, w1) + b1
        # print(out1)
        w2 = tf.Variable(tf.random_normal([neo_size, vocabulary_size], stddev=0.1), name="weights-2")
        b2 = tf.Variable(tf.random_normal([vocabulary_size], stddev=0.1), name="bias-2")

        self.fc_out = tf.matmul(out1, w2) + b2
        self.softmax_res = tf.nn.softmax(self.fc_out)
        # print(self.fc_out)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.fc_out),
            name="loss")

        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(
            self.loss, name="train", global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())
    
    def get_batch(self, pre, id_list, id2vec_set, vocab_size, pos_vec):
        sen_size = len(id_list)
        if sen_size < 2:
            raise ValueError("太短了。")
        if sen_size > 70 or len(pre) > 70:
            raise ValueError("太长了。")
        previous = np.zeros([200], dtype=np.float32)
        pos = 0
        for x in pre:
            previous += np.array(id2vec_set[x], dtype=np.float32)
            previous += np.array(pos_vec[pos], dtype=np.float32)
            pos += 1
            
        previous += np.array(id2vec_set[data_utils.GO_ID], dtype=np.float32)
        previous += np.array(pos_vec[pos], dtype=np.float32)
        pos = 0
        
        inputs, targets = [], []
        for ids in id_list:
            # print(previous)
            # print(em)
            em = np.array(previous)
            inputs.append(em)
            
            targets.append(ids)
            previous += np.array(id2vec_set[ids], dtype=np.float32)
            previous += np.array(pos_vec[pos], dtype=np.float32)
            pos += 1
        
        return np.array(inputs), np.array(targets)
        
    def train(self, sess, inputs, targets):
        feed = {
            self.inputs: inputs,
            self.labels: targets
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss
    
    def test(self, sess, inputs):
        res = sess.run([self.fc_out], feed_dict={self.inputs: inputs})
        
        for i in res[0]:
            selected = (-i).argsort()[0: 30]
            print(selected)


def build_position_vec(sen_size, em_size):
    pos_list = []
    for pos in range(sen_size):
        pos_vec = []
        for i in range(em_size // 2):
            if (2 * i) < em_size:
                pos_vec.append(
                    math.sin(
                        (pos + 1) / (math.pow(10000, (2 * i / em_size)))) / 100)
            if (2 * i + 1) < em_size:
                pos_vec.append(
                    math.cos(
                        (pos + 1) / (math.pow(10000, (2 * i / em_size)))) / 100)
        pos_list.append(pos_vec)
    return pos_list


def train_simple_module():
    from_train_id = r"D:\Data\dia\train_record_Q.txt.zh.vcb.ids40000"
    to_train_id = r"D:\Data\dia\train_record_A.txt.zh.vcb.ids40000"
    vector_path = r"D:\Data\dia\vocabulary_vector40000.txt"
    data_set = data_utils.read_data_pure(from_train_id, to_train_id)
    word2id, id2word = data_utils.initialize_vocabulary(r'D:\Data\dia\vocabulary40000.txt')
    vocabulary_size = min(len(id2word), 40000)
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    pos_vec_list = build_position_vec(71, 200)
    rnn_dir = os.path.join(r'D:\Data\dia', "c_dnn")
    rnn_model = os.path.join(rnn_dir, "c_dnn.ckpt")
    with tf.Session() as sess:
        model = CNNSelect(vocabulary_size=vocabulary_size, save_path=rnn_dir)

        ckpt = tf.train.get_checkpoint_state(rnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        
        start_time = time.time()
        loss = 0
        for step in xrange(200000):
            pre, id_list = random.choice(data_set)
            while len(pre) < 5 or len(pre) > 70 or len(id_list) < 5 or len(id_list) > 70:
                pre, id_list = random.choice(data_set)
            inputs, targets = model.get_batch(pre, id_list, vec_set, vocabulary_size, pos_vec_list)
            step_loss = model.train(sess, inputs, targets)
            loss += step_loss / 2000
            
            if step % 2000 == 0:
                model.saver.save(sess, rnn_model, global_step=model.global_step)
                consume = time.time() - start_time
                print("已%d步，每步耗时%0.2f, 当前平均loss：%0.4f" % (step, consume/2000, loss))
                start_time = time.time()
                loss = 0
 

if __name__ == "__main__":
    train_simple_module()
    from_train_id = r"D:\Data\dia\train_record_Q.txt.zh.vcb.ids40000"
    to_train_id = r"D:\Data\dia\train_record_A.txt.zh.vcb.ids40000"
    vector_path = r"D:\Data\dia\vocabulary_vector40000.txt"
    data_set = data_utils.read_data_pure(from_train_id, to_train_id)
    word2id, id2word = data_utils.initialize_vocabulary(r'D:\Data\dia\vocabulary40000.txt')
    vocabulary_size = min(len(id2word), 40000)
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    pos_vec_list = build_position_vec(71, 200)
    rnn_dir = os.path.join(r'D:\Data\dia', "c_dnn")
    rnn_model = os.path.join(rnn_dir, "c_dnn.ckpt")
    with tf.Session() as sess:
        model = CNNSelect(vocabulary_size=vocabulary_size, save_path=rnn_dir)

        ckpt = tf.train.get_checkpoint_state(rnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        pre, id_list = random.choice(data_set)
        while len(pre) < 5 or len(pre) > 70 or len(id_list) < 5 or len(id_list) > 70:
            pre, id_list = random.choice(data_set)
        print(pre)
        print(id_list)
        inputs, targets = model.get_batch(pre, id_list, vec_set, vocabulary_size, pos_vec_list)
        # print(inputs)
        # print(targets)
        model.test(sess, inputs)
