"""
CNN
判断哪些词出现在应对语句中
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import data_normalize
import data_utils
import math
import numpy as np
import os
import random
import tensorflow as tf
import time
from six.moves import xrange



class CNNModel(object):
    def __init__(self,
                 session=None,
                 batch_size=20,
                 sentence_length=32,
                 vocabulary_size=40000,
                 embedding_size=200,
                 drop_pro=0.4,
                 depth=3,
                 filter_size=3,
                 filter_num=(256, 256, 256),
                 saved_model=None,
                 from_train_id=None,
                 to_train_id=None,
                 save_path=None,
                 id2frq=None):
        self.session = session
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.sentence_length = sentence_length
        self.embedding_size = embedding_size
        self.drop_pro = drop_pro
        self.depth = depth
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.from_train_id = from_train_id
        self.to_train_id = to_train_id
        self.id2frq = id2frq
        self.save_path = save_path
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.build_graph()
        if saved_model is not None and os.path.exists(
                os.path.join(os.path.dirname(saved_model), 'checkpoint')):
            self.saver.restore(self.session, saved_model)

    def build_cnn_layer(self, filter_shape, input_from_last_layer):
        # print(filter_shape)
        weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="H_Weight")
        # print(weight)
        conv = tf.nn.conv2d(
            input_from_last_layer,
            weight,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # print(conv)
        bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]], name="bias"))
        # print(bias)
        # batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)
        # shift = tf.Variable(tf.zeros([5]))
        # scale = tf.Variable(tf.ones([5]))
        # epsilon = 1e-3
        # BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
        # print(relu)
        pool = tf.nn.max_pool(value=relu,
                              ksize=[1, self.filter_size, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding="SAME",
                              name="pool")
        # print(pool)
        return pool

    def new_build_cnn_layer(self, input_from_last_layer, layer_id):
        # print(input_from_last_layer)
        if layer_id == 1:
            filter_shape = [self.filter_size, 200, 256]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.01, shape=[filter_shape[2]], name="conv_Bias"))
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[2],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")
            batch_mean, batch_var = tf.nn.moments(pool, [0, 1], keep_dims=True, name="conv_Moment")
            shift = tf.Variable(tf.zeros([filter_shape[2]]))
            scale = tf.Variable(tf.zeros([filter_shape[2]]))
            BN_out = tf.nn.batch_normalization(pool,
                                               mean=batch_mean,
                                               variance=batch_var,
                                               offset=shift,
                                               scale=scale,
                                               variance_epsilon=1e-3,
                                               name="conv_BN_Norm")
            relu = tf.nn.relu(BN_out, name="conv_Relu")
            res = relu
        elif layer_id == 2:
            filter_shape = [self.filter_size, self.filter_num[0], self.filter_num[1]]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.01, shape=[filter_shape[2]], name="conv_Bias"))
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[1],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")
            batch_mean, batch_var = tf.nn.moments(pool, [0, 1], keep_dims=True, name="conv_Moment")
            shift = tf.Variable(tf.zeros([filter_shape[2]]))
            scale = tf.Variable(tf.zeros([filter_shape[2]]))
            BN_out = tf.nn.batch_normalization(pool,
                                               mean=batch_mean,
                                               variance=batch_var,
                                               offset=shift,
                                               scale=scale,
                                               variance_epsilon=1e-3,
                                               name="conv_BN_Norm")
            relu = tf.nn.relu(BN_out, name="conv_Relu")
            res = relu
        elif layer_id == 3:
            filter_shape = [self.filter_size, self.filter_num[1], self.filter_num[2]]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.01, shape=[filter_shape[2]], name="conv_Bias"))
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[2],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")
            batch_mean, batch_var = tf.nn.moments(pool, [0, 1], keep_dims=True, name="conv_Moment")
            shift = tf.Variable(tf.zeros([filter_shape[2]]))
            scale = tf.Variable(tf.zeros([filter_shape[2]]))
            BN_out = tf.nn.batch_normalization(pool,
                                               mean=batch_mean,
                                               variance=batch_var,
                                               offset=shift,
                                               scale=scale,
                                               variance_epsilon=1e-3,
                                               name="conv_BN_Norm")
            relu = tf.nn.relu(BN_out, name="conv_Relu")
            res = relu
        else:
            raise ValueError("the wrong layer id.")
        # print(conv)
        # print(pre_activation)
        # print(pool)
        # print(BN_out)
        # print(relu)
        return res

    def build_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_sentencelen_embeddingsize_channel = tf.placeholder(
            tf.float32, [None, self.sentence_length, self.embedding_size])
        self.batch_labels = tf.placeholder(tf.float32, [None, 1])

        input_from_last_layer = self.batch_sentencelen_embeddingsize_channel

        for i in xrange(self.depth):
            with tf.name_scope("cnn_block-%d" % (i + 1)):
                input_from_last_layer = self.new_build_cnn_layer(input_from_last_layer, i + 1)
                # if i == 0:
                #     filter_shape = [self.filter_size, self.embedding_size, 1, self.filter_num[0]]
                # else:
                #     filter_shape = [self.filter_size, 1, self.filter_num[i - 1], self.filter_num[i]]
                # input_from_last_layer = self.build_cnn_layer(filter_shape, input_from_last_layer)

        # 这个地方先写死了，记得改
        res_flat = tf.reshape(input_from_last_layer, [-1, 8 * 256])
        # print(res_flat)
        res_drop = tf.nn.dropout(res_flat, self.drop_pro, name="Dropout")
        # full_connect_w = tf.Variable(tf.random_normal([16*1*128, self.vocabulary_size]))
        # full_connect_b = tf.Variable(tf.random_normal([self.vocabulary_size]))

        # 数量太大 不适合
        # fc_w = []
        # fc_b = []
        # for i in xrange(self.vocabulary_size):
        #     if i % 1000 == 0:
        #         print(i)
        #     fc_w.append(tf.get_variable("fc_w%d" % i, shape=[2048], initializer=tf.random_normal_initializer()))
        #     fc_b.append(tf.get_variable("fc_b%d" % i, shape=[], initializer=tf.random_normal_initializer()))
        # for i in xrange(self.vocabulary_size):
        #     print(fc_w[i])
        #     print(fc_b[i])
        # full_connect_out = tf.nn.sigmoid(tf.matmul(res_drop, full_connect_w) + full_connect_b)
        # # print(tf.reduce_sum(tf.square(full_connect_out - self.batch_labels), 1))
        # self.loss = tf.reduce_mean(
        #     tf.reduce_sum(tf.square(full_connect_out - self.batch_labels), 1),
        #     name="Loss"
        # )
        # # print(self.loss)
        # self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        # print(tf.trainable_variables())

        self.fcw = tf.Variable(
            tf.random_normal([self.vocabulary_size, 8 * 256], stddev=0.01, name="all_weight"), trainable=False)
        self.fcb = tf.Variable(
            tf.random_normal([self.vocabulary_size], stddev=0.01, name="all_bias"), trainable=False)
        oss_amplify = tf.constant(self.id2frq, shape=[self.vocabulary_size])
        # selected = tf.constant(15, dtype=tf.int32)
        self.selected = tf.placeholder(tf.int32, [], "the_training_one")
        # print(tf.trainable_variables())
        # selected_w = tf.reshape(tf.Variable(fcw.initialized_value()[self.selected]), [16 * 1 * 128, 1])
        self.selected_w = tf.Variable(tf.zeros([8 * 256, 1]))

        # selected_b = tf.Variable(fcb.initialized_value()[self.selected])
        self.selected_b = tf.Variable(tf.zeros([1]))
        # print(selected_w, selected_b)
        self.assign_op_w = tf.assign(self.selected_w, tf.reshape(self.fcw[self.selected], [8 * 256, 1]))
        self.assign_op_b = tf.assign(self.selected_b, tf.reshape(self.fcb[self.selected], [1]))
        # connect_out = tf.nn.sigmoid(tf.matmul(res_drop, self.selected_w) + self.selected_b)
        # print(connect_out)
        connect_out = tf.matmul(res_drop, self.selected_w) + self.selected_b
        # print(connect_out)
        # print(tf.square(connect_out - self.batch_labels))
        self.pos_loss = tf.reduce_mean(
            tf.multiply(tf.square(connect_out - self.batch_labels), oss_amplify[self.selected]),
            name="pos_Loss"
        )
        self.nag_loss = tf.reduce_mean(
            tf.square(connect_out - self.batch_labels),
            name="nag_Loss"
        )
        # self.loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=connect_out, labels=self.batch_labels),
        #     name="cnn_loss")
        self.pos_train_op = tf.train.AdamOptimizer(0.1).minimize(self.pos_loss, global_step=self.global_step)
        self.nag_train_op = tf.train.AdamOptimizer(1e-3).minimize(self.nag_loss, global_step=self.global_step)
        self.assign_op_fcw = tf.scatter_update(self.fcw, self.selected, tf.reshape(self.selected_w, [8 * 256]))
        self.assign_op_fcb = tf.scatter_update(self.fcb, self.selected, tf.reshape(self.selected_b, []))

        self.connect_out_one = tf.matmul(res_flat, self.selected_w) + self.selected_b
        self.full_connect_out_all = tf.matmul(res_flat, self.fcw, transpose_b=True) + self.fcb

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables())
        self.session.run(self.init)
        # print(fcw[selected].eval())
        # print(fcw[0].eval())
        # print(selected_w.eval())
        # op = selected_w.assign(tf.zeros([16 * 1 * 128]))
        # self.session.run(op)
        # print(selected_w.eval())
        # selected_b.assign(tf.zeros([]))
        # op1 = tf.scatter_update(fcw, selected, selected_w)
        # self.session.run(op1)
        # print(fcw[0].eval())
        # print(fcw[selected].eval())

    def train(self, batch_inputs, batch_labels):
        feed_dict = {
            self.batch_sentencelen_embeddingsize_channel: batch_inputs,
            self.batch_labels: batch_labels
        }
        _, loss_val = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss_val

    def train_one(self, batch_inputs, batch_labels, selected, nag=False):
        feed_dict = {
            self.batch_sentencelen_embeddingsize_channel: batch_inputs,
            self.batch_labels: batch_labels,
            self.selected: selected
        }
        # print(self.fcw[selected].eval())
        # print(self.selected_w.eval())
        self.session.run([self.assign_op_w, self.assign_op_b],
                         feed_dict={self.selected: selected})
        # print(self.fcw[selected].eval())
        # print(self.selected_w.eval())
        if nag:
            _, loss_val = self.session.run([self.nag_train_op, self.nag_loss], feed_dict=feed_dict)
        else:
            _, loss_val = self.session.run([self.pos_train_op, self.pos_loss], feed_dict=feed_dict)
        # print(self.fcw[selected].eval())
        # print(self.selected_w.eval())
        self.session.run([self.assign_op_fcw, self.assign_op_fcb], feed_dict={self.selected: selected})
        # print(self.fcw[selected].eval())
        # print(self.selected_w.eval())
        return loss_val

    def generate_batches(self, w2v_model, data_set):
        cnn_inputs, cnn_labels = [], []
        for _ in xrange(self.batch_size):
            cnn_input, cnn_label = random.choice(data_set)
            cnn_input = w2v_model.get_embeddings(cnn_input)

            temp_cnn_label = np.zeros(40000, dtype=np.int32)
            for x in cnn_label:
                temp_cnn_label[x] = 1
            cnn_inputs.append(cnn_input)
            cnn_labels.append(temp_cnn_label)
        batch_inputs = np.array(cnn_inputs)
        batch_labels = np.array(cnn_labels)
        return batch_inputs, batch_labels

    def generate_batches_vector(self, id2vec_set, data_set,
                                which=None, set_start=0, pos=None, nag=False):
        st_size, em_size = 0, 0
        if pos is not None:
            st_size = len(pos)
            em_size = len(pos[0])

        if which is not None:
            selected = which
        else:
            selected = random.randint(0, self.vocabulary_size)

        data_start = set_start
        data_len = len(data_set)

        cnn_inputs, cnn_labels = [], []
        if nag:
            for _ in xrange(self.batch_size):
                cnn_input_id, cnn_label = random.choice(data_set)
                cnn_input = []
                t_pos = 0
                for x in cnn_input_id:
                    if pos is not None:  # and st_size == len(cnn_input_id)
                        cnn_input.append([id2vec_set[x][each] + pos[t_pos][each] for each in range(em_size)])
                    else:
                        cnn_input.append(id2vec_set[x])
                    t_pos += 1

                temp_cnn_label = np.zeros(1, dtype=np.int32)
                if selected in cnn_label:
                    temp_cnn_label[0] = 1
                cnn_inputs.append(cnn_input)
                cnn_labels.append(temp_cnn_label)
        else:
            counter = 0
            while data_start < data_len and counter < self.batch_size:
                cnn_input_id, cnn_label = data_set[data_start]
                if selected in cnn_label:
                    cnn_input = []
                    if pos is not None:
                        for i in xrange(st_size):
                            cnn_input.append([id2vec_set[cnn_input_id[i]][each] + pos[i][each] for each in range(em_size)])
                    else:
                        for x in cnn_input_id:
                            cnn_input.append(id2vec_set[x])
                    temp_cnn_label = np.ones(1, dtype=np.int32)
                    cnn_inputs.append(cnn_input)
                    cnn_labels.append(temp_cnn_label)
                    counter += 1
                data_start += 1

        batch_inputs = np.array(cnn_inputs)
        batch_labels = np.array(cnn_labels)

        if data_start >= data_len:
            data_start = -1
        return batch_inputs, batch_labels, selected, data_start

    def generator(self, sentence, word2id, id2word, id2vec_set, pos=None):
        word_id_list = data_utils.sentence_to_token_ids(sentence, word2id)
        cnn_inputs, cnn_input = [], []
        if pos is not None:
            st_size = len(pos)
            if st_size != self.sentence_length:
                raise ValueError("sentence's length error!")
            em_size = len(pos[0])
            for i in xrange(self.sentence_length):
                if i < len(word_id_list):
                    cnn_input.append([id2vec_set[word_id_list[i]][each] + pos[i][each] for each in range(em_size)])
                else:
                    cnn_input.append([id2vec_set[data_utils.PAD_ID][each] + pos[i][each] for each in range(em_size)])
        else:
            for x in word_id_list:
                cnn_input.append(id2vec_set[x])
            if len(cnn_input) >= self.sentence_length:
                cnn_input = cnn_input[:self.sentence_length]
            else:
                for i in range((self.sentence_length - len(cnn_input))):
                    cnn_input.append(id2vec_set[data_utils.PAD_ID])
        # word_id_list = data_utils.sentence_to_token_ids(sentence, word2id)
        # cnn_inputs, cnn_input = [], []
        # for x in word_id_list:
        #     cnn_input.append(id2vec_set[x])
        # if len(cnn_input) >= self.sentence_length:
        #     cnn_input = cnn_input[:self.sentence_length]
        # else:
        #     for i in range((self.sentence_length - len(cnn_input))):
        #         cnn_input.append(id2vec_set[data_utils.PAD_ID])
        cnn_inputs.append(cnn_input)
        batch_inputs = np.array(cnn_inputs)
        # self.session.run([self.assign_op_w, self.assign_op_b], feed_dict={self.selected: 80})
        # res_one = self.session.run(
        #     self.connect_out_one,
        #     feed_dict={self.batch_sentencelen_embeddingsize_channel: batch_inputs, self.selected: 80})
        res_all = self.session.run(self.full_connect_out_all,
                               feed_dict={self.batch_sentencelen_embeddingsize_channel: batch_inputs})
        # print(res_one, res_all[0, 80])
        top_k = 30
        res = (-res_all[0, :]).argsort()[0:top_k]
        p_str = "Answer to %s:" % sentence
        for k in xrange(top_k):
            if res[k] < len(id2word):
                p_str = "%s %s" % (p_str, id2word[res[k]])
        print(p_str)


def build_position_vec(sen_size, em_size):
    pos_list = []
    for pos in range(sen_size):
        pos_vec = []
        for i in range(em_size // 2):
            if (2 * i) < em_size:
                pos_vec.append(
                    math.sin(
                        (pos + 1) / (math.pow(10000, (2 * i / em_size)))))
            if (2 * i + 1) < em_size:
                pos_vec.append(
                    math.cos(
                        (pos + 1) / (math.pow(10000, (2 * i / em_size)))))
        pos_list.append(pos_vec)
    return pos_list


def train_second_module(from_train_id, to_train_id, from_dev_id, to_dev_id, vector_path):
    from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
    to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
    vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    data_set = data_utils.read_data(from_train_id, to_train_id)
    word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    vocabulary_size = len(id2word)
    id2frq = data_utils.get_vocab_frq(r"/home/chenyongchang/data/dia/vocabulary40000_frequency.txt",
                                      len(data_set))
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    pos_vec_list = build_position_vec(32, 200)
    cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "cnn")
    saved_model = os.path.join(cnn_dir, "cnn_model.ckpt")
    with tf.Session() as cnn_sess:
        cnn_model = CNNModel(session=cnn_sess,
                             batch_size=20,
                             vocabulary_size=vocabulary_size,
                             save_path=cnn_dir,
                             # saved_model=r'/home/chenyongchang/data/dia/cnn/cnn_model.ckpt',
                             id2frq=id2frq)
        step = 0
        step_time, loss = 0.0, 0.0
        start_time = time.time()
        for which_round in xrange(4):
            for which in xrange(vocabulary_size):
                # 100step nag
                for _ in xrange(100):
                    data_inputs, data_labels, selected, start = cnn_model.generate_batches_vector(
                        id2vec_set=vec_set,
                        data_set=data_set,
                        which=which,
                        pos=pos_vec_list,
                        nag=True)
                    step_loss = cnn_model.train_one(data_inputs, data_labels, selected, nag=True)
                    loss += step_loss / 200
                    step += 1
                    if step % 200 == 0:
                        step_time = (time.time() - start_time) / 200
                        print("[CNN]The %d steps: every step use %.4f time(s). average loss: %.8f "
                              "" % (step, step_time, loss))
                        cnn_model.saver.save(cnn_sess, saved_model, global_step=cnn_model.global_step)
                        # cnn_model.generator("说那么多连一开始分享的快乐都没有了",
                        #                     word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
                        # cnn_model.generator("我嗓子疼，随便吃点流食",
                        #                     word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
                        # cnn_model.generator("那我先去睡咯",
                        #                     word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
                        step_time, loss = 0.0, 0.0
                        start_time = time.time()
                # through the file pos
                start = 0
                start_step = step
                while start != -1:
                    data_inputs, data_labels, selected, start = cnn_model.generate_batches_vector(
                        id2vec_set=vec_set,
                        data_set=data_set,
                        which=which,
                        set_start=start,
                        pos=pos_vec_list)
                    if len(data_inputs) > 0:
                        step_loss = cnn_model.train_one(data_inputs, data_labels, selected)
                        loss += step_loss / 200
                        if step % 200 == 0:
                            step_time = (time.time() - start_time) / 200
                            print("[CNN]The %d round: %s is in training.All steps: %d, every step use %.4f time."
                                  "average loss: %.8f " % (which_round+1, id2word[which], step, step_time, loss))
                            cnn_model.saver.save(cnn_sess, saved_model, global_step=cnn_model.global_step)
                            step_time, loss = 0.0, 0.0
                            start_time = time.time()
                        step += 1
                    if step - start_step >= 400:
                        break
                cnn_model.generator("After training %s||说那么多连一开始分享的快乐都没有了" % id2word[which],
                                    word2id=word2id, id2word=id2word, id2vec_set=vec_set)



                    # step_time = 0.0
                    # loss = 0.0
                    # for step in xrange(50001):
                    #     start_time = time.time()
                    #     data_inputs, data_labels, selected = cnn_model.generate_batches_vector(
                    #         vec_set, data_set, 10, vocabulary_size=None, which=82)
                    #     step_loss = cnn_model.train_one(data_inputs, data_labels, selected)
                    #     step_time += (time.time() - start_time) / 200
                    #     loss += step_loss / 200
                    #     if step % 200 == 0:
                    #         print("[CNN] At %d step, the average loss is %.2f .One step time %.2f" % (step, loss, step_time))
                    #         cnn_model.saver.save(cnn_sess, saved_model, global_step=cnn_model.global_step)
                    #         step_time, loss = 0.0, 0.0
                    # with tf.Graph().as_default() as word2vec_graph:
                    #     with word2vec_graph.name_scope("Word2Vec"):
                    #         with tf.Session() as w2v_sess:
                    #             w2v_model = data_normalize.Word2VecModel(session=w2v_sess,
                    #                                   save_path='D:\Data\dia\w2v',
                    #                                   embedding_size=200,
                    #                                   dictionary=word2id,
                    #                                   reverse_dictionary=id2word,
                    #                                   vocabulary_size=vocabulary_size,
                    #                                   learning_rate=0.1,
                    #                                   window_size=2,
                    #                                   num_neg_samples=100,
                    #                                   sentence_levle=True,
                    #                                   saved_model=r'D:\Data\dia\w2v\word2vec_model.ckpt'
                    #                                   )
                    #             cnn_model = CNNModel(session=w2v_sess)
                    #             average_loss = 0
                    #             for i in xrange(10000):
                    #                 data_inputs, data_labels = cnn_model.generate_batches(w2v_model, data_set, 10)
                    #                 loss = cnn_model.train(data_inputs, data_labels)
                    #                 average_loss += loss
                    #                 if i % 200 == 0:
                    #                     if i > 0:
                    #                         average_loss /= 200
                    #                     # The average loss is an estimate of the loss over the last 2000 batches.
                    #                     print("[CNN] Average loss at step ", i, ": ", average_loss)
                    #                     # 误差 阈值 我直接设定了 可从参数设定
                    #                     if average_loss < 3:
                    #                         print("[CNN] Average loss has been low enough.")
                    #                         break
                    #                     average_loss = 0
                    #             cnn_model.saver.save(w2v_sess, r'D:\Data\dia\cnn')


if __name__ == "__main__":
    # with tf.Session() as sess:
    #     cnn_model = CNNModel(sess)
    # with tf.Session() as sess:
    #     word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    #     vocabulary_size = len(id2word)
    #     cnn_model = CNNModel(sess,
    #                          saved_model=r"/home/chenyongchang/data/dia/cnn/cnn_model.ckpt-0")
    #
    #     vec_set = data_utils.read_id2vec(r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt", vocabulary_size)
    #     cnn_model.generator("说那么多连一开始分享的快乐都没有了", word2id=word2id, id2word=id2word, id2vec_set=vec_set)
        # answer = data_utils.basic_tokenizer("我的快乐就是学到知识与经验 学会做事做人")
        # print(answer)
    train_second_module(None, None, None, None, None)
