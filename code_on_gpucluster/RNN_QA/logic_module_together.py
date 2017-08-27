"""
CNN
判断哪些词出现在应对语句中
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import data_utils
import math
import numpy as np
import os
import random
import tensorflow as tf
import time
from six.moves import xrange
from tensorflow.python.platform import gfile



class CNNModel(object):
    def __init__(self,
                 session=None,
                 batch_size=10,
                 sentence_length=32,
                 vocabulary_size=40000,
                 embedding_size=200,
                 drop_pro=0.4,
                 depth=3,
                 filter_size=3,
                 filter_num=(256, 256, 256),
                 outcome_cutoff=False,
                 buckets_capacity=(256, 512, 1024, 2048, 4096, 8192, 16384, -1),
                 buckets_sample_size=(32, 8, 8, 8, 4, 4, 2, 1),
                 buckets_cnn_depth=2,
                 buckets_cnn_neo_size=(128, 64),
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
        self.outcome_cutoff = outcome_cutoff
        self.buckets_capacity = buckets_capacity
        self.buckets_sample_size = buckets_sample_size
        self.buckets_cnn_depth = buckets_cnn_depth
        self.buckets_cnn_neo_size = buckets_cnn_neo_size
        self.save_path = save_path
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.build_graph()
        if saved_model is not None and os.path.exists(
                os.path.join(os.path.dirname(saved_model), 'checkpoint')):
            self.saver.restore(self.session, saved_model)
            print("Loading model from %s." % saved_model)

    # def build_cnn_layer(self, filter_shape, input_from_last_layer):
    #     # print(filter_shape)
    #     weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="H_Weight")
    #     # print(weight)
    #     conv = tf.nn.conv2d(
    #         input_from_last_layer,
    #         weight,
    #         strides=[1, 1, 1, 1],
    #         padding="VALID",
    #         name="conv")
    #     # print(conv)
    #     bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]], name="bias"))
    #     # print(bias)
    #     # batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)
    #     # shift = tf.Variable(tf.zeros([5]))
    #     # scale = tf.Variable(tf.ones([5]))
    #     # epsilon = 1e-3
    #     # BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)
    #     relu = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
    #     # print(relu)
    #     pool = tf.nn.max_pool(value=relu,
    #                           ksize=[1, self.filter_size, 1, 1],
    #                           strides=[1, 1, 1, 1],
    #                           padding="SAME",
    #                           name="pool")
    #     # print(pool)
    #     return pool

    # def new_build_cnn_layer(self, input_from_last_layer, blockid):
    #     # print(input_from_last_layer)
    #     if blockid == 1:
    #         filter_shape = [self.filter_size, self.embedding_size, 1, self.filter_num[0]]
    #         weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="H_Weight")
    #         # 卷积
    #         conv = tf.nn.conv2d(
    #             input_from_last_layer,
    #             weight,
    #             strides=[1, 1, 1, 1],
    #             padding="SAME",
    #             name="conv")
    #         # print(conv)
    #         bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]], name="bias"))
    #         pre_activation = tf.nn.bias_add(conv, bias)
    #         # print(pre_activation)
    #         # 池化 pooling
    #         pool1 = tf.nn.max_pool(value=pre_activation,
    #                               ksize=[1, self.filter_size, self.embedding_size, 1],
    #                               strides=[1, 1, 1, 1],
    #                               padding="VALID",
    #                               name="pool")
    #         # print(pool1)
    #         # BN归一
    #         batch_mean, batch_var = tf.nn.moments(pool1, [0, 1, 2], keep_dims=True)
    #         shift = tf.Variable(tf.zeros([filter_shape[3]]))
    #         scale = tf.Variable(tf.ones([filter_shape[3]]))
    #         BN_out = tf.nn.batch_normalization(pool1, batch_mean, batch_var, shift, scale, 1e-3,
    #                                            name="bn_norm")
    #         # print(BN_out)
    #         # relu激活
    #         relu = tf.nn.relu(BN_out, name="relu")
    #         # print(relu)
    #         res = relu
    #     elif blockid == 2:
    #         filter_shape = [self.filter_size, 1, self.filter_num[0], self.filter_num[1]]
    #         weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="H_Weight")
    #         # 卷积
    #         conv = tf.nn.conv2d(
    #             input_from_last_layer,
    #             weight,
    #             strides=[1, 1, 1, 1],
    #             padding="SAME",
    #             name="conv")
    #         # print(conv)
    #         bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]], name="bias"))
    #         pre_activation = tf.nn.bias_add(conv, bias)
    #         # print(pre_activation)
    #         # 池化 pooling
    #         pool = tf.nn.max_pool(value=pre_activation,
    #                               ksize=[1, self.filter_size, 1, 1],
    #                               strides=[1, 1, 1, 1],
    #                               padding="SAME",
    #                               name="pool")
    #         # print(pool)
    #         # BN归一
    #         batch_mean, batch_var = tf.nn.moments(pool, [0, 1, 2], keep_dims=True)
    #         shift = tf.Variable(tf.zeros([filter_shape[3]]))
    #         scale = tf.Variable(tf.ones([filter_shape[3]]))
    #         BN_out = tf.nn.batch_normalization(pool, batch_mean, batch_var, shift, scale, 1e-3,
    #                                            name="bn_norm")
    #         # print(BN_out)
    #         # relu激活
    #         relu = tf.nn.relu(BN_out, name="relu")
    #         # print(relu)
    #         res = relu
    #     else:
    #         filter_shape = [self.filter_size, 1, self.filter_num[1], self.filter_num[2]]
    #         weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="H_Weight")
    #         # 卷积
    #         conv = tf.nn.conv2d(
    #             input_from_last_layer,
    #             weight,
    #             strides=[1, 1, 1, 1],
    #             padding="SAME",
    #             name="conv")
    #         # print(conv)
    #         bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]], name="bias"))
    #         pre_activation = tf.nn.bias_add(conv, bias)
    #         # print(pre_activation)
    #         # BN归一
    #         batch_mean, batch_var = tf.nn.moments(pre_activation, [0, 1, 2], keep_dims=True)
    #         shift = tf.Variable(tf.zeros([filter_shape[3]]))
    #         scale = tf.Variable(tf.ones([filter_shape[3]]))
    #         BN_out = tf.nn.batch_normalization(pre_activation, batch_mean, batch_var, shift, scale, 1e-3,
    #                                            name="bn_norm")
    #         # print(BN_out)
    #         # relu激活
    #         relu = tf.nn.relu(BN_out, name="relu")
    #         # print(relu)
    #         # 池化 pooling
    #         pool = tf.nn.max_pool(value=relu,
    #                               ksize=[1, self.filter_size, 1, 1],
    #                               strides=[1, 2, 1, 1],
    #                               padding="VALID",
    #                               name="pool")
    #         # print(pool)
    #         res = pool
    #     return res

    def new_build_buckets_layer(self, input_from_last_layer):
        with tf.name_scope("buckets_cnn_layer_1"):
            weight1 = tf.Variable(tf.truncated_normal([4, 200, 128], stddev=0.5), name="conv_Weight")
            tf.add_to_collection("bck_vars", weight1)
            conv1 = tf.nn.conv1d(input_from_last_layer, weight1, stride=1, padding="SAME")
            bias1 = tf.Variable(tf.constant(0.1, shape=[128]), name="conv_Bias")
            tf.add_to_collection("bck_vars", bias1)
            pool1 = tf.nn.pool(tf.nn.bias_add(conv1, bias1),
                               window_shape=[3], strides=[2], pooling_type="MAX",
                               padding="SAME", name="conv_Max_Pool")
            relu1 = tf.nn.relu(pool1, name="conv_Relu")
        with tf.name_scope("buckets_cnn_layer_2"):
            weight2 = tf.Variable(tf.truncated_normal([4, 128, 64], stddev=0.5), name="conv_Weight")
            tf.add_to_collection("bck_vars", weight2)
            conv2 = tf.nn.conv1d(relu1, weight2, stride=1, padding="SAME")
            bias2 = tf.Variable(tf.constant(0.1, shape=[64]), name="conv_Bias")
            tf.add_to_collection("bck_vars", bias2)
            pool2 = tf.nn.pool(tf.nn.bias_add(conv2, bias2),
                               window_shape=[3], strides=[2], pooling_type="MAX",
                               padding="SAME", name="conv_Max_Pool")
            relu2 = tf.nn.relu(pool2, name="conv_Relu")
        bucket_flat = tf.reshape(relu2, [-1, 8 * 64])

        bucket_fc_w = tf.Variable(tf.random_normal([8 * 64, len(self.buckets_capacity)], stddev=0.5), name="bck_fc_w")
        tf.add_to_collection("bck_vars", bucket_fc_w)
        bucket_fc_b = tf.Variable(tf.random_normal([len(self.buckets_capacity)], stddev=0.5), name="bck_fc_b")
        tf.add_to_collection("bck_vars", bucket_fc_b)

        self.bucket_fc_out_with_sigmoid = tf.sigmoid(tf.matmul(bucket_flat, bucket_fc_w) + bucket_fc_b)
        self.bucket_fc_out_without_sigmoid = tf.matmul(bucket_flat, bucket_fc_w) + bucket_fc_b
        self.bucket_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.bucket_fc_out_without_sigmoid, labels=self.buckets_labels),
            axis=0,
            name="bck_loss")
        # for i in tf.get_collection("bck_vars"):
        #     print(i.name)
        self.buckets_train_op = tf.train.AdamOptimizer(1e-3).minimize(
            self.bucket_loss, var_list=tf.get_collection("bck_vars"))

    def new_build_cnn_layer(self, input_from_last_layer, layer_id):
        # print(input_from_last_layer)
        if layer_id == 1:
            filter_shape = [self.filter_size, 200, 256]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.5), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[2]]), name="conv_Bias")
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[2],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")

            # batch_mean, batch_var = tf.nn.moments(pool, [0, 1], keep_dims=True, name="conv_Moment")
            # shift = tf.Variable(tf.zeros([filter_shape[2]]), name="conv_Moment_Shift")
            # scale = tf.Variable(tf.zeros([filter_shape[2]]), name="conv_Moment_Scale")
            # BN_out = tf.nn.batch_normalization(pool,
            #                                    mean=batch_mean,
            #                                    variance=batch_var,
            #                                    offset=shift,
            #                                    scale=scale,
            #                                    variance_epsilon=1e-3,
            #                                    name="conv_BN_Norm")
            relu = tf.nn.relu(pool, name="conv_Relu")
            res = relu
            self.t_out_1 = res

        elif layer_id == 2:
            filter_shape = [self.filter_size, self.filter_num[0], self.filter_num[1]]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.5), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[2]]), name="conv_Bias")
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[1],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")
            # batch_mean, batch_var = tf.nn.moments(pool, [0, 1], keep_dims=True, name="conv_Moment")
            # shift = tf.Variable(tf.zeros([filter_shape[2]]), name="conv_Moment_Shift")
            # scale = tf.Variable(tf.zeros([filter_shape[2]]), name="conv_Moment_Scale")
            # BN_out = tf.nn.batch_normalization(pool,
            #                                    mean=batch_mean,
            #                                    variance=batch_var,
            #                                    offset=shift,
            #                                    scale=scale,
            #                                    variance_epsilon=1e-3,
            #                                    name="conv_BN_Norm")
            relu = tf.nn.relu(pool, name="conv_Relu")
            res = relu
            self.t_out_2 = res
        elif layer_id == 3:
            filter_shape = [self.filter_size, self.filter_num[1], self.filter_num[2]]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.5), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[2]]), name="conv_Bias")
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[2],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")
            # batch_mean, batch_var = tf.nn.moments(pool, [0, 1], keep_dims=True, name="conv_Moment")
            # shift = tf.Variable(tf.zeros([filter_shape[2]]), name="conv_Moment_Shift")
            # scale = tf.Variable(tf.zeros([filter_shape[2]]), name="conv_Moment_Scale")
            # BN_out = tf.nn.batch_normalization(pool,
            #                                    mean=batch_mean,
            #                                    variance=batch_var,
            #                                    offset=shift,
            #                                    scale=scale,
            #                                    variance_epsilon=1e-3,
            #                                    name="conv_BN_Norm")
            relu = tf.nn.relu(pool, name="conv_Relu")
            res = relu
            self.t_out_3 = pool
        else:
            raise ValueError("the wrong layer id.")
        tf.add_to_collection("cnn_vars", weight)
        tf.add_to_collection("cnn_vars", bias)
        # tf.add_to_collection("cnn_vars", shift)
        # tf.add_to_collection("cnn_vars", scale)
        # print(conv)
        # print(pre_activation)
        # print(pool)
        # print(BN_out)
        # print(relu)

        return res

    def build_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_sentencelen_embeddingsize_channel = tf.placeholder(
            tf.float32,
            [None, self.sentence_length, self.embedding_size])
        self.batch_labels = tf.placeholder(tf.float32,
                                           [None, self.vocabulary_size])
        self.buckets_labels = tf.placeholder(tf.float32,
                                             [None, len(self.buckets_capacity)])

        input_from_last_layer = self.batch_sentencelen_embeddingsize_channel

        self.new_build_buckets_layer(input_from_last_layer)

        for i in xrange(self.depth):
            with tf.name_scope("CNN-layer-%d" % (i + 1)):
                input_from_last_layer = self.new_build_cnn_layer(input_from_last_layer, i + 1)
        # for i in xrange(self.depth):
        #     with tf.name_scope("conv-relu-maxpool-%d" % (i + 1)):
        #         if i == 0:
        #             filter_shape = [self.filter_size, self.embedding_size, 1, self.filter_num[0]]
        #         else:
        #             filter_shape = [self.filter_size, 1, self.filter_num[i - 1], self.filter_num[i]]
        #         input_from_last_layer = self.build_cnn_layer(filter_shape, input_from_last_layer)
        # print(input_from_last_layer)
        # 这个地方先写死了，记得改
        self.t_out = input_from_last_layer
        res_flat = tf.reshape(input_from_last_layer, [-1, 8 * 256])
        # print(res_flat)
        res_drop = tf.nn.dropout(res_flat, self.drop_pro, name="Dropout")
        full_connect_w = tf.Variable(tf.random_normal([8 * 256, self.vocabulary_size], stddev=0.1), name="cnn_fc_w")
        full_connect_b = tf.Variable(tf.random_normal([self.vocabulary_size], stddev=0.1), name="cnn_fc_b")
        tf.add_to_collection("cnn_vars", full_connect_w)
        tf.add_to_collection("cnn_vars", full_connect_b)

        self.full_connect_out_without_sigmoid = tf.matmul(res_drop, full_connect_w) + full_connect_b
        self.full_connect_out_without_drop_with_sigmoid = tf.sigmoid(
            tf.matmul(res_flat, full_connect_w) + full_connect_b)
        # # print(tf.reduce_sum(tf.square(full_connect_out - self.batch_labels), 1))
        # self.loss = tf.reduce_mean(
        #     tf.reduce_sum(tf.square(self.full_connect_out - self.batch_labels), 1),
        #     name="cnn_tgth_loss"
        # )
        # oss_amplify = tf.constant(self.id2frq, shape=[self.vocabulary_size])

        # self.loss = tf.reduce_mean(
        #     tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=self.full_connect_out_without_sigmoid, labels=self.batch_labels), oss_amplify), 1),
        #     name="cnn_tgth_loss"
        # )
        # self.loss = tf.reduce_mean(
        #     tf.reduce_sum(tf.square(self.full_connect_out - self.batch_labels), 1),
        #     name="cnn_tgth_loss"
        # )
        # print(self.loss)
        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.full_connect_out_without_sigmoid, labels=self.batch_labels),
            axis=0,
            name="cnn_tgth_loss")


        # for i in tf.trainable_variables():
        #     print(i.name)
        # for i in tf.get_collection("cnn_vars"):
        #     print(i.name)
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(
            self.loss, global_step=self.global_step, var_list=tf.get_collection("cnn_vars"))

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables())
        self.session.run(self.init)

    def train(self, batch_inputs, batch_labels):
        feed_dict = {
            self.batch_sentencelen_embeddingsize_channel: batch_inputs,
            self.batch_labels: batch_labels
        }
        _, loss_val = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss_val

    def train_buckets(self, batch_inputs, batch_labels, buckets_labels):
        feed_dict = {
            self.batch_sentencelen_embeddingsize_channel: batch_inputs,
            self.batch_labels: batch_labels,
            self.buckets_labels: buckets_labels
        }
        _, loss_val, __, ___ = self.session.run([self.train_op, self.loss, self.buckets_train_op, self.bucket_loss],
                                                feed_dict=feed_dict)
        return loss_val

    def generate_batches_vector(self, id2vec_set, data_set, pos=None, id2frq=None):
        st_size, em_size = 0, 0
        if pos is not None:
            st_size = len(pos)
            em_size = len(pos[0])
            if em_size != self.embedding_size or st_size != self.sentence_length:
                raise ValueError("position vector error.")
        cnn_inputs, cnn_labels = [], []
        buckets_labels = []
        for _ in xrange(self.batch_size):
            cnn_input_id, cnn_label = random.choice(data_set)
            cnn_input = []
            for i in xrange(self.sentence_length):
                if pos is not None:
                    cnn_input.append(
                        [id2vec_set[cnn_input_id[i]][each] + pos[i][each] for each in range(em_size)])
                else:
                    cnn_input.append(id2vec_set[cnn_input_id[i]])

            temp_cnn_label = np.zeros(self.vocabulary_size, dtype=np.int32)
            temp_bucket_label = np.zeros(len(self.buckets_capacity), dtype=np.int32)
            for ids in cnn_label:
                temp_cnn_label[ids] = 1
                for i in xrange(len(self.buckets_capacity)):
                    if self.buckets_capacity[i] == -1:
                        temp_bucket_label[i] = 1
                        break
                    elif ids < self.buckets_capacity[i]:
                        temp_bucket_label[i] = 1
                        break
            cnn_inputs.append(cnn_input)
            cnn_labels.append(temp_cnn_label)
            buckets_labels.append(temp_bucket_label)

        batch_inputs = np.array(cnn_inputs)
        batch_labels = np.array(cnn_labels)
        buckets_labels = np.array(buckets_labels)

        return batch_inputs, batch_labels, buckets_labels

    def generator(self, sentence, word2id, id2word, id2vec_set, pos=None):
        word_id_list = data_utils.sentence_to_token_ids(sentence, word2id)
        # print(word_id_list)
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
                # print(id2vec_set[x])
                cnn_input.append(id2vec_set[x])
            if len(cnn_input) >= self.sentence_length:
                cnn_input = cnn_input[:self.sentence_length]
            else:
                for i in range((self.sentence_length - len(cnn_input))):
                    cnn_input.append(id2vec_set[data_utils.PAD_ID])
        # for i in cnn_input:
        #     print(i)
        # print(cnn_input[0])

        cnn_inputs.append(cnn_input)
        # print(cnn_inputs)
        batch_inputs = np.array(cnn_inputs)
        # print(batch_inputs)
        # res = self.session.run(self.t_out_1, feed_dict={self.batch_sentencelen_embeddingsize_channel: batch_inputs})
        # for each in res[0]:
        #     print(each)
        # res = self.session.run(self.t_out_2, feed_dict={self.batch_sentencelen_embeddingsize_channel: batch_inputs})
        # for each in res[0]:
        #     print(each)
        # res = self.session.run(self.t_out_3, feed_dict={self.batch_sentencelen_embeddingsize_channel: batch_inputs})
        # for each in res[0]:
        #     print(each)
        res = self.session.run(self.full_connect_out_without_drop_with_sigmoid,
                               feed_dict={self.batch_sentencelen_embeddingsize_channel: batch_inputs})
        bck = self.session.run(self.bucket_fc_out_with_sigmoid,
                               feed_dict={self.batch_sentencelen_embeddingsize_channel: batch_inputs})
        # print(res[0][word2id["做人"]])
        idlist = self.result_analysis(res[0], bucket=False,
                                      top_k=(0, 100),
                                      only_present_average=False,
                                      comprehensive_analysis=True,
                                      pulse_analysis=False)

        return idlist

    def statistic_words(self, dataset=None, id2vec_set=None, pos=None, gene_data=None, statistics_path=None):
        self.gene_data = np.zeros([self.vocabulary_size, 10], dtype=np.float32)
        if gene_data is not None and dataset is None:
            self.gene_data = np.array(gene_data)
            return
        if dataset is None or id2vec_set is None or statistics_path is None:
            raise ValueError(r"dataset or id2vecset or statistics path shouldn't be empty.")
        # 0: present times
        # 1: present value sum
        # 2: present min
        # 3: present max
        # 4: present average
        # 5: absent times
        # 6: absent value sum
        # 7: absent min
        # 8: absent max
        # 9: absent average
        self.gene_data[0:self.vocabulary_size, (2, 7)] = 1.0
        counter = 0
        bt_sz = 100
        st = 0
        all_st = min(len(dataset), 100000)
        while st < all_st:
            batch_inputs, batch_labels = [], []
            # for _ in xrange(bt_sz):
            # if st < all_st:
                # if all_st < 100000:
            input_id_list, correct_labels = dataset[st]
            # else:
            #     input_id_list, correct_labels = random.choice(dataset)
            if len(input_id_list) < 5 or len(correct_labels) < 3:
                st += 1
                continue
            cnn_input = []
            for i in xrange(self.sentence_length):
                if pos is not None:
                    cnn_input.append(
                        [id2vec_set[input_id_list[i]][each] + pos[i][each] for each in
                         range(self.embedding_size)])
            batch_inputs.append(cnn_input)
            batch_labels.append(correct_labels)
            st += 1
            # else:
            #     break

            inputs = np.array(batch_inputs)
            all_res = self.session.run(self.full_connect_out_without_drop_with_sigmoid,
                                       feed_dict={self.batch_sentencelen_embeddingsize_channel: inputs})
            for st_res in xrange(len(all_res)):
                res = all_res[st_res]
                for w in xrange(self.vocabulary_size):
                    res_w = res[w]
                    if w in batch_labels[st_res]:
                        self.gene_data[w, 0] += 1
                        self.gene_data[w, 1] += res_w
                        if res_w < self.gene_data[w, 2]:
                            self.gene_data[w, 2] = res_w
                        if res_w > self.gene_data[w, 3]:
                            self.gene_data[w, 3] = res_w
                    else:
                        self.gene_data[w, 5] += 1
                        self.gene_data[w, 6] += res_w
                        if res_w < self.gene_data[w, 7]:
                            self.gene_data[w, 7] = res_w
                        if res_w > self.gene_data[w, 8]:
                            self.gene_data[w, 8] = res_w
            if st % 200 == 0:
                print("[CNN] Dealed %d sentence. " % st)

        # for input_id_list, correct_labels in dataset:
        #     cnn_input = []
        #     for i in xrange(self.sentence_length):
        #         if pos is not None:
        #             cnn_input.append(
        #                 [id2vec_set[input_id_list[i]][each] + pos[i][each] for each in range(self.embedding_size)])
        #     inputs = np.array([cnn_input])
        #     all_res = self.session.run(self.full_connect_out_without_drop_with_sigmoid,
        #                                feed_dict={self.batch_sentencelen_embeddingsize_channel: inputs})
        #     res = all_res[0]
        #     # top_k = 100
        #     # top_res = (-all_res[0, :]).argsort()[0:top_k]
        #     # p_str = "Answer to %s:" % str(input_id_list)
        #     # for k in xrange(top_k):
        #     #     p_str = "%s %s" % (p_str, id2word[top_res[k]])
        #     # print(p_str)
        #
        #     # update times sum min max
        #     for w in xrange(self.vocabulary_size):
        #         res_w = res[w]
        #         if w in correct_labels:
        #             self.gene_data[w, 0] += 1
        #             self.gene_data[w, 1] += res_w
        #             if res_w < self.gene_data[w, 2]:
        #                 self.gene_data[w, 2] = res_w
        #             if res_w > self.gene_data[w, 3]:
        #                 self.gene_data[w, 3] = res_w
        #         else:
        #             self.gene_data[w, 5] += 1
        #             self.gene_data[w, 6] += res_w
        #             if res_w < self.gene_data[w, 7]:
        #                 self.gene_data[w, 7] = res_w
        #             if res_w > self.gene_data[w, 8]:
        #                 self.gene_data[w, 8] = res_w
        #         # print(self.gene_data[w])
        #     counter += 1
        #     if counter % 2000 == 0:
        #         print("[CNN] Have handled %d sentences." % counter)
        # update average
        for w in xrange(self.vocabulary_size):
            if self.gene_data[w, 0] > 0:
                self.gene_data[w, 4] = self.gene_data[w, 1] / self.gene_data[w, 0]
            else:
                self.gene_data[w, 4] = -1
            if self.gene_data[w, 5] > 0:
                self.gene_data[w, 9] = self.gene_data[w, 6] / self.gene_data[w, 5]
            else:
                self.gene_data[w, 9] = -1
                # print(self.gene_data[w])
        with gfile.GFile(statistics_path, mode="w") as statistics_file:
            _cn = 0
            for label_m in self.gene_data:
                for _sp in label_m:
                    statistics_file.write(str(_sp))
                    statistics_file.write(" ")
                statistics_file.write("\n")
                _cn += 1
                if _cn % 20000 == 0:
                    print("[CNN] Having handled %d words." % _cn)

    def result_analysis(self, res,
                        bucket=False, top_k=(-1, -1),
                        only_present_average=False,
                        comprehensive_analysis=True,
                        pulse_analysis=False):
        """
        :param pulse_analysis:
        :param comprehensive_analysis:
        :param only_present_average: upper average's present
        :param top_k: directly achieve top k from all result
        :param res: label size list. the value should be in [0, 1]
        :param bucket:
        :return: id_list max to min
        """
        id_res = []
        if bucket:
            fore = 0
            rear = self.buckets_capacity[0]
            for i in xrange(len(self.buckets_capacity)):
                if fore < self.vocabulary_size and rear < self.vocabulary_size:
                    if rear == -1:
                        bucket_res = res[fore:]
                    else:
                        bucket_res = res[fore:rear]
                else:
                    break
                for index in (-bucket_res[:]).argsort()[0:self.buckets_sample_size[i]]:
                    id_res.append(index + fore)
                fore = rear
                rear = self.buckets_capacity[i + 1]
            return id_res
        if pulse_analysis:
            # Warning : shouldn't be used in uneven labels. some drift words will get so large value.
            for i in xrange(self.vocabulary_size):
                # all absent
                if self.gene_data[i, 0] < 1:
                    res[i] = 0.0
                # all present
                elif self.gene_data[i, 5] < 1:
                    res[i] = 1.0
                else:
                    if self.gene_data[i, 4] > self.gene_data[i, 9]:
                        # res[i] = res[i] / self.gene_data[i, 4]
                        # p
                        res[i] = (res[i] / self.gene_data[i, 4] + res[i] / self.gene_data[i, 9])
                    res[i] = 1 / (1 + math.exp(-res[i]))
            if top_k[0] > -1:
                id_res = (-res[:]).argsort()[top_k[0]:top_k[1]]
            else:
                for i in xrange(self.vocabulary_size):
                    if res[i] > 1:
                        id_res.append(i)
            return id_res
        if comprehensive_analysis:
            # Comprehensive analysis
            as_res = np.array(res)
            for i in xrange(self.vocabulary_size):
                # all absent
                if self.gene_data[i, 0] < 1:
                    as_res[i] = 0.0
                # all present
                elif self.gene_data[i, 5] < 1:
                    as_res[i] = 1.0
                else:
                    as_res[i] = (res[i] - self.gene_data[i, 4]) + (res[i] - self.gene_data[i, 9])
            if top_k[0] > -1:
                id_res = (-as_res[:]).argsort()[top_k[0]:top_k[1]]
            else:
                for i in xrange(self.vocabulary_size):
                    if as_res[i] > 0:
                        id_res.append(i)
            return id_res

        if only_present_average:
            if top_k[0] > -1:
                for i in xrange(self.vocabulary_size):
                    if 0 < self.gene_data[i, 4] < res[i]:
                        res[i] = res[i] - self.gene_data[i, 4]
                id_res = (-res[:]).argsort()[top_k[0]:top_k[1]]

            else:
                for i in xrange(self.vocabulary_size):
                    if 0 < self.gene_data[i, 4] < res[i]:
                        id_res.append(i)
            return id_res

        if top_k[0] > -1:
            id_res.extend((-res[:]).argsort()[top_k[0]:top_k[1]])
            return id_res

        raise ValueError("cannot analysis as this.")


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


def train_second_module(from_train_id, to_train_id, from_dev_id, to_dev_id,
                        vector_path, vocab_path=None,
                        para=None):
    if para is None:
        from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
        to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
        vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
        data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=32)
        word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
        id2frq = data_utils.get_vocab_frq(r"/home/chenyongchang/data/dia/vocabulary40000_frequency.txt",
                                          len(data_set))
        vocabulary_size = len(id2word)
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        pos_vec_list = build_position_vec(32, 200)
        cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "bucket_cnn_tgth")
        statistics_path = os.path.join(cnn_dir, "label_material")
        saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
    else:
        from_train_id = from_train_id
        to_train_id = to_train_id
        vector_path = vector_path
        data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=para.cnn_sentence_size)
        word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        # frq_path = os.path.join(para.data_dir, "vocabulary%d_frequency.txt" % para.vocabulary_size)
        # if not os.path.exists(frq_path):
        #     data_utils.creat_times_of_words([para.from_train_data, para.to_train_data], id2word, frq_path)
        # id2frq = data_utils.get_vocab_frq(frq_path, len(data_set))
        vocabulary_size = len(id2word)
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        pos_vec_list = build_position_vec(para.cnn_sentence_size, para.word_vector_neuron_size)
        cnn_dir = os.path.join(para.data_dir, "bucket_cnn_tgth")
        statistics_path = os.path.join(cnn_dir, "label_material")
        saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
    with tf.Session() as cnn_sess:
        cnn_model = CNNModel(session=cnn_sess,
                             batch_size=20,
                             vocabulary_size=vocabulary_size,
                             save_path=cnn_dir,
                             saved_model=os.path.join(cnn_dir, "cnn_tgth_model.ckpt-%d" % 200001)
                             )
        step_time, loss = 0.0, 0.0
        start_time = time.time()
        for step in xrange(200001):
            data_inputs, data_labels, buckets_labels = cnn_model.generate_batches_vector(
                id2vec_set=vec_set,
                data_set=data_set,
                pos=pos_vec_list)
            step_loss = cnn_model.train_buckets(data_inputs, data_labels, buckets_labels)

            loss += np.sum(step_loss) / 200
            if step % 200 == 0:
                step_time = (time.time() - start_time) / 200
                print("[CNN]The %d steps: every step use %.4f time(s). average loss: %.8f "
                      "" % (step, step_time, loss))
                # cnn_model.saver.save(cnn_sess, saved_model, global_step=cnn_model.global_step)
                # cnn_model.generator("说那么多连一开始分享的快乐都没有了",
                #                     word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
                # cnn_model.generator("我嗓子疼，随便吃点流食",
                #                     word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
                # cnn_model.generator("那我先去睡咯",
                #                     word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
                step_time, loss = 0.0, 0.0
                start_time = time.time()
                # if step % 10000 == 0:
        cnn_model.statistic_words(
                    dataset=data_set, id2vec_set=vec_set, pos=pos_vec_list, statistics_path=statistics_path)


def get_word_list(sentence):
    vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    vocabulary_size = len(id2word)
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    pos_vec_list = build_position_vec(32, 200)
    cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "bucket_cnn_tgth_scalar_loss")
    statistics_path = os.path.join(cnn_dir, "label_material")
    saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
    with tf.Session() as cnn_sess:
        cnn_model = CNNModel(session=cnn_sess,
                             batch_size=20,
                             vocabulary_size=vocabulary_size,
                             save_path=cnn_dir,
                             saved_model=r'/home/chenyongchang/data/dia/bucket_cnn_tgth/cnn_tgth_model.ckpt-200001')
        if os.path.exists(statistics_path):
            gene_data = data_utils.read_gene_data(statistics_path)
            cnn_model.statistic_words(gene_data=gene_data)
        id_list = cnn_model.generator(sentence,
                                      word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
        return id_list


if __name__ == "__main__":
    # train_second_module(None, None, None, None, None)
    # from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
    # to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
    # vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    # data_set = data_utils.read_data(from_train_id, to_train_id)
    # word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    # id2frq = data_utils.get_vocab_frq(r"/home/chenyongchang/data/dia/vocabulary40000_frequency.txt",
    #                                   len(data_set))
    # vocabulary_size = len(id2word)
    # vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    # pos_vec_list = build_position_vec(20, 200)
    # cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "cnn_tgth")
    # with tf.Session() as sess:
    #     cnn_model = CNNModel(session=sess,
    #                          batch_size=20,
    #                          vocabulary_size=vocabulary_size,
    #                          save_path=cnn_dir,
    #                          id2frq=id2frq)
    # from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
    # to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
    # vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    # # data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=32)
    # word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    # # id2frq = data_utils.get_vocab_frq(r"/home/chenyongchang/data/dia/vocabulary40000_frequency.txt",
    # #                                   len(data_set))
    # vocabulary_size = len(id2word)
    # vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    # pos_vec_list = build_position_vec(32, 200)
    # cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "bucket_cnn_tgth_scalar_loss")
    # statistics_path = os.path.join(cnn_dir, "label_material")
    #
    # saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
    # with tf.Session() as cnn_sess:
    #     cnn_model = CNNModel(session=cnn_sess,
    #                          batch_size=1,
    #                          vocabulary_size=vocabulary_size,
    #                          save_path=cnn_dir,
    #                          saved_model=r'/home/chenyongchang/data/dia/'
    #                                      r'bucket_cnn_tgth_scalar_loss/cnn_tgth_model.ckpt-340002')
    #     # for i in pos_vec_list:
    #     #     print(i)
    #     # if os.path.exists(statistics_path):
    #     #     gene_data = data_utils.read_gene_data(statistics_path)
    #     #     cnn_model.statistic_words(gene_data=gene_data)
    #     # else:
    #     #     cnn_model.statistic_words(
    #     #         dataset=data_set, id2vec_set=vec_set, pos=pos_vec_list,statistics_path=statistics_path)
    #     # print(cnn_model.gene_data[word2id["分享"]])
    #     # with gfile.GFile(statistics_path, mode="w") as statistics_file:
    #     #     _cn = 0
    #     #     for label_m in cnn_model.gene_data:
    #     #         for _sp in label_m:
    #     #             statistics_file.write(str(_sp))
    #     #             statistics_file.write(" ")
    #     #         statistics_file.write("\n")
    #     #         _cn += 1
    #     #         if _cn % 2000 == 0:
    #     #             print("[CNN] Having handled %d words." % _cn)
    #     sentence_list = [
    #         # "说那么多连一开始分享的快乐都没有了",
    #         #  "我嗓子疼，随便吃点流食",
    #         #  "那我先去睡咯",
    #         "聊的才开心呢",
    #         # "就陪父母看电视聊电视,然后散步走走,看景色",
    #         # "我觉得很一般",
    #         "我们已经是好友啦，一起来聊天吧！",
    #         # "果然我们的想法都不一样感受也不容易理解",
    #         # "说那么多连一开始分享的快乐都没有了",
    #         "今天天气不错啊",
    #         "在外面玩吗。别冻着。",
    #         "快去吧  到宿舍给我说声啊",
    #         "你今天可起来早了",
    #         "我现在准备去吃饭",
    #         "就这这性格啊 谁说的了",
    #         "都不知道人家怎么弄的，天天看着也很闲，也没有认真",
    #         "你实际更闲啦！",
    #         "刚买了一点点小零食"]
    #
    #     for sentence in sentence_list:
    #         id_list = cnn_model.generator(sentence,
    #                                       word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
    #         id_list = list(id_list)
    #         data_utils.show_id_list_word("Answer to %s:" % sentence, id2word, id_list, sp=" ")
            # with gfile.GFile(r'/home/chenyongchang/data/dia/train_record_Q.txt', mode="r") as f:
            #     with gfile.GFile(r'/home/chenyongchang/data/dia/train_middle_Q.txt', mode="w") as writer:
            #         counter = 0
            #         for line in f:
            #             line = tf.compat.as_text(line)
            #             id_list = cnn_model.generator(line,
            #                                           word2id=word2id, id2word=id2word, id2vec_set=vec_set,
            #                                           pos=pos_vec_list)
            #             for ids in id_list:
            #                 writer.write(str(ids))
            #                 writer.write(" ")
            #             writer.write("\n")
            #
            #             if counter % 300 == 0:
            #                 data_utils.show_id_list_word(
            #                     "Answer to %s (line %d):" % (line, counter), id2word, id_list, sp=" ")
            #             counter += 1

    # train_second_module(None, None, None, None, None)
    # word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    # data_utils.creat_times_of_words(
    #     [r'/home/chenyongchang/data/dia/train_record_Q.txt',
    #      r'/home/chenyongchang/data/dia/train_record_A.txt'],
    #     id2word,r'/home/chenyongchang/data/dia/vocabulary40000_frequency.txt')

    # --data_dir = / home / chenyongchang / data / movie_subtitles_en
    # --train_dir = / home / chenyongchang / data / movie_subtitles_en
    # --from_train_data = / home / chenyongchang / data / movie_subtitles_en / movie_subtitles_en_Q.txt
    # --to_train_data = / home / chenyongchang / data / movie_subtitles_en / movie_subtitles_en_A.txt
    # --from_dev_data = / home / chenyongchang / data / movie_subtitles_en / dev_movie_subtitles_en_Q.txt
    # --to_dev_data = / home / chenyongchang / data / movie_subtitles_en / dev_movie_subtitles_en_A.txt

    # from_train_id = r"/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_Q.txt.zh.vcb.ids40000"
    # to_train_id = r"/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_A.txt.zh.vcb.ids40000"
    # vector_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary_vector40000.txt"
    vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    # from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
    # to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
    # vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    # data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=32)
    word2id, id2word = data_utils.initialize_vocabulary(
        r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    # word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    # id2frq = data_utils.get_vocab_frq(r"/home/chenyongchang/data/dia/vocabulary40000_frequency.txt",
    #                                   len(data_set))
    vocabulary_size = len(id2word)
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    pos_vec_list = build_position_vec(32, 200)
    cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "bucket_cnn_tgth")
    # cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "bucket_cnn_tgth")
    statistics_path = os.path.join(cnn_dir, "label_material")
    saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
    with tf.Session() as cnn_sess:
        cnn_model = CNNModel(session=cnn_sess,
                             batch_size=1,
                             vocabulary_size=vocabulary_size,
                             save_path=cnn_dir,
                             saved_model=os.path.join(cnn_dir, "cnn_tgth_model.ckpt-%d" % 200001)
                             # id2frq=id2frq
                             )
        if os.path.exists(statistics_path):
            gene_data = data_utils.read_gene_data(statistics_path)
            size = len(gene_data)
            avr_sum = 0.0
            con = 0
            for l in gene_data:
                if l[4] > l[9] and con < 100:
                    avr_sum += l[9]
                    con += 1

            # print(size)
            print(avr_sum / con)
            raise ValueError("=-=")
            cnn_model.statistic_words(gene_data=gene_data)
        # else:
        #     cnn_model.statistic_words(
        #         dataset=data_set, id2vec_set=vec_set, pos=pos_vec_list,statistics_path=statistics_path)

        sentence_list = [
            # "We must reach the Valley of Reeds before the next dawn.",
            # "You know, I could just as well go without seeing him today."
            "说那么多连一开始分享的快乐都没有了",
            "我嗓子疼，随便吃点流食",
            "吃饭了吗",
            "那我先去睡咯",
            "晚安咯",
            "聊的才开心呢",
            "就陪父母看电视聊电视,然后散步走走,看景色",
            "我觉得很一般",
            # "我们已经是好友啦，一起来聊天吧！",
            "果然我们的想法都不一样感受也不容易理解",
            # "说那么多连一开始分享的快乐都没有了",
            "今天天气不错啊",
            "在外面玩吗。别冻着。",
            "快去吧  到宿舍给我说声啊",
            # "你今天可起来早了",
            # "我现在准备去吃饭",
            # "就这这性格啊 谁说的了",
            "都不知道人家怎么弄的，天天看着也很闲，也没有认真",
            "刚买了一点点小零食"
        ]
        for sentence in sentence_list:
            id_list = cnn_model.generator(sentence,
                                          word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
            id_list = list(id_list)
            # data_utils.show_id_list_word("Answer to %s:" % sentence, id2word, id_list, sp=" ")

            # data_utils.show_id_list_word("Words recommendation for %s\t\t" % sentence, id2word, id_list, sp=" ")
            print("Word recommendation for answering \"%s\":\t\t" % sentence)
            for x in xrange(8):
                ths_id = id_list[x*10: (x+1)*10]
                data_utils.show_id_list_word("\t\t", id2word, ths_id, sp=" ")
