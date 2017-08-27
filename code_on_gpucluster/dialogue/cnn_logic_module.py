"""
CNN
判断哪些词出现在应对语句中
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from six.moves import xrange

import data_utils


class CNNModel(object):
    def __init__(self,
                 batch_size=10,
                 sentence_length=32,
                 vocabulary_size=40000,
                 embedding_size=512,
                 drop_pro=0.4,
                 depth=3,
                 filter_size=3,
                 filter_num=(1024, 512, 256),
                 save_path=None):
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.sentence_length = sentence_length
        self.embedding_size = embedding_size
        self.drop_pro = drop_pro
        self.depth = depth
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.save_path = save_path
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.build_graph()

    def build_cnn_layer(self, input_from_last_layer, layer_id):
        if layer_id == 1:
            filter_shape = [self.filter_size, self.embedding_size, self.filter_num[0]]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.01, shape=[filter_shape[2]]), name="conv_Bias")
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[2],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")
            relu = tf.nn.relu(pool, name="conv_Relu")
            res = relu
        elif layer_id == 2:
            filter_shape = [self.filter_size, self.filter_num[0], self.filter_num[1]]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.01, shape=[filter_shape[2]]), name="conv_Bias")
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[1],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")
            relu = tf.nn.relu(pool, name="conv_Relu")
            res = relu
        elif layer_id == 3:
            filter_shape = [self.filter_size, self.filter_num[1], self.filter_num[2]]
            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_Weight")
            conv = tf.nn.conv1d(input_from_last_layer, weight, stride=1, padding="SAME")
            bias = tf.Variable(tf.constant(0.01, shape=[filter_shape[2]]), name="conv_Bias")
            pre_activation = tf.nn.bias_add(conv, bias)
            pool = tf.nn.pool(pre_activation,
                              window_shape=[self.filter_size],
                              strides=[2],
                              pooling_type="MAX",
                              padding="SAME",
                              name="conv_Max_Pool")
            relu = tf.nn.relu(pool, name="conv_Relu")
            res = relu
        else:
            raise ValueError("the wrong layer id.")
        tf.add_to_collection("cnn_vars", weight)
        tf.add_to_collection("cnn_vars", bias)
        return res

    def build_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_sentence_len_embedding_size = tf.placeholder(
            tf.float32, [None, self.sentence_length, self.embedding_size])
        self.batch_labels = tf.placeholder(tf.float32, [None, self.vocabulary_size])

        input_from_last_layer = self.batch_sentence_len_embedding_size

        for i in xrange(self.depth):
            with tf.name_scope("CNN-layer-%d" % (i + 1)):
                input_from_last_layer = self.build_cnn_layer(input_from_last_layer, i + 1)

        self.hidden_out = input_from_last_layer
        flat_length = (self.sentence_length // 4) * self.filter_num[2]

        res_flat = tf.reshape(input_from_last_layer, [-1, flat_length])

        res_drop = tf.nn.dropout(res_flat, self.drop_pro, name="Dropout")
        full_connect_w = tf.Variable(
            tf.random_normal([flat_length, self.vocabulary_size], stddev=0.01), name="cnn_fc_w")
        full_connect_b = tf.Variable(
            tf.random_normal([self.vocabulary_size], stddev=0.01), name="cnn_fc_b")
        tf.add_to_collection("cnn_vars", full_connect_w)
        tf.add_to_collection("cnn_vars", full_connect_b)

        self.full_connect_out_with_drop = tf.matmul(res_drop, full_connect_w) + full_connect_b
        self.full_connect_out_without_drop = tf.matmul(res_flat, full_connect_w) + full_connect_b

        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.full_connect_out_with_drop, labels=self.batch_labels), axis=0,
            name="cnn_loss")

        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(
            self.loss, global_step=self.global_step, var_list=tf.get_collection("cnn_vars"))

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch_inputs, batch_labels):
        feed_dict = {
            self.batch_sentence_len_embedding_size: batch_inputs,
            self.batch_labels: batch_labels
        }
        _, loss_val = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        # hidden_res = sess.run(self.hidden_out, feed_dict=feed_dict)
        # for each in hidden_res[0]:
        #     print(each)
        return loss_val

    def generate_batches_vector(self, id2vec_set, data_set, pos=None):
        st_size, em_size = 0, 0
        if pos is not None:
            st_size = len(pos)
            em_size = len(pos[0])
            if em_size != self.embedding_size or st_size != self.sentence_length:
                raise ValueError("position vector error.")
        cnn_inputs, cnn_labels = [], []
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
            for ids in cnn_label:
                temp_cnn_label[ids] = 1
            cnn_inputs.append(cnn_input)
            cnn_labels.append(temp_cnn_label)

        batch_inputs = np.array(cnn_inputs)
        batch_labels = np.array(cnn_labels)

        return batch_inputs, batch_labels

    def generator(self, sess, sentence_ids, id2vec_set, pos=None):
        cnn_inputs, cnn_input = [], []
        if pos is not None:
            st_size = len(pos)
            if st_size != self.sentence_length:
                raise ValueError("sentence's length error!")
            em_size = len(pos[0])
            for i in xrange(self.sentence_length):
                if i < len(sentence_ids):
                    cnn_input.append([id2vec_set[sentence_ids[i]][each] + pos[i][each] for each in range(em_size)])
                else:
                    cnn_input.append([id2vec_set[data_utils.PAD_ID][each] + pos[i][each] for each in range(em_size)])
        else:
            for x in sentence_ids:
                cnn_input.append(id2vec_set[x])
            if len(cnn_input) >= self.sentence_length:
                cnn_input = cnn_input[:self.sentence_length]
            else:
                for i in range((self.sentence_length - len(cnn_input))):
                    cnn_input.append(id2vec_set[data_utils.PAD_ID])

        cnn_inputs.append(cnn_input)
        batch_inputs = np.array(cnn_inputs)

        res = sess.run(self.full_connect_out_without_drop,
                       feed_dict={self.batch_sentence_len_embedding_size: batch_inputs})
        # hidden = sess.run(self.hidden_out, feed_dict={self.batch_sentence_len_embedding_size: batch_inputs})
        # print(hidden)
        idlist = self.result_analysis(res[0],
                                      top_k=(0, 100),
                                      only_present_average=False,
                                      comprehensive_analysis=False,
                                      pulse_analysis=True)
        return idlist

    def result_analysis(self, res, top_k=(-1, -1),
                        only_present_average=False,
                        comprehensive_analysis=True,
                        pulse_analysis=False):
        """
        :param pulse_analysis:
        :param comprehensive_analysis:
        :param only_present_average: upper average's present
        :param top_k: directly achieve top k from all result (top_k0, top_k1)
        :param res: label size list. the value should be in [0, 1]
        :return: id_list max to min
        """
        id_res = []
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

    def statistic_words(self, sess,
                        dataset=None, id2vec_set=None,
                        pos=None, gene_data=None,
                        statistics_path=None):
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

        bt_sz = 100
        st = 0
        all_st = min(len(dataset), 100000)
        while st < all_st:
            batch_inputs, batch_labels = [], []
            for _ in xrange(bt_sz):
                if st < all_st:
                    input_id_list, correct_labels = random.choice(dataset)
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
                else:
                    break

            inputs = np.array(batch_inputs)
            all_res = sess.run(self.full_connect_out_without_drop,
                               feed_dict={self.batch_sentence_len_embedding_size: inputs})

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


def train_second_module(from_train_id, to_train_id, vocab_path, vector_path, para=None):
    if para is None:
        from_train_id = r"/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_Q.txt.zh.vcb.ids40000"
        to_train_id = r"/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_A.txt.zh.vcb.ids40000"
        vocab_path = r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000.txt'
        vector_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary_vector40000.txt"
        data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=32)
        word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        vocabulary_size = min(len(id2word), 40000)
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        pos_vec_list = build_position_vec(32, 512)
        embedding_size = 512
        cnn_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "cnn_logic")
        statistics_path = os.path.join(cnn_dir, "label_material")
        saved_model = os.path.join(cnn_dir, "logic.ckpt")
        # from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
        # to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
        # vocab_path = r'/home/chenyongchang/data/dia/vocabulary40000.txt'
        # vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
        # data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=32)
        # word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        # vocabulary_size = min(len(id2word), 40000)
        # vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        # pos_vec_list = build_position_vec(32, 200)
        # embedding_size = 200
        # cnn_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "cnn_logic_test0")
        # statistics_path = os.path.join(cnn_dir, "label_material")
        # saved_model = os.path.join(cnn_dir, "logic.ckpt")
    else:
        from_train_id = from_train_id
        to_train_id = to_train_id
        vector_path = vector_path
        data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=para.cnn_sentence_size)
        word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        vocabulary_size = min(len(id2word), para.vocabulary_size)
        embedding_size = para.word_vector_neuron_size
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        pos_vec_list = build_position_vec(para.cnn_sentence_size, para.word_vector_neuron_size)
        cnn_dir = os.path.join(para.data_dir, "logic_cnn")
        statistics_path = os.path.join(cnn_dir, "label_material")
        saved_model = os.path.join(cnn_dir, "logic.ckpt")
    with tf.Session() as cnn_sess:
        cnn_model = CNNModel(batch_size=20,
                             vocabulary_size=vocabulary_size,
                             embedding_size=embedding_size,
                             save_path=cnn_dir)
        ckpt = tf.train.get_checkpoint_state(cnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            cnn_model.saver.restore(cnn_sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            cnn_sess.run(tf.global_variables_initializer())
        step_time, loss = 0.0, 0.0
        start_time = time.time()
        for step in xrange(200001):
            data_inputs, data_labels = cnn_model.generate_batches_vector(
                id2vec_set=vec_set,
                data_set=data_set,
                pos=pos_vec_list)

            step_loss = cnn_model.train(cnn_sess, data_inputs, data_labels)

            loss += np.sum(step_loss) / 200
            if step % 200 == 0:
                step_time = (time.time() - start_time) / 200
                print("[CNN]The %d steps: every step use %.4f time(s). average loss: %.8f "
                      "" % (step, step_time, loss))
                # cnn_model.saver.save(cnn_sess, saved_model, global_step=cnn_model.global_step)
                step_time, loss = 0.0, 0.0
                start_time = time.time()
        cnn_model.statistic_words(sess=cnn_sess,
                                  dataset=data_set,
                                  id2vec_set=vec_set,
                                  pos=pos_vec_list,
                                  statistics_path=statistics_path)

if __name__ == "__main__":
    # train_second_module(None, None, None, None, None)
    vector_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary_vector40000.txt"
    word2id, id2word = data_utils.initialize_vocabulary(
        r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000.txt')
    vocabulary_size = min(len(id2word), 40000)
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    pos_vec_list = build_position_vec(32, 512)
    cnn_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "cnn_logic")
    statistics_path = os.path.join(cnn_dir, "label_material")
    saved_model = os.path.join(cnn_dir, "logic.ckpt")
    with tf.Session() as cnn_sess:
        cnn_model = CNNModel(batch_size=1,
                             vocabulary_size=vocabulary_size,
                             embedding_size=512,
                             save_path=cnn_dir)
        ckpt = tf.train.get_checkpoint_state(cnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            cnn_model.saver.restore(cnn_sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("no model to restore.")
        gene_data = data_utils.read_gene_data(statistics_path)
        size = len(gene_data)
        avr_sum = 0.0
        ch_sum = 0.0
        bu_sum = 0.0
        con = 0
        for l in gene_data:
            print(l)
            if l[4] > l[9] and con < 100:
                avr_sum += l[4] - l[9]
                ch_sum += l[4]
                bu_sum += l[9]
                con += 1

        # print(size)
        print(avr_sum / con)
        print(ch_sum / con)
        print(bu_sum / con)
        raise ValueError("=-=")

        cnn_model.statistic_words(sess=cnn_sess, gene_data=gene_data)

        with gfile.GFile(r'/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_Q.txt', mode="r") as f:
            with gfile.GFile(r'/home/chenyongchang/data/movie_subtitles_en/train_middle_Q_t.txt', mode="w") as writer:
                counter = 0
                for line in f:
                    line = tf.compat.as_text(line)
                    sen_ids = data_utils.sentence_to_token_ids(line, word2id)
                    id_list = cnn_model.generator(sess=cnn_sess, sentence_ids=sen_ids,
                                                  id2vec_set=vec_set, pos=pos_vec_list)
                    # print(id_list)
                    for ids in id_list:
                        writer.write(str(ids))
                        writer.write(" ")
                    writer.write("\n")

                    if counter % 300 == 0:
                        data_utils.show_id_list_word(
                            "Answer to %s (line %d):" % (line, counter), id2word, id_list, sp=" ")
                    counter += 1

        sentence_list = [
            "You have my word. As a gentleman",
            "Fine.Have a seat.",
            "that's all we're willing to say.",
            "Why would he take it? He don't even know you.",
            "How are you holding up?"
            ]
        for sentence in sentence_list:
            sen_ids = data_utils.sentence_to_token_ids(sentence, word2id)
            id_list = cnn_model.generator(
                sess=cnn_sess, sentence_ids=sen_ids, id2vec_set=vec_set, pos=pos_vec_list)
            id_list = list(id_list)
            data_utils.show_id_list_word("Answer to %s\t" % sentence, id2word, id_list, sp=" ")
