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

    def train(self, batch_inputs, batch_labels):
        feed_dict = {
            self.batch_sentencelen_embeddingsize_channel: batch_inputs,
            self.batch_labels: batch_labels
        }
        _, loss_val = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss_val

    def generate_batches_vector(self, id2vec_set, data_set, pos=None, id2frq=None):
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
                if id2frq[ids] > 0:
                    temp_cnn_label[ids] = 1
            cnn_inputs.append(cnn_input)
            cnn_labels.append(temp_cnn_label)

        batch_inputs = np.array(cnn_inputs)
        batch_labels = np.array(cnn_labels)

        return batch_inputs, batch_labels

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

        res = self.session.run(self.full_connect_out_without_drop,
                               feed_dict={self.batch_sentencelen_embeddingsize_channel: batch_inputs})
        # print(res[0][word2id["做人"]])
        idlist = self.result_analysis(res[0], bucket=False,
                                      top_k=100,
                                      only_present_average=False,
                                      comprehensive_analysis=False,
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
            all_res = self.session.run(self.full_connect_out_without_drop,
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
                        bucket=False, top_k=-1,
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
            raise ValueError("Now we have no buckets.")
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
            if top_k > 0:
                id_res = (-res[:]).argsort()[0:top_k]
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
            if top_k > 0:
                id_res = (-as_res[:]).argsort()[0:top_k]
            else:
                for i in xrange(self.vocabulary_size):
                    if as_res[i] > 0:
                        id_res.append(i)
            return id_res

        if only_present_average:
            if top_k > 0:
                for i in xrange(self.vocabulary_size):
                    if 0 < self.gene_data[i, 4] < res[i]:
                        res[i] = res[i] - self.gene_data[i, 4]
                id_res = (-res[:]).argsort()[0:top_k]

            else:
                for i in xrange(self.vocabulary_size):
                    if 0 < self.gene_data[i, 4] < res[i]:
                        id_res.append(i)
            return id_res

        if top_k > 0:
            id_res.extend((-res[:]).argsort()[0:top_k])
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


def train_second_module(from_train_id, to_train_id, vocab_path, vector_path, para=None):
    if para is None:
        from_train_id = r"/home/chenyongchang/data/weibo/stc_weibo_train_post.zh.vcb.ids40000"
        to_train_id = r"/home/chenyongchang/data/weibo/stc_weibo_train_response.zh.vcb.ids40000"
        vocab_path = r'/home/chenyongchang/data/weibo/vocabulary40000.txt'
        vector_path = r"/home/chenyongchang/data/weibo/vocabulary_vector40000.txt"
        data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=32)
        word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        vocabulary_size = min(len(id2word), 40000)
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        pos_vec_list = build_position_vec(32, 512)
        cnn_dir = os.path.join(r'/home/chenyongchang/data/weibo/', "cnn_logic")
        statistics_path = os.path.join(cnn_dir, "label_material")
        saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
    else:
        from_train_id = from_train_id
        to_train_id = to_train_id
        vector_path = vector_path
        data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=para.cnn_sentence_size)
        word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        vocabulary_size = len(id2word)
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        pos_vec_list = build_position_vec(para.cnn_sentence_size, para.word_vector_neuron_size)
        cnn_dir = os.path.join(para.data_dir, "logic_cnn")
        statistics_path = os.path.join(cnn_dir, "label_material")
        saved_model = os.path.join(cnn_dir, "logic.ckpt")
    with tf.Session() as cnn_sess:
        cnn_model = CNNModel(batch_size=20,
                             vocabulary_size=vocabulary_size,
                             embedding_size=para.word_vector_neuron_size,
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
                pos=pos_vec_list,
                id2frq=id2frq)
            step_loss = cnn_model.train(data_inputs, data_labels)

            loss += np.sum(step_loss) / 2000
            if step % 2000 == 0:
                step_time = (time.time() - start_time) / 2000
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
        cnn_model.statistic_words(
                    dataset=data_set, id2vec_set=vec_set, pos=pos_vec_list, statistics_path=statistics_path)

if __name__ == "__main__":
    vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    vocabulary_size = min(len(id2word), 40000)
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    pos_vec_list = build_position_vec(32, 512)
    cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "logic_cnn")
    statistics_path = os.path.join(cnn_dir, "label_material")
    saved_model = os.path.join(cnn_dir, "logic.ckpt")
    with tf.Session() as cnn_sess:
        cnn_model = CNNModel(batch_size=1,
                             vocabulary_size=vocabulary_size,
                             save_path=cnn_dir)
        ckpt = tf.train.get_checkpoint_state(cnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            cnn_model.saver.restore(cnn_sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            cnn_sess.run(tf.global_variables_initializer())
        gene_data = data_utils.read_gene_data(statistics_path)
        cnn_model.statistic_words(gene_data=gene_data)
        # else:
        #     cnn_model.statistic_words(
        #         dataset=data_set, id2vec_set=vec_set, pos=pos_vec_list,statistics_path=statistics_path)

        sentence_list = [
            # "说那么多连一开始分享的快乐都没有了",
             "我嗓子疼，随便吃点流食",
             "那我先去睡咯",
            "聊的才开心呢",
            "就陪父母看电视聊电视,然后散步走走,看景色",
            "我觉得很一般",
            "我们已经是好友啦，一起来聊天吧！",
            "果然我们的想法都不一样感受也不容易理解",
            "说那么多连一开始分享的快乐都没有了",
            "今天天气不错啊",
            "在外面玩吗。别冻着。",
            "快去吧  到宿舍给我说声啊",
            "你今天可起来早了",
            "我现在准备去吃饭",
            "就这这性格啊 谁说的了",
            "都不知道人家怎么弄的，天天看着也很闲，也没有认真",
            "你实际更闲啦！",
            "刚买了一点点小零食"]
        for sentence in sentence_list:
            id_list = cnn_model.generator(sentence,
                                          word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
            id_list = list(id_list)
            # data_utils.show_id_list_word("Answer to %s:" % sentence, id2word, id_list, sp=" ")
            data_utils.show_id_list_word("Answer to %s\t" % sentence, id2word, id_list, sp=" ")
