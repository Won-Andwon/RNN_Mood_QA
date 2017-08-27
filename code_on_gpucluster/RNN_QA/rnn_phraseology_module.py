"""
rnn 模型
用于生成可读句子
"""
import data_utils
import numpy as np
import os
import random
import tensorflow as tf
import time
from six.moves import xrange



class RNNModel(object):
    def __init__(self,
                 cell_type="GRU",
                 num_layers=3,
                 neo_size=512,
                 output_size=200,
                 vocabulary_size=40000,
                 num_steps=32,
                 batch_size=20,
                 learning_rate=0.001,
                 learning_decay=0.99,
                 max_gradient_norm=5.0,
                 save_path=None):
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.save_path = save_path
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_decay)

        # 定义rnn的cell
        def single_cell():
            if cell_type == "GRU":
                return tf.contrib.rnn.GRUCell(neo_size)
            elif cell_type == "LSTM":
                return tf.contrib.rnn.BasicLSTMCell(neo_size)
            else:
                raise ValueError("Only support GRU or LSTM.")

        cell = single_cell()
        # 可以加上Dropout壳
        # tf.contrib.rnn.DropoutWrapper(
        #     cell(), output_keep_prob=keep_prob)
        # 多层模型
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        self.inputs_word_list = tf.placeholder(tf.int32, [None, num_steps],
                                               "sentence_inputs")
        self.targets = []
        for i in xrange(num_steps):
            self.targets.append(
                tf.placeholder(tf.float32, shape=[None, output_size], name="target{0}".format(i)))

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocabulary_size, neo_size], tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.inputs_word_list)
        # optional dropout
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        fc_w = tf.get_variable("full_connect_w", [neo_size, output_size], tf.float32)
        fc_b = tf.get_variable("full_connect_b", [output_size])

        self.logits = []
        losses = []
        for time_step in range(num_steps):
            logit = tf.matmul(outputs[time_step], fc_w) + fc_b
            self.logits.append(logit)
            loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(logit - self.targets[time_step]), axis=1) / output_size)
            losses.append(loss)
        all_loss = tf.add_n(losses)
        self.cost = all_loss
        self.final_state = state

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.learning_rate)

        gradients = tf.gradients(self.cost, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, session, inputs, targets):
        """
        inputs 应该是个句子的id列表 固定长度
        对应的targets 是个list 则是其中每个词后面词的embedding 同样长度
        :param session:
        :param inputs:
        :param targets:
        :return:
        """
        feeds = {}
        if len(inputs) != self.batch_size or len(targets) != self.num_steps:
            raise ValueError("输入句子的id长度和循环步数不同。%d %d %d %d"
                             "" % (self.num_steps, len(targets), len(inputs), self.batch_size))
        feeds[self.inputs_word_list] = inputs
        for l in xrange(self.num_steps):
            feeds[self.targets[l].name] = targets[l]
        outputs = session.run([self.updates, self.gradient_norms, self.cost], feeds)
        return outputs[1], outputs[2]

    def test(self, session, inputs, targets):
        feeds = {}
        feeds[self.inputs_word_list] = inputs
        for l in xrange(self.num_steps):
            feeds[self.targets[l].name] = targets[l]
        fetches = [self.cost]
        for l in xrange(self.num_steps):
            fetches.append(self.logits[l])
        outputs = session.run(fetches, feeds)
        return outputs[0], outputs[1:]

    def get_batches(self, data_set, id2vec_set):
        inputs = []
        targets_list = []
        for _ in xrange(self.batch_size):
            id_list = random.choice(data_set)
            inputs.append(id_list)

        for pos in xrange(self.num_steps - 1):
            pos_list = [id2vec_set[inputs[batch_idx][pos + 1]] for batch_idx in xrange(self.batch_size)]
            targets_list.append(np.array(pos_list) * 100)

        targets_list.append(np.array([id2vec_set[data_utils.PAD_ID] for _ in xrange(self.batch_size)]) * 100)

        batch_inputs = np.array(inputs)
        return batch_inputs, targets_list


def train_third_model(para=None, from_train_id=None, to_train_id=None,
                      vector_path=None, word_sample_path=None, vocab_path=None):
    if para is None:
        from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
        to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
        vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
        word_sample_path = r"/home/chenyongchang/data/dia/train_middle_Q.txt"
        data_set = data_utils.read_data_alloy(from_train_id, to_train_id, size=32)
        data_word = data_utils.read_data_one_file(word_sample_path)
        word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
        vocabulary_size = min(len(id2word), 40000)
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        rnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "another_rnn")
        rnn_model = os.path.join(rnn_dir, "rnn_model.ckpt")
    else:
        from_train_id = from_train_id
        to_train_id = to_train_id
        word_sample_path = word_sample_path
        data_set = data_utils.read_data_alloy(from_train_id, to_train_id, size=para.rnn_sentence_size)
        data_word = data_utils.read_data_one_file(word_sample_path)
        word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        vocabulary_size = min(len(id2word), para.vocabulary_size)
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        rnn_dir = os.path.join(para.data_dir, "another_rnn")
        rnn_model = os.path.join(rnn_dir, "rnn_model.ckpt")
    with tf.Session() as sess:
        model = RNNModel(save_path=rnn_dir, vocabulary_size=vocabulary_size)
        ckpt = tf.train.get_checkpoint_state(rnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        # Read data into buckets and compute their sizes.
        print("Reading data and get into training.")

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            start_time = time.time()
            inputs, targets = model.get_batches(data_set, id2vec_set=vec_set)
            _, step_loss = model.train(sess, inputs, targets)
            step_time += (time.time() - start_time) / 200
            loss += step_loss / 200
            current_step += 1
            if current_step % 200 == 0:
                print("[Seq2Seq model] global step %d learning rate %.4f step-time %.2f loss "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                model.saver.save(sess, rnn_model, global_step=model.global_step)
                step_time, loss = 0.0, 0.0


def cal_proximity(em1, em2):
    size1, size2 = len(em1), len(em2)
    if size1 != size2:
        raise ValueError("length not equal.")
    diff = 0.0
    for i in range(size1):
        diff += abs(em1[i] - em2[i])
    return diff


def fine_neibor(em, vec_set):
    size = len(vec_set)
    pro = []
    for i in xrange(size):
        e = cal_proximity(vec_set[i], em)
        pro.append(e)
    pro = np.array(pro)
    top_k = 10
    return pro.argsort()[0:top_k]


if __name__ == "__main__":
    train_third_model()

    from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
    to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
    vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    word_sample_path = r"/home/chenyongchang/data/dia/train_middle_Q.txt"
    data_set = data_utils.read_data_alloy(from_train_id, to_train_id, size=32)
    data_word = data_utils.read_data_one_file(word_sample_path)
    word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    vocabulary_size = min(len(id2word), 40000)
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    rnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "another_rnn")
    rnn_model = os.path.join(rnn_dir, "rnn_model.ckpt")
    with tf.Session() as sess:
        model = RNNModel(save_path=rnn_dir, vocabulary_size=vocabulary_size, batch_size=1)
        ckpt = tf.train.get_checkpoint_state(rnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        inputs, targets = model.get_batches(data_set, id2vec_set=vec_set)
        _, res = model.test(sess, inputs, targets)
        print(inputs)
        print(targets)
        print(res)
        data_utils.show_id_list_word("the sentence:", id_list=inputs[0], id2word=id2word, sp=" ")
        for i in xrange(32):
            wanted = targets[i][0]
            gene = res[i][0]
            print(cal_proximity(wanted, gene))
            other = fine_neibor(gene, vec_set)
            print(other)
            data_utils.show_id_list_word("the chosen:", id_list=other, id2word=id2word, sp=" ")


        # Read data into buckets and compute their sizes.
        # print("Reading data and get into training.")
        #
        # # This is the training loop.
        # step_time, loss = 0.0, 0.0
        # current_step = 0
        # previous_losses = []
        # while True:
        #     start_time = time.time()
        #     inputs, targets = model.get_batches(data_set, id2vec_set=vec_set)
        #     _, step_loss = model.train(sess, inputs, targets)
        #     step_time += (time.time() - start_time) / 200
        #     loss += step_loss / 200
        #     current_step += 1
        #     if current_step % 200 == 0:
        #         print("[Seq2Seq model] global step %d learning rate %.4f step-time %.2f loss "
        #               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, loss))
        #         # Decrease learning rate if no improvement was seen over last 3 times.
        #         if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
        #             sess.run(model.learning_rate_decay_op)
        #         previous_losses.append(loss)
        #         # Save checkpoint and zero timer and loss.
        #         model.saver.save(sess, rnn_model, global_step=model.global_step)
        #         step_time, loss = 0.0, 0.0