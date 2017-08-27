"""
rnn 模型
用于生成可读句子
"""
import data_utils
import math
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
                 output_size=40000,
                 vocabulary_size=40000,
                 num_steps=16,  # 等同于句子长度 固定较为方便实现
                 batch_size=20,  # 我这个例子可能越小越好？
                 learning_rate=0.1,
                 learning_decay=0.99,
                 max_gradient_norm=5.0,
                 save_path=None,
                 on_training=True):
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size

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

        encoder_cell = single_cell()
        decoder_cell = single_cell()
        # 可以加上Dropout壳
        # tf.contrib.rnn.DropoutWrapper(
        #     cell(), output_keep_prob=keep_prob)
        # 多层模型
        if num_layers > 1:
            encoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
            decoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        self.initial_state = encoder_cell.zero_state(batch_size, tf.float32)
        # print(self.initial_state)
        self.encoder_inputs_word_list = tf.placeholder(tf.int32, [None, num_steps],
                                                       "sentence_encoder_inputs")
        self.decoder_inputs_word_list = tf.placeholder(tf.int32, [None, num_steps],
                                                       "sentence_decoder_inputs")
        self.targets = []
        for i in xrange(num_steps):
            self.targets.append(
                tf.placeholder(tf.int32, shape=[None], name="target{0}".format(i)))

        self.mask = []
        for i in xrange(num_steps):
            self.mask.append(
                tf.placeholder(tf.float32, shape=[None, vocabulary_size], name="mask{0}".format(i)))

        self.test_inputs_state = []
        for i in xrange(num_layers):
            self.test_inputs_state.append(tf.placeholder(tf.float32, [None, neo_size], "input_state{0}".format(i)))
        # print(self.test_inputs_state)
        self.test_inputs = tf.placeholder(tf.int32, [None, 1], "input_word")
        self.test_mask = tf.placeholder(tf.float32, [None, vocabulary_size], "input_word_mask")

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocabulary_size, neo_size], tf.float32)
            encoder_inputs = tf.nn.embedding_lookup(embedding, self.encoder_inputs_word_list)
            decoder_inputs = tf.nn.embedding_lookup(embedding, self.decoder_inputs_word_list)
            test_em_inputs = tf.nn.embedding_lookup(embedding, self.test_inputs)
        # optional dropout
        # encoder_outputs = []
        encoder_state = self.initial_state
        with tf.variable_scope("RNN_Encoder"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, encoder_state = encoder_cell(encoder_inputs[:, time_step, :], encoder_state)
                # encoder_outputs.append(cell_output)
        # print(state)
        self.encoder_state_output = encoder_state
        decoder_state = encoder_state
        outputs = []
        with tf.variable_scope("RNN_Decoder"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, decoder_state = decoder_cell(decoder_inputs[:, time_step, :], decoder_state)
                outputs.append(cell_output)

        t_fc_w = tf.get_variable("full_connect_w", [output_size, neo_size], tf.float32)
        fc_w = tf.transpose(t_fc_w)
        fc_b = tf.get_variable("full_connect_b", [output_size])

        def sampled_loss(labels, logits):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(
                weights=t_fc_w,
                biases=fc_b,
                labels=labels,
                inputs=logits,
                num_sampled=300,
                num_classes=vocabulary_size)
        softmax_loss_function = sampled_loss

        # losses = []
        logits = []
        for time_step in xrange(num_steps):
            ori_logit = tf.matmul(outputs[time_step], fc_w) + fc_b
            logit = tf.multiply(ori_logit, self.mask[time_step])
            # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels=self.targets[time_step], logits=logit), name="loss")
            # losses.append(loss)
            logits.append(logit)
        # all_loss = tf.add_n(losses)
        loss = tf.contrib.legacy_seq2seq.sequence_loss(
            logits=logits,
            targets=self.targets,
            weights=[tf.ones([batch_size], dtype=tf.float32)] * num_steps)

        self.cost = loss
        self.final_state = decoder_state
        self.logits_out = logits[0]
        # print(self.logits_out)

        if not on_training:
            self.test_output, next_state = decoder_cell(test_em_inputs[:, 0, :], self.test_inputs_state)
            ori_logit = tf.matmul(self.test_output, fc_w) + fc_b
            logit = tf.multiply(ori_logit, self.test_mask)
            self.res_logit = logit
            self.final_de_state = next_state

        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        gradients = tf.gradients(self.cost, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, session, encoder_inputs, decoder_inputs, targets, masks):
        feeds = {}
        if len(encoder_inputs) != self.batch_size \
                or len(decoder_inputs) != self.batch_size \
                or len(targets) != self.num_steps \
                or len(masks) != self.num_steps:
            raise ValueError("输入长度和循环步数不同。%d %d %d %d %d"
                             "" % (self.num_steps, len(encoder_inputs), len(decoder_inputs), len(targets), len(masks)))
        feeds[self.encoder_inputs_word_list] = encoder_inputs
        feeds[self.decoder_inputs_word_list] = decoder_inputs
        for l in xrange(self.num_steps):
            feeds[self.targets[l].name] = targets[l]
            feeds[self.mask[l].name] = masks[l]
        outputs = session.run([self.updates, self.gradient_norms, self.cost], feeds)
        # logit = session.run(self.logits_out, feeds)
        # encoder_state = session.run(self.encoder_state_output, feeds)
        # print(encoder_state[0][0][0:30])
        # print(encoder_state[1][0][0:30])
        # print(encoder_state[2][0][0:30])
        # print(logit[0][0:30])
        # print((-logit[0]).argsort()[0:30])
        # print(np.sort(-logit[0])[0:30])
        # print(masks[0][0][0:30])
        # print(targets[0][0])
        # outputs = [1,2,3]
        return outputs[1], outputs[2]

    def get_batches(self, data_set, id2vec_set, id_mask):
        encoder_inputs = []
        decoder_inputs = []
        targets_list = []
        mask_list = []
        for _ in xrange(self.batch_size):
            en_id, de_id = random.choice(data_set)
            en_id = list(reversed(en_id))
            encoder_inputs.append(en_id)
            decoder_inputs.append(de_id)
        # target is the next decoder word.
        for pos in xrange(self.num_steps - 1):
            pos_list = [decoder_inputs[batch_idx][pos + 1] for batch_idx in xrange(self.batch_size)]
            targets_list.append(np.array(pos_list))
        # for each word in decoder, making a mask
        for pos in xrange(self.num_steps):
            mask = [
                self.get_mask(decoder_inputs[batch_idx][pos], id_mask) for batch_idx in xrange(self.batch_size)]
            mask_list.append(np.array(mask))

        targets_list.append([data_utils.PAD_ID for _ in xrange(self.batch_size)])
        # mask_list.append(
        #     [np.zeros(shape=[self.vocabulary_size], dtype=np.float32) for _ in xrange(self.batch_size)])

        batch_encoder_inputs = np.array(encoder_inputs)
        batch_decoder_inputs = np.array(decoder_inputs)
        return batch_encoder_inputs, batch_decoder_inputs, targets_list, mask_list

    def get_mask(self, idix, id_mask):
        init_mask = np.zeros(shape=[self.vocabulary_size], dtype=np.float32)
        for each in id_mask[idix]:
            init_mask[each[0]] = each[1]
        return init_mask

    def test(self, sess, sentence, word2id, id2word, id_mask):
        encoder_word_list = data_utils.sentence_to_token_ids(sentence, word2id)
        if len(encoder_word_list) >= self.num_steps:
            encoder_word_list = list(reversed(encoder_word_list[0:self.num_steps]))
        else:
            pad_size = self.num_steps - len(encoder_word_list)
            encoder_word_list = list(reversed(encoder_word_list + [data_utils.PAD_ID] * pad_size))
        state = sess.run([self.encoder_state_output], {self.encoder_inputs_word_list: encoder_word_list})
        feeds = {}
        decoder_input = [data_utils.GO_ID]
        feeds[self.test_inputs] = decoder_input
        decoder_input_mask = [self.get_mask(data_utils.GO_ID, id_mask)]
        feeds[self.test_mask] = decoder_input_mask
        for l in xrange(self.num_layers):
            feeds[self.test_inputs_state[l].name] = state[l]
        res, state = sess.run([self.res_logit, self.final_de_state], feeds)

        res_id = (-res[0]).argsort()[0:10]
        for idix in res_id:
            print(id2word[idix])


def train_third_model(para=None, from_train_id=None, to_train_id=None,
                      vector_path=None, word_sample_path=None, vocab_path=None):
    if para is None:
        # from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
        # to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
        # vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
        # # word_sample_path = r"/home/chenyongchang/data/dia/train_middle_Q.txt"
        # id_mask_path = r"/home/chenyongchang/data/dia/bin_group.txt"
        # data_set = data_utils.read_data_divide(from_train_id, to_train_id, size=16)
        # id_mask = data_utils.read_mask_data(id_mask_path)
        # # data_word = data_utils.read_data_one_file(word_sample_path)
        # word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
        # vocabulary_size = min(len(id2word), 40000)
        # vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        # rnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "rnn_word_new_loss_test0")
        # rnn_model = os.path.join(rnn_dir, "rnn_model.ckpt")
        from_train_id = r"/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_Q.txt.zh.vcb.ids40000"
        to_train_id = r"/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_A.txt.zh.vcb.ids40000"
        vector_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary_vector40000.txt"
        # word_sample_path = r"/home/chenyongchang/data/dia/train_middle_Q.txt"
        id_mask_path = r"/home/chenyongchang/data/movie_subtitles_en/bin_group.txt"
        data_set = data_utils.read_data_divide(from_train_id, to_train_id, size=16)
        id_mask = data_utils.read_mask_data(id_mask_path)
        # data_word = data_utils.read_data_one_file(word_sample_path)
        word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000.txt')
        vocabulary_size = min(len(id2word), 40000)
        vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        rnn_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "rnn_word_new_loss_test0")
        rnn_model = os.path.join(rnn_dir, "rnn_model.ckpt")
    else:
        from_train_id = from_train_id
        to_train_id = to_train_id
        word_sample_path = word_sample_path
        id_mask_path = r"/home/chenyongchang/data/dia/bin_group.txt"
        data_set = data_utils.read_data_divide(from_train_id, to_train_id, size=para.rnn_sentence_size)
        id_mask = data_utils.read_mask_data(id_mask_path)
        # data_word = data_utils.read_data_one_file(word_sample_path)
        word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        vocabulary_size = min(len(id2word), para.vocabulary_size)
        # vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
        rnn_dir = os.path.join(para.data_dir, "rnn_word_new_loss")
        rnn_model = os.path.join(rnn_dir, "rnn_model.ckpt")
    with tf.Session() as sess:
        model = RNNModel(
            save_path=rnn_dir, output_size=vocabulary_size, vocabulary_size=vocabulary_size, on_training=True)
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
            en_inputs, de_inputs, targets, mask = model.get_batches(data_set, id2vec_set=vec_set, id_mask=id_mask)
            _, step_loss = model.train(sess, en_inputs, de_inputs, targets, mask)
            step_time += (time.time() - start_time) / 20
            loss += step_loss / 20
            current_step += 1
            if current_step % 20 == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("[Seq2Seq model] global step %d learning rate %.4f step-time %.4f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                # model.saver.save(sess, rnn_model, global_step=model.global_step)
                step_time, loss = 0.0, 0.0


if __name__ == "__main__":
    # RNNModel(on_training=False)

    train_third_model()

    # vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    # word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    # vocabulary_size = min(len(id2word), 40000)
    # vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    # rnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "rnn")
    # rnn_model = os.path.join(rnn_dir, "rnn_model.ckpt")