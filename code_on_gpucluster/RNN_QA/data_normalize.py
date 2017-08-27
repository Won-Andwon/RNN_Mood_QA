"""
功能：
1、Word2Vec (Embedding)：TensorFlow skip-gram model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import data_utils
import math
import numpy as np
import os
import random
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.platform import gfile



class Word2VecModel(object):
    """
    Word2Vec (Embedding)：TensorFlow skip-gram model
    """

    def __init__(self,
                 session,
                 save_path,
                 embedding_size,
                 dictionary,
                 reverse_dictionary,
                 train_data=None,
                 vocabulary_size=40000,
                 learning_rate=0.1,
                 batch_size=128,
                 num_skips=2,
                 window_size=1,
                 epochs_to_train=3,
                 num_neg_samples=1000,
                 concurrent_steps=12,
                 subsample=1e-3,
                 checkpoint_interval=600,
                 sentence_levle=False,
                 saved_model=None):
        """
        参数设定
        :param session: 传入的会话
        :param embedding_size: The embedding dimension size. Embedding dimension.
        :param vocabulary_size: The number of words taked into account. Will be changed in fact.
        :param dictionary: Vocabulary
        :param reverse_dictionary: Reversed vocabulary
        :param train_data: Source data for training
        :param learning_rate: Initial learning rate.
        :param batch_size: Number of training examples processed per step (size of a minibatch).
        :param num_skip: Number of each words radiation
        :param window_size: The number of words to predict to the left and right of the target word.
        :param epochs_to_train: Number of epochs to train. Each epoch processes the training data once completely.
        :param save_path: Directory to write the model and training summaries.
        :param num_neg_samples: Negative samples per training example.
        :param concurrent_steps: The number of concurrent training steps.
        :param subsample: Subsample threshold for word occurrence.
                        Words that appear with higher frequency will be randomly down-sampled.
                        Set to 0 to disable.
        :param checkpoint_interval: Checkpoint the model (i.e. save the parameters)
                                    every n seconds (rounded up to statistics interval).
        :param sentence_levle: 原语料是不是基于句的，若是，则batch不定，以句子为准
        """
        self.session = session
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.train_data = train_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        assert batch_size % num_skips == 0
        self.num_skips = num_skips
        assert num_skips <= 2 * window_size
        self.window_size = window_size
        self.epochs_to_train = epochs_to_train
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.num_neg_samples = num_neg_samples
        self.concurrent_steps = concurrent_steps
        self.subsample = subsample
        self.checkpoint_interval = checkpoint_interval
        self.data_index = 0
        if sentence_levle:
            self.batch_size = None  # 改由句子长度决定
            self.build_graph_for_sentence()
        else:
            self.build_graph()
        # saved_model 现在接受ckpt（路径）为输入 通过checkpoint判断是否存在，然后再喂给恢复
        # 实际上并没有ckpt结尾的这个文件 而是4个文件
        if saved_model is not None and os.path.exists(
                os.path.join(os.path.dirname(saved_model), 'checkpoint')):
            self.saver.restore(self.session, saved_model)
        else:
            print("并没有读取呀，我的傻大宝！")

    def init_tf_var(self):
        """
        单独拿出来 因为可能不需要初始化（直接load），否则会导致无法读取模型。
        实际上并没有这个问题，多虑了，故而悬着以备后用
        :return:
        """
        # 执行初始化
        self.session.run(self.init)

    def generate_batch(self):
        batch = np.ndarray(shape=self.batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.window_size + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.train_data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.train_data)
        for i in range(self.batch_size // self.num_skips):
            target = self.window_size  # target label at the center of the buffer
            targets_to_avoid = [self.window_size]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.window_size]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.train_data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.train_data)
        return batch, labels

    def generate_batch_from_sentence(self, sentence):
        batch_inputs = []
        batch_labels = []

        sentence_len = len(sentence)
        for i in range(sentence_len):
            start = max(0, i - self.window_size)
            end = min(sentence_len, i + self.window_size + 1)
            for index in range(start, end):
                if index == i:
                    continue
                else:
                    input_id = sentence[i]
                    label_id = sentence[index]
                    if not (input_id and label_id):
                        continue
                    batch_inputs.append(input_id)
                    batch_labels.append(label_id)
        if len(batch_inputs) == 0:
            return
        batch_inputs = np.array(batch_inputs, dtype=np.int32)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])
        return batch_inputs, batch_labels

    def build_graph(self):
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        num_sampled = 64  # Number of negative examples to sample.

        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        self._loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=self.vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self._loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        self._train_inputs = train_inputs
        self._train_labels = train_labels
        self._similarity = similarity
        self._valid_size = valid_size
        self._valid_examples = valid_examples
        self._normalized_embeddings = normalized_embeddings
        self.saver = tf.train.Saver()

        # Add variable initializer.
        tf.global_variables_initializer().run()

    def build_graph_for_sentence(self):
        # 输入和标签（标准输出）的占位符
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        # vector保存
        self.embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
        )
        # 找到输入的向量
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)  # batch_size
        # NCE 权重 偏置
        self.nce_weight = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                          stddev=1.0 / math.sqrt(self.embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        # 得到NCE损失
        self._loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_weight,
                biases=self.nce_biases,
                labels=self.train_labels,
                inputs=embed,
                num_sampled=self.num_neg_samples,
                num_classes=self.vocabulary_size
            )
        )

        # 根据 nce loss 来更新梯度和embedding
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self._loss)  # 训练操作

        # 各词向量的L2模
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm

        # 变量初始化
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        self.session.run(self.init)

    def train(self):
        num_steps = len(self.train_data) // (self.batch_size // self.num_skips) * self.epochs_to_train + 1
        print(num_steps)

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = self.generate_batch()
            feed_dict = {self._train_inputs: batch_inputs, self._train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = self.session.run([self.optimizer, self._loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = self._similarity.eval()
                for i in xrange(self._valid_size):
                    valid_word = self.reverse_dictionary[self._valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        if nearest[k] < len(self.reverse_dictionary):
                            close_word = self.reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        self.final_embeddings = self._normalized_embeddings.eval()

    def train_by_sentence(self, sentence):
        if len(sentence) < 2:
            return
        batch_inputs, batch_labels = self.generate_batch_from_sentence(sentence)
        if batch_inputs is None:
            return
        feed_dict = {
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        }
        _, loss_val = self.session.run([self.optimizer, self._loss], feed_dict=feed_dict)
        return loss_val

    def get_ids_embeddings(self, word):
        valid_examples = self.dictionary[word]
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        valid_embeddings = tf.nn.embedding_lookup(
            self.normalized_embeddings, valid_dataset)
        return valid_embeddings.eval()

    def get_embedding(self, id):
        valid_dataset = tf.constant(id, dtype=tf.int32)
        valid_embeddings = tf.nn.embedding_lookup(
            self.normalized_embeddings, valid_dataset)
        return valid_embeddings.eval()

    def get_embeddings(self, id_list):
        valid_dataset = tf.constant(id_list, dtype=tf.int32)
        valid_embeddings = tf.nn.embedding_lookup(
            self.normalized_embeddings, valid_dataset)
        return valid_embeddings.eval()

    def test(self, word_list):
        """
        测试效果
        :param word_list:
        :return:
        """
        valid_examples = [self.dictionary[x] for x in word_list]
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        valid_embeddings = tf.nn.embedding_lookup(
            self.normalized_embeddings, valid_dataset)

        similarity = tf.matmul(
            valid_embeddings, self.normalized_embeddings, transpose_b=True)
        sim = similarity.eval()
        for i in xrange(len(word_list)):
            valid_word = self.reverse_dictionary[valid_examples[i]]
            top_k = 2  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % valid_word
            for k in xrange(top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)


def initialize_vectors(vector_path):
    """
    从文件读取id（词汇）对应的embedding
    :param vector_path:  存放位置
    :return: embedding_list
    """
    if os.path.exists(vector_path):
        embbedings = np.loadtxt(vector_path)
        return embbedings
    else:
        raise ValueError("路径错误，无此文件。")


def word2vec_test(model_dir, test_word_list):
    """
    看看效果怎么样
    :param model_dir: 模型存放位置
    :param test_word_list: 需要测试的词汇
    :return:
    """
    word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    vocabulary_size = len(id2word)
    with tf.Graph().as_default() as word2vec_graph:
        with tf.device("/cpu:0"), word2vec_graph.name_scope("Word2Vec"):
            with tf.Session() as sess:
                model = Word2VecModel(session=sess,
                                      save_path='/home/chenyongchang/data/dia/w2v',
                                      embedding_size=200,
                                      dictionary=word2id,
                                      reverse_dictionary=id2word,
                                      vocabulary_size=vocabulary_size,
                                      learning_rate=0.1,
                                      window_size=2,
                                      num_neg_samples=100,
                                      sentence_levle=True,
                                      saved_model=model_dir
                                      )
                model.test(test_word_list)
                # print(model.get_ids_embeddings('宿舍'))
                # np_embeddings = initialize_vectors(r'D:\Data\dia\vocabulary_vector40000.txt')
                # print(np_embeddings[word2id['宿舍']])
                # print(np_embeddings[word2id['啦']])
                # 实验证明，embedding不是按顺序存储的 tf占地更小 效率更高（可能是现算出来的）
                # 直接保存 embedding矩阵不合理 载入模型 用的时候算一下是合适的
                # 或者直接保存计算出来的结果（而不是embedding矩阵本身）


def train_ids_to_vectors(parameters, from_train_id, to_train_id, from_dev_id, to_dev_id, vocab_path):
    vec_path = os.path.join(parameters.train_dir,
                            "vocabulary_vector%d_test0.txt" % parameters.vocabulary_size)
    if os.path.exists(vec_path):
        print("embedding表已存在，如不符请删除后重新运行。")
        return vec_path

    word2id, id2word = data_utils.initialize_vocabulary(vocab_path)

    paths = [from_train_id, to_train_id, from_dev_id, to_dev_id]
    # # 读取所有数据并合一
    # # 旧版 将所有数据变成一行进行训练的方式 注意 WordVecModel的参数顺序在不同版本并不一致
    # data = data_utils.read_data_from_many_id_files(paths)
    #
    # with tf.Graph().as_default(), tf.Session() as session:
    #     with tf.device("/cpu:0"):
    #         word2vec_model = Word2VecModel(session,
    #                                        parameters.word_vector_neuron_size,
    #                                        parameters.vocabulary_size,
    #                                        word2id,
    #                                        id2word,
    #                                        data,
    #                                        parameters.word2vec_learning_rate,
    #                                        parameters.word2vec_batch_size,
    #                                        parameters.word2vec_num_skips,
    #                                        parameters.word2vec_window_size,
    #                                        parameters.word2vec_epochs_to_train,
    #                                        parameters.train_dir,
    #                                        parameters.word2vec_num_neg_samples,
    #                                        parameters.word2vec_concurrent_steps,
    #                                        parameters.word2vec_subsample,
    #                                        parameters.word2vec_checkpoint_interval
    #                                        )
    #         word2vec_model.train()
    #         word2vec_model.saver.save(session,
    #                                   os.path.join(parameters.train_dir, "w2v_model.ckpt"))
    #         vec_path = os.path.join(parameters.train_dir, "vocabulary_vector%d.txt" % parameters.vocabulary_size)
    #         np.savetxt(vec_path, word2vec_model.final_embeddings)
    #         # vec_path = os.path.join(parameters.train_dir, "vocabulary_vector%d.txt" % parameters.vocabulary_size)
    #         # with gfile.GFile(vec_path, mode="w") as vec_file:
    #         #     for w in embed_res:
    #         #         print(session.run([w]))
    #         #         vec_file.write(session.run([w]))
    #         #         vec_file.write("\n")

    # 新版 以句子为喂食单位进行运算
    data_sentence_list = data_utils.read_sentences_from_many_id_files(paths)
    # 默认图
    with tf.Graph().as_default() as word2vec_graph:
        # 在默认设备 cpu 0 上
        with tf.device("/cpu:0"):
            # 命名空间
            with word2vec_graph.name_scope("Word2Vec") as word2vec_scope:
                # 开个session 会话
                with tf.Session(graph=word2vec_graph) as word2vec_session:
                    # 一些参数的确定
                    vocabulary_size = min(len(id2word), parameters.vocabulary_size)
                    word2vec_dir = os.path.join(parameters.train_dir, "w2v_test0")
                    saved_model = os.path.join(word2vec_dir, "word2vec_model.ckpt")
                    # 建立模型咯 使用了部分参数
                    word2vec_model = Word2VecModel(session=word2vec_session,
                                                   save_path=word2vec_dir,
                                                   embedding_size=parameters.word_vector_neuron_size,
                                                   dictionary=word2id,
                                                   reverse_dictionary=id2word,
                                                   vocabulary_size=vocabulary_size,
                                                   learning_rate=parameters.word2vec_learning_rate,
                                                   window_size=parameters.word2vec_window_size,
                                                   num_neg_samples=parameters.word2vec_num_neg_samples,
                                                   sentence_levle=True,
                                                   saved_model=saved_model
                                                   )

                    # 训练
                    data_sentence_list_len = min(len(data_sentence_list), 180000)
                    num_steps = data_sentence_list_len * parameters.word2vec_epochs_to_train
                    print("[Word to Vector Model] Here are %d lines of sentence, so it will train %d steps."
                          % (data_sentence_list_len, num_steps))
                    average_loss = 0
                    for i in xrange(num_steps):
                        if data_sentence_list_len < 180000:
                            sentence = data_sentence_list[i % data_sentence_list_len]
                        else:
                            sentence = random.choice(data_sentence_list)
                        loss_var = word2vec_model.train_by_sentence(sentence)
                        average_loss += loss_var if loss_var is not None else 0
                        if i % (parameters.word2vec_checkpoint_interval * 10) == 0:
                            if i > 0:
                                average_loss /= (parameters.word2vec_checkpoint_interval * 10)
                            # The average loss is an estimate of the loss over the last 2000 batches.
                            print("[Word to Vector Model] Average loss at step ", i, ": ", average_loss)
                            # 误差 阈值 我直接设定了 可改程序从参数设定
                            if average_loss < 3:
                                print("[Word to Vector Model] Average loss has been low enough.")
                                break
                            average_loss = 0
                            # if i % (parameters.word2vec_checkpoint_interval * 10) == 0:
                            #     word2vec_model.test(
                            #         ['宿舍', '就是', '手机', '准备', '这么', '写', '乖', '开心', '看到', '你们'])

                    # 模型保存 通过tf方法
                    word2vec_model.saver.save(word2vec_session,
                                              saved_model)
                    # # embedding保存 不存了 费劲
                    # with gfile.GFile(vec_path, mode="w") as vec_file:
                    #     for word in id2word:
                    #         word_embedding = word2vec_model.get_ids_embedings(word)
                    #         vec_file.write(str(word_embedding) + '\n')
                    # 为了下一步 还是要按照ID顺序保存其Embedding
                    with gfile.GFile(vec_path, mode="w") as vec_file:
                        word_embeddings = word2vec_model.get_embeddings(
                            [x for x in range(vocabulary_size)])
                        for i in xrange(vocabulary_size):
                            if i % 200 == 0:
                                print("[Word to Vector Model] Has written %d words. " % i)
                            word_embedding = word_embeddings[i]

                            for _sp in word_embedding:
                                vec_file.write(str(_sp))
                                vec_file.write(" ")
                            vec_file.write("\n")
    return vec_path


if __name__ == '__main__':
    word2vec_test(r'/home/chenyongchang/data/dia/w2v/word2vec_model.ckpt',
                  [
                      '宿舍',
                      '就是',
                      # '手机',
                      # '准备',
                      '这么',
                      '什么',
                      # '知道',
                      '没有',
                      '想',
                      '吃饭',
                      '别人',
                      '我'
                      # '写',
                      # '乖',
                      # '开心',
                      # '看到',
                      # '你们'
                  ])
    file_reader = open(r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt", mode='r', encoding="UTF-8")
    # no = 82
    # linescounter = 0
    # for line in file_reader:
    #     if linescounter == no:
    #         print(line.strip(' \n'))
    #         print(len(line.split()))
    #     linescounter += 1

