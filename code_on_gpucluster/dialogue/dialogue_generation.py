import data_utils
import data_normalize
import cnn_logic_module
import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_integer("vocabulary_size", 40000, "可识别词数")
tf.app.flags.DEFINE_integer("window_size", 20, "每多少，窗口大小")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("apply", False,
                            "Set to True for using.")
tf.app.flags.DEFINE_boolean("ch_level", False, "是否基于字符（字）")
tf.app.flags.DEFINE_boolean("chs_mode", True,
                            "是否采用中文分词（是否是汉语或已分词）")
# Word2Vec flags
tf.app.flags.DEFINE_integer("word_vector_neuron_size", 512,
                            "词向量长度，用于感知存储词汇信息的神经元个数。The embedding dimension size.")
tf.app.flags.DEFINE_integer("word2vec_epochs_to_train", 3,
                            "Number of epochs to train. Each epoch processes the training data once "
                            "completely.")
tf.app.flags.DEFINE_float("word2vec_learning_rate", 0.1, "Initial learning rate.")
tf.app.flags.DEFINE_integer("word2vec_num_neg_samples", 100,
                            "Negative samples per training example.")
tf.app.flags.DEFINE_integer("word2vec_batch_size", 128,
                            "Number of training examples processed per step "
                            "(size of a minibatch).")
tf.app.flags.DEFINE_integer("word2vec_num_skips", 4,
                            "每个词向四周找几个")
tf.app.flags.DEFINE_integer("word2vec_concurrent_steps", 12,
                            "The number of concurrent training steps.")
tf.app.flags.DEFINE_integer("word2vec_window_size", 2,
                            "The number of words to predict to the left and right "
                            "of the target word.")
tf.app.flags.DEFINE_float("word2vec_subsample", 1e-3,
                          "Subsample threshold for word occurrence. Words that appear "
                          "with higher frequency will be randomly down-sampled. Set "
                          "to 0 to disable.")
tf.app.flags.DEFINE_integer("word2vec_checkpoint_interval", 600,
                            "Checkpoint the model (i.e. save the parameters) every n "
                            "seconds (rounded up to statistics interval).")
tf.app.flags.DEFINE_integer("train_module", 0,
                            "训练分块，"
                            "0表示准备数据 变成id和向量表示；"
                            "1表示训练判断词汇模块；"
                            "2表示训练词汇成句模块")
# CNN flags
tf.app.flags.DEFINE_integer("cnn_sentence_size", 32, "the length of one sentence in cnn model.")


PARAMETERS = tf.app.flags.FLAGS


def train_module_0():
    # 所有文本用数字表示
    from_train_id, to_train_id, from_dev_id, to_dev_id, vocab_path = data_utils.train_text_to_ids(PARAMETERS)
    # 所有数字改用向量表示(Embedding)
    vector_path = data_normalize.train_ids_to_vectors(
        PARAMETERS, from_train_id, to_train_id, from_dev_id, to_dev_id, vocab_path)

    # cnn_logic_module.train_second_module(
    #     from_train_id, to_train_id, vocab_path, vector_path, PARAMETERS)


def train_module_1():
    # single training module CNN the second.
    from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
    to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
    vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    vocab_path = r"/home/chenyongchang/data/dia/vocabulary.txt"
    cnn_logic_module.train_second_module(
        from_train_id, to_train_id, vocab_path, vector_path, PARAMETERS)


def train_module_2():
    pass


def apply():
    return


def main(_):
    if PARAMETERS.apply:
        apply()
    else:
        if PARAMETERS.train_module == 0:
            train_module_0()
        elif PARAMETERS.train_module == 1:
            train_module_1()
        elif PARAMETERS.train_module == 2:
            train_module_2()
        else:
            raise ValueError("参数错误，无法判断训练哪一模块。")

if __name__ == "__main__":
    tf.app.run()
