"""
output an answer.
"""
import os
import math
import sys
import time
import random

import data_utils
import cnn_logic_module
import rnn_phraseology_module

import tensorflow as tf
import numpy as np
from six.moves import xrange

maximum_sequence = 8


def softmax_res(res):
    exp_res = np.exp(res)
    denominator = np.sum(exp_res)
    for i in range(exp_res.size):
        exp_res[i] /= denominator
    return exp_res


def get_top(seq2seq_res, cnn_recommend,
            word_rarity=None, word_mask=None,
            appeared=None, score_decay=None, softmax=True, no_eos=False):
    top_id = []
    top_score = []

    if word_mask is not None:
        mask = np.zeros([40000], dtype=np.float32)
        for each in word_mask:
            mask[each[0]] = each[1]
        seq2seq_res = np.multiply(seq2seq_res, mask)

    if word_rarity is not None:
        seq2seq_res = np.multiply(seq2seq_res, word_rarity)

    if softmax:
        seq2seq_res = softmax_res(seq2seq_res)

    if no_eos:
        candidates = cnn_recommend
    else:
        candidates = cnn_recommend + [data_utils.EOS_ID]

    sorted_res_id = (-seq2seq_res).argsort()
    serial = 0
    for ids in sorted_res_id:
        if serial >= maximum_sequence:
            break
        if appeared is None:
            if ids in candidates:
                top_id.append(ids)
                if score_decay is None:
                    top_score.append(seq2seq_res[ids])
                else:
                    top_score.append(seq2seq_res[ids] * math.pow(score_decay, serial))
                serial += 1
        else:
            if ids not in appeared:
                top_id.append(ids)
                if score_decay is None:
                    top_score.append(seq2seq_res[ids])
                else:
                    top_score.append(seq2seq_res[ids] * math.pow(score_decay, serial))
                serial += 1
    return top_id, top_score


def recommend_ori(sentence, para=None, only_best=False):
    if para is None:
        train_set_size = 221616
        vocabulary_size = 40000
        vocabulary_path = r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000.txt'
        vector_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary_vector40000.txt"
        word_frequency_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary40000_frequency.txt"
        word_mask_path = r"/home/chenyongchang/data/movie_subtitles_en/bin_group.txt"
        cnn_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "cnn_logic")
        cnn_statistics_path = os.path.join(cnn_dir, "label_material")
        cnn_saved_model = os.path.join(cnn_dir, "logic.ckpt")
        seq2seq_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "seq2seq")
        seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
    else:
        train_set_size = para.train_set_size
        vocabulary_size = para.vocabulary_size
        vocabulary_path = para.vocabulary_path
        vector_path = para.vector_path
        word_frequency_path = para.word_frequency_pathr
        word_mask_path = para.word_mask_path
        cnn_dir = os.path.join(para.data_dir, "bucket_cnn_tgth")
        cnn_statistics_path = os.path.join(cnn_dir, "label_material")
        cnn_saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
        seq2seq_dir = os.path.join(para.data_dir, "seq2seq")
        seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")

    word2id, id2word = data_utils.initialize_vocabulary(vocabulary_path)
    cnn_vocabulary_size = min(len(id2word), vocabulary_size)
    s2s_vocabulary_size = vocabulary_size
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    id2frq = data_utils.get_vocab_frq_with_no_truc(word_frequency_path, train_set_size)
    id2mask = data_utils.read_mask_data(word_mask_path)
    pos_vec_list = cnn_logic_module.build_position_vec(32, 512)

    word_rarity = np.zeros([40000], dtype=np.float32)
    for i in xrange(len(id2frq)):
        word_rarity[i] = id2frq[i]

    sentence_ids = data_utils.sentence_to_token_ids(tf.compat.as_text(sentence), word2id)

    candidates = []
    answer_ids_list = []
    for _ in xrange(maximum_sequence):
        answer_ids_list.append([])
    answer_scores = [0.0] * maximum_sequence

    with tf.Session() as cnn_sess:
        cnn_model = cnn_logic_module.CNNModel(batch_size=1,
                             vocabulary_size=vocabulary_size,
                             embedding_size=512,
                             save_path=cnn_dir)
        ckpt = tf.train.get_checkpoint_state(cnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            cnn_model.saver.restore(cnn_sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("no model to restore.")
        if os.path.exists(cnn_statistics_path):
            gene_data = data_utils.read_gene_data(cnn_statistics_path)
            cnn_model.statistic_words(sess=cnn_sess, gene_data=gene_data)
        else:
            raise ValueError("Pivotal File Missing.")

        cnn_recommend_id_list = cnn_model.generator(
                sess=cnn_sess, sentence_ids=sentence_ids, id2vec_set=vec_set, pos=pos_vec_list)

    cnn_recommend_id_list = list(cnn_recommend_id_list)

    with tf.Graph().as_default():
        with tf.Session() as seq2seq_sess:
            seq2seq_model = rnn_phraseology_module.Seq2SeqModel(
                vocab_size=s2s_vocabulary_size,
                save_path=seq2seq_dir,
                batch_size=1,
                forward_only=True)
            ckpt = tf.train.get_checkpoint_state(seq2seq_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                seq2seq_model.saver.restore(seq2seq_sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                seq2seq_sess.run(tf.global_variables_initializer())

            # having = 0
            start_time = time.time()
            for i in xrange(16):
                # if having == maximum_sequence:
                #     break
                if i == 0:
                    answer_ids = []
                    word_mask = id2mask[data_utils.GO_ID]
                    seq2seq_res = seq2seq_model.next_generator(
                        seq2seq_sess, sentence_ids, answer_ids, cnn_recommend_id_list)

                    top_id, top_score = get_top(
                        seq2seq_res,
                        cnn_recommend_id_list,
                        # word_mask=word_mask,
                        # word_rarity=word_rarity,
                        appeared=answer_ids
                    )

                    for podium in xrange(maximum_sequence):
                        answer_ids_list[podium].append(top_id[podium])
                        answer_scores[podium] += top_score[podium]
                else:
                    top_m_m_id_list = []
                    top_m_m_score = []
                    for k in xrange(maximum_sequence):
                        answer_ids = answer_ids_list[k]
                        word_mask = id2mask[answer_ids[-1]]
                        score = answer_scores[k]
                        seq2seq_res = seq2seq_model.next_generator(seq2seq_sess, sentence_ids, answer_ids,
                                                                   cnn_recommend_id_list)
                        top_id, top_score = get_top(
                            seq2seq_res,
                            cnn_recommend_id_list,
                            # word_mask=word_mask,
                            # word_rarity=word_rarity,
                            appeared=answer_ids)
                        # if data_utils.EOS_ID in answer_ids:
                        #     answer_this = answer_ids[:]
                        #     # answer_this.append(top_id[0])
                        #     top_m_m_id_list.append(answer_this)
                        #     top_m_m_score.append(score + top_score[0])
                        # else:
                        for podium in xrange(maximum_sequence):
                            answer_this = answer_ids[:]
                            answer_this.append(top_id[podium])
                            if top_id[podium] == data_utils.EOS_ID:
                                candidates.append(answer_this)
                            top_m_m_id_list.append(answer_this)
                            top_m_m_score.append(score + top_score[podium])
                    if len(top_m_m_id_list) > maximum_sequence * maximum_sequence \
                            or len(top_m_m_id_list) < maximum_sequence\
                            or len(top_m_m_score) > maximum_sequence * maximum_sequence \
                            or len(top_m_m_score) < maximum_sequence:
                        raise ValueError("Don't get enough candidates.")
                    else:
                        new_top = (-np.array(top_m_m_score)).argsort()[0:maximum_sequence]
                        for podium in xrange(maximum_sequence):
                            id_k = new_top[podium]
                            answer_ids_list[podium] = top_m_m_id_list[id_k]
                            answer_scores[podium] = top_m_m_score[id_k]
                        # if data_utils.EOS_ID in answer_ids_list[having]:
                        #     having += 1
                # for each in answer_ids_list:
                #     data_utils.show_id_list_word("", id2word, each, sp=" ")
                # print()

            expend = time.time() - start_time
            print("(耗时 %0.2f)" % expend)
            if only_best:
                each = answer_ids_list[0]
                if data_utils.EOS_ID in each:
                    eos_pos = each.index(data_utils.EOS_ID)
                    each = each[:eos_pos]
                data_utils.show_id_list_word("Answer to %s: " % sentence, id2word, each, sp=" ")
                # print(answer_ids_list)
            else:
                for podium in xrange(len(answer_ids_list)):
                    each = answer_ids_list[podium]
                    if data_utils.EOS_ID in each:
                        eos_pos = each.index(data_utils.EOS_ID)
                        each = each[:eos_pos]
                    data_utils.show_id_list_word("Answer[%d] to \"%s\": " % (podium, sentence), id2word, each, sp=" ")
                    # print(answer_scores)
                for each in candidates:
                    each.pop()  # delete eos character
                    data_utils.show_id_list_word("Answer to %s: " % sentence, id2word, each, sp=" ")


def recommend(sentence, para=None, only_best=False):
    if para is None:
        train_set_size = 221616
        vocabulary_size = 40000
        vocabulary_path = r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000.txt'
        vector_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary_vector40000.txt"
        word_frequency_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary40000_frequency.txt"
        word_mask_path = r"/home/chenyongchang/data/movie_subtitles_en/bin_group.txt"
        cnn_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "cnn_logic")
        cnn_statistics_path = os.path.join(cnn_dir, "label_material")
        cnn_saved_model = os.path.join(cnn_dir, "logic.ckpt")
        seq2seq_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "seq2seq")
        seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
    else:
        train_set_size = para.train_set_size
        vocabulary_size = para.vocabulary_size
        vocabulary_path = para.vocabulary_path
        vector_path = para.vector_path
        word_frequency_path = para.word_frequency_pathr
        word_mask_path = para.word_mask_path
        cnn_dir = os.path.join(para.data_dir, "bucket_cnn_tgth")
        cnn_statistics_path = os.path.join(cnn_dir, "label_material")
        cnn_saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
        seq2seq_dir = os.path.join(para.data_dir, "seq2seq")
        seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")

    word2id, id2word = data_utils.initialize_vocabulary(vocabulary_path)
    cnn_vocabulary_size = min(len(id2word), vocabulary_size)
    s2s_vocabulary_size = vocabulary_size
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    id2frq = data_utils.get_vocab_frq_with_no_truc(word_frequency_path, train_set_size)
    id2mask = data_utils.read_mask_data(word_mask_path)
    pos_vec_list = cnn_logic_module.build_position_vec(32, 512)

    word_rarity = np.zeros([40000], dtype=np.float32)
    for i in xrange(len(id2frq)):
        word_rarity[i] = id2frq[i]

    sentence_ids = data_utils.sentence_to_token_ids(tf.compat.as_text(sentence), word2id)

    candidates = []
    answer_ids_list = []
    for _ in xrange(maximum_sequence):
        answer_ids_list.append([])
    answer_scores = [0.0] * maximum_sequence

    with tf.Session() as cnn_sess:
        cnn_model = cnn_logic_module.CNNModel(batch_size=1,
                             vocabulary_size=vocabulary_size,
                             embedding_size=512,
                             save_path=cnn_dir)
        ckpt = tf.train.get_checkpoint_state(cnn_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            cnn_model.saver.restore(cnn_sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("no model to restore.")
        if os.path.exists(cnn_statistics_path):
            gene_data = data_utils.read_gene_data(cnn_statistics_path)
            cnn_model.statistic_words(sess=cnn_sess, gene_data=gene_data)
        else:
            raise ValueError("Pivotal File Missing.")

        cnn_recommend_id_list = cnn_model.generator(
                sess=cnn_sess, sentence_ids=sentence_ids, id2vec_set=vec_set, pos=pos_vec_list)

    cnn_recommend_id_list = list(cnn_recommend_id_list)

    with tf.Graph().as_default():
        with tf.Session() as seq2seq_sess:
            seq2seq_model = rnn_phraseology_module.Seq2SeqModel(
                vocab_size=s2s_vocabulary_size,
                save_path=seq2seq_dir,
                batch_size=1,
                forward_only=True)
            ckpt = tf.train.get_checkpoint_state(seq2seq_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                seq2seq_model.saver.restore(seq2seq_sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                seq2seq_sess.run(tf.global_variables_initializer())

            having = 0
            start_time = time.time()
            for i in xrange(16):
                if having == maximum_sequence:
                    break
                if i == 0:
                    answer_ids = []
                    word_mask = id2mask[data_utils.GO_ID]
                    seq2seq_res = seq2seq_model.next_generator(
                        seq2seq_sess, sentence_ids, answer_ids, cnn_recommend_id_list)

                    top_id, top_score = get_top(
                        seq2seq_res,
                        cnn_recommend_id_list,
                        word_mask=word_mask,
                        # word_rarity=word_rarity,
                        appeared=answer_ids)

                    for podium in xrange(maximum_sequence):
                        answer_ids_list[podium].append(top_id[podium])
                        answer_scores[podium] += top_score[podium]
                else:
                    top_m_m_id_list = []
                    top_m_m_score = []
                    for k in xrange(having, maximum_sequence):
                        answer_ids = answer_ids_list[k]
                        word_mask = id2mask[answer_ids[-1]]
                        score = answer_scores[k]
                        seq2seq_res = seq2seq_model.next_generator(seq2seq_sess, sentence_ids, answer_ids,
                                                                   cnn_recommend_id_list)
                        top_id, top_score = get_top(
                            seq2seq_res,
                            cnn_recommend_id_list,
                            word_mask=word_mask,
                            # word_rarity=word_rarity,
                            appeared=answer_ids)
                        if data_utils.EOS_ID in answer_ids:
                            answer_this = answer_ids[:]
                            # answer_this.append(top_id[0])
                            top_m_m_id_list.append(answer_this)
                            top_m_m_score.append(score + top_score[0])
                        else:
                            for podium in xrange(maximum_sequence):
                                answer_this = answer_ids[:]
                                answer_this.append(top_id[podium])
                                if top_id[podium] == data_utils.EOS_ID:
                                    candidates.append(answer_this)
                                top_m_m_id_list.append(answer_this)
                                top_m_m_score.append(score + top_score[podium])
                    if len(top_m_m_id_list) > (maximum_sequence - having) * maximum_sequence \
                            or len(top_m_m_id_list) < (maximum_sequence - having) \
                            or len(top_m_m_score) > (maximum_sequence - having) * maximum_sequence \
                            or len(top_m_m_score) < (maximum_sequence - having):
                        raise ValueError("Don't get enough candidates.")
                    else:
                        new_top = (-np.array(top_m_m_score)).argsort()[0:(maximum_sequence - having)]
                        for podium in xrange(having, maximum_sequence):
                            id_k = new_top[podium - having]
                            answer_ids_list[podium] = top_m_m_id_list[id_k]
                            answer_scores[podium] = top_m_m_score[id_k]
                        if data_utils.EOS_ID in answer_ids_list[having]:
                            having += 1
                # for each in answer_ids_list:
                #     data_utils.show_id_list_word("", id2word, each, sp=" ")
                # print()

            expend = time.time() - start_time
            print("(耗时 %0.2f)" % expend)
            if only_best:
                each = answer_ids_list[0]
                if data_utils.EOS_ID in each:
                    eos_pos = each.index(data_utils.EOS_ID)
                    each = each[:eos_pos]
                data_utils.show_id_list_word("Answer to %s: " % sentence, id2word, each, sp=" ")
                # print(answer_ids_list)
            else:
                for podium in xrange(len(answer_ids_list)):
                    each = answer_ids_list[podium]
                    if data_utils.EOS_ID in each:
                        eos_pos = each.index(data_utils.EOS_ID)
                        each = each[:eos_pos]
                    data_utils.show_id_list_word("Answer[%d] to \"%s\": " % (podium, sentence), id2word, each, sp=" ")
                    # print(answer_scores)
                for each in candidates:
                    each.pop()  # delete eos character
                    data_utils.show_id_list_word("Answer to %s: " % sentence, id2word, each, sp=" ")


def continuous_dialogue(para=None):
    if para is None:
        train_set_size = 88256
        vocabulary_size = 40000
        vocabulary_path = r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000.txt'
        vector_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary_vector40000.txt"
        word_frequency_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary40000_frequency.txt"
        word_mask_path = r"/home/chenyongchang/data/movie_subtitles_en/bin_group.txt"
        cnn_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "")
        cnn_statistics_path = os.path.join(cnn_dir, "label_material")
        cnn_saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt-200001")
        seq2seq_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "seq2seq")
        seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
    else:
        train_set_size = para.train_set_size
        vocabulary_size = para.vocabulary_size
        vocabulary_path = para.vocabulary_path
        vector_path = para.vector_path
        word_frequency_path = para.word_frequency_pathr
        word_mask_path = para.word_mask_path
        cnn_dir = os.path.join(para.data_dir, "bucket_cnn_tgth")
        cnn_statistics_path = os.path.join(cnn_dir, "label_material")
        cnn_saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
        seq2seq_dir = os.path.join(para.data_dir, "seq2seq")
        seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")

    word2id, id2word = data_utils.initialize_vocabulary(vocabulary_path)
    cnn_vocabulary_size = min(len(id2word), vocabulary_size)
    s2s_vocabulary_size = vocabulary_size
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    id2frq = data_utils.get_vocab_frq(word_frequency_path, train_set_size)
    id2mask = data_utils.read_mask_data(word_mask_path)
    pos_vec_list = cnn_logic_module.build_position_vec(32, 200)

    word_rarity = np.zeros([40000], dtype=np.float32)
    for i in xrange(len(id2frq)):
        word_rarity[i] = id2frq[i]

    cnn_graph = tf.Graph()
    s2s_graph = tf.Graph()

    with cnn_graph.as_default():
        cnn_sess = tf.Session(graph=cnn_graph)
        cnn_model = cnn_logic_module.CNNModel(
            vocabulary_size=cnn_vocabulary_size,
            save_path=cnn_dir)
        if os.path.exists(cnn_statistics_path):
            gene_data = data_utils.read_gene_data(cnn_statistics_path)
            cnn_model.statistic_words(gene_data=gene_data)
        else:
            raise ValueError("Pivotal File Missing.")

    with s2s_graph.as_default():
        s2s_sess = tf.Session(graph=s2s_graph)
        seq2seq_model = rnn_phraseology_module.Seq2SeqModel(
            vocab_size=s2s_vocabulary_size,
            save_path=seq2seq_dir,
            batch_size=1,
            forward_only=True)
        ckpt = tf.train.get_checkpoint_state(seq2seq_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            seq2seq_model.saver.restore(s2s_sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            s2s_sess.run(tf.global_variables_initializer())

    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        start_time = time.time()
        sentence_ids = data_utils.sentence_to_token_ids(tf.compat.as_text(sentence), word2id)
        maximum_sequence = 8
        answer_ids_list = [[], [], [], [], [], [], [], []]
        answer_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        cnn_recommend_id_list = cnn_model.generator(
            sentence, word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)

        cnn_recommend_id_list = list(cnn_recommend_id_list)

        for i in xrange(16):
            if i == 0:
                answer_ids = []
                word_mask = id2mask[data_utils.GO_ID]
                seq2seq_res = seq2seq_model.next_generator(
                    s2s_sess, sentence_ids, answer_ids, cnn_recommend_id_list)

                top_id, top_score = get_top(
                    seq2seq_res,
                    cnn_recommend_id_list,
                    word_mask=word_mask,
                    word_rarity=word_rarity,
                    appeared=answer_ids)

                for podium in xrange(maximum_sequence):
                    answer_ids_list[podium].append(top_id[podium])
                    answer_scores[podium] += top_score[podium]
            else:
                top_m_m_id_list = []
                top_m_m_score = []
                for k in xrange(maximum_sequence):
                    answer_ids = answer_ids_list[k]
                    word_mask = id2mask[answer_ids[-1]]
                    score = answer_scores[k]
                    seq2seq_res = seq2seq_model.next_generator(s2s_sess, sentence_ids, answer_ids,
                                                               cnn_recommend_id_list)
                    top_id, top_score = get_top(
                        seq2seq_res,
                        cnn_recommend_id_list,
                        word_mask=word_mask,
                        word_rarity=word_rarity,
                        appeared=answer_ids)
                    for podium in xrange(maximum_sequence):
                        answer_this = answer_ids[:]
                        answer_this.append(top_id[podium])
                        top_m_m_id_list.append(answer_this)
                        top_m_m_score.append(score + top_score[podium])
                if len(top_m_m_id_list) != maximum_sequence * maximum_sequence \
                        or len(top_m_m_score) != maximum_sequence * maximum_sequence:
                    raise ValueError("Don't get max * max candidates.")
                else:
                    new_top = (-np.array(top_m_m_score)).argsort()[0:maximum_sequence]
                    for podium in xrange(maximum_sequence):
                        id_k = new_top[podium]
                        answer_ids_list[podium] = top_m_m_id_list[id_k]
                        answer_scores[podium] = top_m_m_score[id_k]

        each = answer_ids_list[0]
        if data_utils.EOS_ID in each:
            eos_pos = each.index(data_utils.EOS_ID)
            each = each[:eos_pos]
        expend = time.time() - start_time
        data_utils.show_id_list_word("(耗时 %0.2f): " % expend, id2word, each, sp="")
        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

    cnn_sess.close()
    s2s_sess.close()


def in_bash_dialogue(para=None):
    if para is None:
        train_set_size = 88256
        vocabulary_size = 40000
        vocabulary_path = r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000.txt'
        vector_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary_vector40000.txt"
        word_frequency_path = r"/home/chenyongchang/data/movie_subtitles_en/vocabulary40000_frequency.txt"
        word_mask_path = r"/home/chenyongchang/data/movie_subtitles_en/bin_group.txt"
        cnn_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "cnn_logic")
        cnn_statistics_path = os.path.join(cnn_dir, "label_material")
        cnn_saved_model = os.path.join(cnn_dir, "logic.ckpt-200001")
        seq2seq_dir = os.path.join(r'/home/chenyongchang/data/movie_subtitles_en/', "seq2seq")
        seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
    else:
        train_set_size = para.train_set_size
        vocabulary_size = para.vocabulary_size
        vocabulary_path = para.vocabulary_path
        vector_path = para.vector_path
        word_frequency_path = para.word_frequency_pathr
        word_mask_path = para.word_mask_path
        cnn_dir = os.path.join(para.data_dir, "cnn_logic")
        cnn_statistics_path = os.path.join(cnn_dir, "label_material")
        cnn_saved_model = os.path.join(cnn_dir, "logic.ckpt")
        seq2seq_dir = os.path.join(para.data_dir, "seq2seq")
        seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")

    word2id, id2word = data_utils.initialize_vocabulary(vocabulary_path)
    cnn_vocabulary_size = min(len(id2word), vocabulary_size)
    s2s_vocabulary_size = vocabulary_size
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    id2frq = data_utils.get_vocab_frq_with_no_truc(word_frequency_path, train_set_size)
    id2mask = data_utils.read_mask_data(word_mask_path)
    pos_vec_list = cnn_logic_module.build_position_vec(32, 200)

    word_rarity = np.zeros([40000], dtype=np.float32)
    for i in xrange(len(id2frq)):
        word_rarity[i] = id2frq[i]

    cnn_graph = tf.Graph()
    s2s_graph = tf.Graph()

    with cnn_graph.as_default():
        cnn_sess = tf.Session(graph=cnn_graph)
        cnn_model = cnn_logic_module.CNNModel(
            session=cnn_sess,
            vocabulary_size=cnn_vocabulary_size,
            save_path=cnn_dir,
            id2frq=id2frq,
            saved_model=cnn_saved_model)
        if os.path.exists(cnn_statistics_path):
            gene_data = data_utils.read_gene_data(cnn_statistics_path)
            cnn_model.statistic_words(gene_data=gene_data)
        else:
            raise ValueError("Pivotal File Missing.")

    with s2s_graph.as_default():
        s2s_sess = tf.Session(graph=s2s_graph)
        seq2seq_model = rnn_phraseology_module.Seq2SeqModel(
            vocab_size=s2s_vocabulary_size,
            save_path=seq2seq_dir,
            batch_size=1,
            forward_only=True)
        ckpt = tf.train.get_checkpoint_state(seq2seq_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            seq2seq_model.saver.restore(s2s_sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            s2s_sess.run(tf.global_variables_initializer())

    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

    while sentence and sentence != "exit()":
        sentence_ids = data_utils.sentence_to_token_ids(tf.compat.as_text(sentence), word2id)
        maximum_sequence = 8
        candidates = []
        answer_ids_list = []
        for _ in xrange(maximum_sequence):
            answer_ids_list.append([])
        answer_scores = [0.0] * maximum_sequence

        cnn_recommend_id_list = cnn_model.generator(
            sentence, word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)

        cnn_recommend_id_list = list(cnn_recommend_id_list)

        having = 0
        start_time = time.time()
        for i in xrange(16):
            if having == maximum_sequence:
                break
            if i == 0:
                answer_ids = []
                word_mask = id2mask[data_utils.GO_ID]
                seq2seq_res = seq2seq_model.next_generator(
                    s2s_sess, sentence_ids, answer_ids, cnn_recommend_id_list)

                top_id, top_score = get_top(
                    seq2seq_res,
                    cnn_recommend_id_list,
                    word_mask=word_mask,
                    word_rarity=word_rarity,
                    appeared=answer_ids)

                for podium in xrange(maximum_sequence):
                    answer_ids_list[podium].append(top_id[podium])
                    answer_scores[podium] += top_score[podium]
            else:
                top_m_m_id_list = []
                top_m_m_score = []
                for k in xrange(having, maximum_sequence):
                    answer_ids = answer_ids_list[k]
                    word_mask = id2mask[answer_ids[-1]]
                    score = answer_scores[k]
                    seq2seq_res = seq2seq_model.next_generator(s2s_sess, sentence_ids, answer_ids,
                                                               cnn_recommend_id_list)
                    top_id, top_score = get_top(
                        seq2seq_res,
                        cnn_recommend_id_list,
                        word_mask=word_mask,
                        word_rarity=word_rarity,
                        appeared=answer_ids)
                    if data_utils.EOS_ID in answer_ids:
                        answer_this = answer_ids[:]
                        # answer_this.append(top_id[0])
                        top_m_m_id_list.append(answer_this)
                        top_m_m_score.append(score + top_score[0])
                    else:
                        for podium in xrange(maximum_sequence):
                            answer_this = answer_ids[:]
                            answer_this.append(top_id[podium])
                            if top_id[podium] == data_utils.EOS_ID:
                                candidates.append(answer_this)
                            top_m_m_id_list.append(answer_this)
                            top_m_m_score.append(score + top_score[podium])
                if len(top_m_m_id_list) > (maximum_sequence - having) * maximum_sequence \
                        or len(top_m_m_id_list) < (maximum_sequence - having) \
                        or len(top_m_m_score) > (maximum_sequence - having) * maximum_sequence \
                        or len(top_m_m_score) < (maximum_sequence - having):
                    raise ValueError("Don't get enough candidates.")
                else:
                    new_top = (-np.array(top_m_m_score)).argsort()[0:(maximum_sequence - having)]
                    for podium in xrange(having, maximum_sequence):
                        id_k = new_top[podium - having]
                        answer_ids_list[podium] = top_m_m_id_list[id_k]
                        answer_scores[podium] = top_m_m_score[id_k]
                    if data_utils.EOS_ID in answer_ids_list[having]:
                        having += 1

        # max_id = -1
        # max = 0
        # for podium in xrange(len(answer_ids_list)):
        #     each = answer_ids_list[podium]
        #     if data_utils.EOS_ID in each:
        #         eos_pos = each.index(data_utils.EOS_ID)
        #         each = each[:eos_pos]
        #         answer_ids_list[podium] = each
        #     if len(each) > max:
        #         max_id = podium
        which = random.choice(range(0, maximum_sequence-1))
        each = answer_ids_list[0]
        if data_utils.EOS_ID in each:
            eos_pos = each.index(data_utils.EOS_ID)
            each = each[:eos_pos]
        expend = time.time() - start_time
        data_utils.show_id_list_word("(耗时 %0.2f s): " % expend, id2word, each, sp="")

        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

    cnn_sess.close()
    s2s_sess.close()


if __name__ == "__main__":
    # vocabulary_path = r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000.txt'
    # word2id, id2word = data_utils.initialize_vocabulary(vocabulary_path)
    # data_utils.creat_times_of_words(
    #         [r'/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_Q.txt',
    #          r'/home/chenyongchang/data/movie_subtitles_en/movie_subtitles_en_A.txt'],
    #         id2word,r'/home/chenyongchang/data/movie_subtitles_en/vocabulary40000_frequency.txt')
    # answers = recommend("那我先去睡咯")
    # answers = recommend("说话啊")
    # answers = recommend("你在干嘛")
    # answers = recommend("我不是很懂")
    # answers = recommend("我觉得你还需要学习")
    # answers = recommend("行了再见啦")
    # answers = recommend("你在宿舍干嘛呢")


    answers = recommend("What's wrong?")

    # answers = recommend("Then that's all you had to say.")

    # answers = recommend("Right. See? You're ready for the quiz.")

    # answers = recommend("Then that's all you had to say.")
    # answers = recommend("I'm sorry.")
    # answers = recommend("I'm sorry.")
    # answers = recommend("Then that's all you had to say.")
    # answers = recommend("Then that's all you had to say.")
    # answers = recommend("Then that's all you had to say.")

    # answers = recommend("嗯在家呢")
    # answers = recommend("有意思")
    # answers = recommend("原来你的说话方式是这样的")
    # answers = recommend("有点儿意思")
    # answers = recommend("好玩儿！")
    # answers = recommend("还高冷起来了")
    # answers = recommend("参考参考，以后留用啦")
    # answers = recommend("中午好好休息")
    # answers = recommend("吃饭了吗")
    # answers = recommend("还可以，看这个结果")
    # answers = recommend("还可以，看这个结果", only_best=True)
    # answers = recommend("噗,竟然怼我！揍你", only_best=True)
    # answers = recommend("我现在要打死你", only_best=True)
    # answers = recommend("我不要理你了", only_best=True)
    # answers = recommend("我才不要来一个这么坏的机器人呢！", only_best=True)
    # answers = recommend("你还憨！你最坏了", only_best=True)
    # answers = recommend("你真是能厉害", only_best=True)
    # answers = recommend("我心情好差", only_best=True)
    # answers = continuous_dialogue()
    # answers = in_bash_dialogue()

    # answers = recommend("晚安咯")
    # answers = recommend("我就准备起来呢")
    # answers = recommend("你在实验室嘛")

# def continuous_dialogue(para=None):
#     if para is None:
#         train_set_size = 88256
#         vocabulary_size = 40000
#         vocabulary_path = r'/home/chenyongchang/data/dia/vocabulary40000.txt'
#         vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
#         word_frequency_path = r"/home/chenyongchang/data/dia/vocabulary40000_frequency.txt"
#         word_mask_path = r"/home/chenyongchang/data/dia/bin_group.txt"
#         cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "bucket_cnn_tgth")
#         cnn_statistics_path = os.path.join(cnn_dir, "label_material")
#         cnn_saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt-200001")
#         seq2seq_dir = os.path.join(r'/home/chenyongchang/data/dia/', "seq2seq")
#         seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
#     else:
#         train_set_size = para.train_set_size
#         vocabulary_size = para.vocabulary_size
#         vocabulary_path = para.vocabulary_path
#         vector_path = para.vector_path
#         word_frequency_path = para.word_frequency_pathr
#         word_mask_path = para.word_mask_path
#         cnn_dir = os.path.join(para.data_dir, "bucket_cnn_tgth")
#         cnn_statistics_path = os.path.join(cnn_dir, "label_material")
#         cnn_saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
#         seq2seq_dir = os.path.join(para.data_dir, "seq2seq")
#         seq2seq_saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
#
#     word2id, id2word = data_utils.initialize_vocabulary(vocabulary_path)
#     cnn_vocabulary_size = min(len(id2word), vocabulary_size)
#     s2s_vocabulary_size = vocabulary_size
#     vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
#     id2frq = data_utils.get_vocab_frq(word_frequency_path, train_set_size)
#     id2mask = data_utils.read_mask_data(word_mask_path)
#     pos_vec_list = logic_module_together.build_position_vec(32, 200)
#
#     cnn_graph = tf.Graph()
#     s2s_graph = tf.Graph()
#
#     with cnn_graph.as_default():
#         cnn_sess = tf.Session(graph=cnn_graph)
#         cnn_model = logic_module_together.CNNModel(
#             session=cnn_sess,
#             vocabulary_size=cnn_vocabulary_size,
#             save_path=cnn_dir,
#             id2frq=id2frq,
#             saved_model=cnn_saved_model)
#         if os.path.exists(cnn_statistics_path):
#             gene_data = data_utils.read_gene_data(cnn_statistics_path)
#             cnn_model.statistic_words(gene_data=gene_data)
#         else:
#             raise ValueError("Pivotal File Missing.")
#
#     with s2s_graph.as_default():
#         s2s_sess = tf.Session(graph=s2s_graph)
#         seq2seq_model = phraseology_module.Seq2SeqModel(
#             vocab_size=s2s_vocabulary_size,
#             save_path=seq2seq_dir,
#             batch_size=1,
#             forward_only=True)
#         ckpt = tf.train.get_checkpoint_state(seq2seq_dir)
#         if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#             print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
#             seq2seq_model.saver.restore(s2s_sess, ckpt.model_checkpoint_path)
#         else:
#             print("Created model with fresh parameters.")
#             s2s_sess.run(tf.global_variables_initializer())
#
#     sys.stdout.write("> ")
#     sys.stdout.flush()
#     sentence = sys.stdin.readline()
#     while sentence:
#         start_time = time.time()
#         sentence_ids = data_utils.sentence_to_token_ids(tf.compat.as_text(sentence), word2id)
#         maximum_sequence = 8
#         answer_ids_list = [[], [], [], [], [], [], [], []]
#         answer_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#
#         cnn_recommend_id_list = cnn_model.generator(
#             sentence, word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
#
#         cnn_recommend_id_list = list(cnn_recommend_id_list)
#
#         for i in xrange(16):
#             if i == 0:
#                 answer_ids = []
#                 word_mask = id2mask[data_utils.GO_ID]
#                 seq2seq_res = seq2seq_model.next_generator(
#                     s2s_sess, sentence_ids, answer_ids, cnn_recommend_id_list)
#
#                 top_id, top_score = get_top(
#                     seq2seq_res,
#                     cnn_recommend_id_list,
#                     word_mask=word_mask,
#                     word_rarity=id2frq,
#                     appeared=answer_ids)
#
#                 for podium in xrange(maximum_sequence):
#                     answer_ids_list[podium].append(top_id[podium])
#                     answer_scores[podium] += top_score[podium]
#             else:
#                 top_m_m_id_list = []
#                 top_m_m_score = []
#                 for k in xrange(maximum_sequence):
#                     answer_ids = answer_ids_list[k]
#                     word_mask = id2mask[answer_ids[-1]]
#                     score = answer_scores[k]
#                     seq2seq_res = seq2seq_model.next_generator(s2s_sess, sentence_ids, answer_ids,
#                                                                cnn_recommend_id_list)
#                     top_id, top_score = get_top(
#                         seq2seq_res,
#                         cnn_recommend_id_list,
#                         word_mask=word_mask,
#                         word_rarity=id2frq,
#                         appeared=answer_ids)
#                     for podium in xrange(maximum_sequence):
#                         answer_this = answer_ids[:]
#                         answer_this.append(top_id[podium])
#                         top_m_m_id_list.append(answer_this)
#                         top_m_m_score.append(score + top_score[podium])
#                 if len(top_m_m_id_list) != maximum_sequence * maximum_sequence \
#                         or len(top_m_m_score) != maximum_sequence * maximum_sequence:
#                     raise ValueError("Don't get max * max candidates.")
#                 else:
#                     new_top = (-np.array(top_m_m_score)).argsort()[0:maximum_sequence]
#                     for podium in xrange(maximum_sequence):
#                         id_k = new_top[podium]
#                         answer_ids_list[podium] = top_m_m_id_list[id_k]
#                         answer_scores[podium] = top_m_m_score[id_k]
#
#         each = answer_ids_list[0]
#         if data_utils.EOS_ID in each:
#             eos_pos = each.index(data_utils.EOS_ID)
#             each = each[:eos_pos]
#         expend = time.time() - start_time
#         data_utils.show_id_list_word("(耗时 %0.2f): " % expend, id2word, each, sp="")
#         print("> ", end="")
#         sys.stdout.flush()
#         sentence = sys.stdin.readline()
#
#     cnn_sess.close()
#     s2s_sess.close()
