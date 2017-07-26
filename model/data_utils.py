# 此处部分代码部分来源于
# https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate
# 有改动和添加 因为是源码改动 所以没有import 借鉴的部分的链接如上
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from tensorflow.python.platform import gfile
import tensorflow as tf
import jieba

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_DIG = "_DIG"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# ？！《》{}-——+=、 字符数字的去除
_WORD_SPLIT = re.compile("([.,!?\"':;，。；“‘’”：)(])")
_DIGIT_RE = re.compile(r"\d")
# 我去掉了符号?!？！ 这两个很常用且能表意
_SPECIAL_CHAR = " \t\\,./;'\"" \
                "[]`~@#$%^&*()_+-={}:<>" \
                "～~·@#￥%……&*（）——【】：“”；’‘《》，。、" \
                "abcdefghijklmnopqrstuvwxyz" \
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
                "ばれやうしヾヘてまにづらでノつゥヽすりっい｀がのかえよみんおねｯｸﾞ" \
                "①②③④ａｄｅеｇｈнｋｎｏοＱｔＵ🇮с×а" \
                "αｎΘ" \
                "︺︹」└「┍┑╯╰╭╮︶‿／＼〉|∠＞〉︿─＿￣＝◡⌒" \
                "Σ≤≥≧≦єεз←→↓≡⊂⊃∩∀⊙▽∇△∈≠›" \
                "＂　◉◕｡ಥ❁ω♡ㅂｌ－﹏´" \
                "ԅы•ⅱдлшوΠ�．●･ฅ°✧・□〃％℃＠☞☜✲ﾟ〜˘ˇé︴゜＃"
_MYDICT = r"../resource/mydict.txt"


def basic_tokenizer(sentence, chs=True, char_level=False):
    """
    基本分词工具
    :param sentence: 源语句
    :param chs: TRUE 汉语且未分词 FALSE 基于空格分词
    :param char_level: 是否基于字符
    :return: 词列表
    """
    if char_level:
        return [x for x in sentence.strip() if x not in _SPECIAL_CHAR]
    else:
        # 调用结巴的中文分词
        if chs:
            sentence = sentence.strip()
            words_list = jieba.cut(sentence, cut_all=False)
            return [x.lower() for x in words_list if x not in _SPECIAL_CHAR]
        # 非中文 基于空格分词
        else:
            words = []
            for space_separated_fragment in sentence.strip().split():
                words.extend(_WORD_SPLIT.split(space_separated_fragment))
            return [w.lower() for w in words if w]


def create_vocabulary(vocabulary_path, data_path_list,
                      max_character_size, tokenizer=None,
                      chs=True, char_level=False,
                      normalize_digits=True):
    """
    创建词汇表文件（若不存在）
    :param vocabulary_path: 存储位置
    :param data_path_list: 用于创建词汇表的文件路径列表
    :param max_character_size: 有效词汇数 保留词汇最大值
    :param tokenizer: 分词器
    :param chs: 中文模式（未分词中文）
    :param char_level: 基于字符
    :param normalize_digits: 把数字均替换为_DIG
    :return:
    """
    # data_path 是个路径的数组
    if not gfile.Exists(vocabulary_path):
        print("创建词汇表")
        vcb = {}
        for path in data_path_list:
            with gfile.GFile(path, mode="r") as f:
                print("处理%s文件" % path)
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 10000 == 0:
                        print("已处理%s行" % counter)
                    line = tf.compat.as_text(line)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line, chs, char_level)
                    for token in tokens:
                        word = _DIGIT_RE.sub(_DIG, token) if normalize_digits else token
                        if word in vcb:
                            vcb[word] += 1
                        else:
                            vcb[word] = 1
        vocab_list = _START_VOCAB + sorted(vcb, key=vcb.get, reverse=True)
        if len(vocab_list) > max_character_size:
            vocab_list = vocab_list[:max_character_size]
        with gfile.GFile(vocabulary_path, mode="w") as vocabulary_file:
            for w in vocab_list:
                vocabulary_file.write(w + "\n")
    else:
        print("词汇表已存在，如果不对应请删除后再次运行。")


def initialize_vocabulary(vocabulary_path):
    """
    从词汇表文件读取词汇表
    :param vocabulary_path:
    :return: dict[词，数字] and list[词]
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_text(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, chs=True, char_level=False,
                          normalize_digits=True):
    """
    返回由ID组成的句子
    :param sentence:
    :param vocabulary: 词汇表 [词， 数字]
    :param tokenizer:
    :param normalize_digits:
    :param chs:
    :param char_level:
    :return: 词汇ID的列表
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence, chs, char_level)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    return [vocabulary.get(_DIGIT_RE.sub(_DIG, w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, chs=True, char_level=False,
                      normalize_digits=True):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 10000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_text(line), vocab,
                                                      tokenizer,
                                                      chs=chs, char_level=char_level,
                                                      normalize_digits=normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
    else:
        print("Tokenizing data is existed.(%s)" % target_path)


def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path,
                 vocabulary_size, tokenizer=None, chs=True, char_level=False):
    """
    把训练文件夹下的文件转化成token id文件
    :param data_dir:
    :param from_train_path:
    :param to_train_path:
    :param from_dev_path:
    :param to_dev_path:
    :param vocabulary_size:
    :param tokenizer:
    :param chs:
    :param char_level:
    :return:    (对话先说训练id表示，对话应答训练id表示，
                对话先说测试id表示， 对话应答测试id表示，
                词汇表路径)
    """
    postfix = ".%s.%s.ids%d" % ("zh" if chs else "en", "ch" if char_level else "vcb", vocabulary_size)
    vocab_path = os.path.join(data_dir, "vocabulary%d.txt" % vocabulary_size)
    create_vocabulary(vocab_path, [from_train_path, to_train_path], vocabulary_size, tokenizer,
                      chs=chs, char_level=char_level)
    
    to_train_ids_path = to_train_path + postfix
    from_train_ids_path = from_train_path + postfix
    data_to_token_ids(to_train_path, to_train_ids_path, vocab_path, tokenizer, chs=chs, char_level=char_level)
    data_to_token_ids(from_train_path, from_train_ids_path, vocab_path, tokenizer, chs=chs, char_level=char_level)
    
    to_dev_ids_path = to_dev_path + postfix
    from_dev_ids_path = from_dev_path + postfix
    data_to_token_ids(to_dev_path, to_dev_ids_path, vocab_path, tokenizer, chs=chs, char_level=char_level)
    data_to_token_ids(from_dev_path, from_dev_ids_path, vocab_path, tokenizer, chs=chs, char_level=char_level)
    return (from_train_ids_path, to_train_ids_path,
            from_dev_ids_path, to_dev_ids_path,
            vocab_path)


def train_text_to_ids(para, vcb_sz=None, ch_lvl=None, chs_md=None):
    """
    将参数中的地址指定的文件转换成由id表示。
    :param vcb_sz:
    :param ch_lvl:
    :param chs_md:
    :return:
    """
    from_train = None
    to_train = None
    from_dev = None
    to_dev = None
    vocab_path = None
    vocabulary_size = para.vocabulary_size if vcb_sz is None else vcb_sz
    ch_level = para.ch_level if ch_lvl is None else ch_lvl
    chs_mode = para.chs_mode if chs_md is None else chs_md
    if para.from_train_data and para.to_train_data:
        from_train_data = para.from_train_data
        to_train_data = para.to_train_data
        from_dev_data = from_train_data
        to_dev_data = to_train_data
        if para.from_dev_data and para.to_dev_data:
            from_dev_data = para.from_dev_data
            to_dev_data = para.to_dev_data
        from_train, to_train, from_dev, to_dev, vocab_path = prepare_data(
            para.data_dir,
            from_train_data, to_train_data,
            from_dev_data, to_dev_data,
            vocabulary_size, chs=chs_mode, char_level=ch_level)
        return from_train, to_train, from_dev, to_dev, vocab_path
    else:
        print("Please specify the file path in %s" % para.data_dir)
        return from_train, to_train, from_dev, to_dev, vocab_path


def read_data_from_many_id_files(paths):
    data = []
    for filepath in paths:
        if gfile.Exists(filepath):
            with gfile.GFile(filepath, mode="r") as f:
                line = f.readline()
                while line:
                    data.extend([int(x) for x in line.split()])
                    data.append(EOS_ID)
                    line = f.readline()
        else:
            raise ValueError("file %s not found.", filepath)
    return data


def read_sentences_from_many_id_files(paths):
    sentence_list = []
    for filepath in paths:
        if gfile.Exists(filepath):
            with gfile.GFile(filepath, mode='r') as f:
                line = f.readline()
                # 若读出了一行
                while line:
                    # 这行数据转换成id的list
                    sentence = [int(x) for x in line.split()]
                    # 不为空则添加这行数据 保证没有空的数据
                    if len(sentence) > 0:
                        sentence_list.append(sentence)
                    # 下一行
                    line = f.readline()
        else:
            raise ValueError("file %s not found.", filepath)
    return sentence_list
