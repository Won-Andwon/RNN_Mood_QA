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
_WORD_SPLIT = re.compile("([.,!?\"':;，。！；“‘’”：)(])")
_DIGIT_RE = re.compile(r"\d")
_SPECIAL_CHAR = " \t\\,./;'\"" \
                "[]!`~@#$%^&*()_+-={}:<>?" \
                "~·！@#￥%……&*（）——【】：“”；’‘《》，。？、" \
                "abcdefghijklmnopqrstuvwxyz" \
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
                "」「＂"

# def basic_tokenizer(sentence):
#   words = []
#   for space_separated_fragment in sentence.strip().split():
#     words.extend(_WORD_SPLIT.split(space_separated_fragment))
#   return [w for w in words if w]


# 分词
def basic_tokenizer(sentence, chs=True, char_level=False):
  if char_level and chs:
    return [x for x in sentence.strip() if x not in _SPECIAL_CHAR]
  else:
    # 调用结巴的中文分词
    if chs:
      sentence = sentence.strip()
      words_list = jieba.cut(sentence, cut_all=False)
      return [x.lower() for x in words_list if x]
    # 非中文
    else:
      words = []
      for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
      return [w.lower() for w in words if w]


def create_character(character_path, data_path, max_character_size, tokenizer=None, normalize_digits=True):
  # data_path 是个路径的数组
  if not gfile.Exists(character_path):
    print("创建字符文件")
    chars = {}
    for path in data_path:
      with gfile.GFile(path, mode="r") as f:
        print("处理%s文件" % path)
        counter = 0
        for line in f:
          counter += 1
          if counter % 10000 == 0:
            print("处理%s行" % counter)
          line = tf.compat.as_text(line)
          characters = tokenizer(line) if tokenizer else basic_tokenizer(line, True, True)
          for ch in characters:
            char = _DIGIT_RE.sub(_DIG, ch) if normalize_digits else ch
            if char in chars:
              chars[char] += 1
            else:
              chars[char] = 1
      char_list = _START_VOCAB + sorted(chars, key=chars.get, reverse=True)
      if len(char_list) > max_character_size:
        char_list = char_list[:max_character_size]
      with gfile.GFile(character_path, mode="w") as char_file:
        for w in char_list:
          char_file.write(w + "\n")
  else:
    print("字符文件已存在，如果不对应请删除后再次运行。")


# def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
#                       tokenizer=None, normalize_digits=True):
#   if not gfile.Exists(vocabulary_path):
#     print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
#     vocab = {}
#     with gfile.GFile(data_path, mode="r") as f:
#       counter = 0
#       for line in f:
#         counter += 1
#         if counter % 10000 == 0:
#           print("  processing line %d" % counter)
#         line = tf.compat.as_text(line)
#         tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
#         for w in tokens:
#           word = _DIGIT_RE.sub("0", w) if normalize_digits else w
#           if word in vocab:
#             vocab[word] += 1
#           else:
#             vocab[word] = 1
#       vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
#       if len(vocab_list) > max_vocabulary_size:
#         vocab_list = vocab_list[:max_vocabulary_size]
#       with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
#         for w in vocab_list:
#           vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
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
                          tokenizer=None, normalize_digits=True):
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence, True, True)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(_DIG, w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
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
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


# def get_QA_train_set(directory):
#   train_path = os.path.join(directory, "train_record_")
#   if not (gfile.Exists(train_path +"A.txt") and gfile.Exists(train_path +"Q.txt")):
#     print(train_path+"A.txt"+"文件未找到")
#   return train_path
#
#
# def get_QA_dev_set(direcory):
#   dev_name = "dev_record_"
#   dev_path = os.path.join(direcory, dev_name)
#   if not (gfile.Exists(dev_path + "A.txt") and gfile.Exists(dev_path+ "Q.txt")):
#     print(dev_path + "A.txt"+"文件未找到")
#   return dev_path


# def prepare_QA_data(data_dir, Q_vocabulary_size, A_vocabulary_size, tokenizer=None):
#   train_path = get_QA_train_set(data_dir)
#   dev_path = get_QA_dev_set(data_dir)
#
#   from_train_path = train_path + "Q.txt"
#   to_train_path = train_path + "A.txt"
#   from_dev_path = dev_path + "Q.txt"
#   to_dev_path = dev_path + "A.txt"
#   return prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path,
#                       Q_vocabulary_size, A_vocabulary_size, tokenizer)


def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path,
                 character_size, tokenizer=None):
  char_path = os.path.join(data_dir, "character%d.txt" % character_size)
  create_character(char_path, [from_train_path, to_train_path], character_size, tokenizer)
  
  to_train_ids_path = to_train_path + (".ids%d" % character_size)
  from_train_ids_path = from_train_path + (".ids%d" % character_size)
  data_to_token_ids(to_train_path, to_train_ids_path, char_path, tokenizer)
  data_to_token_ids(from_train_path, from_train_ids_path, char_path, tokenizer)

  to_dev_ids_path = to_dev_path + (".ids%d" % character_size)
  from_dev_ids_path = from_dev_path + (".ids%d" % character_size)
  data_to_token_ids(to_dev_path, to_dev_ids_path, char_path, tokenizer)
  data_to_token_ids(from_dev_path, from_dev_ids_path, char_path, tokenizer)
  return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          char_path)

# def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path,
#                  from_vocabulary_size, to_vocabulary_size, tokenizer=None):
#   to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
#   from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
#   create_vocabulary(to_vocab_path, to_train_path, to_vocabulary_size, tokenizer)
#   create_vocabulary(from_vocab_path, from_train_path, from_vocabulary_size, tokenizer)
#
#   to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
#   from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
#   data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path, tokenizer)
#   data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, tokenizer)
#
#   to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
#   from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
#   data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
#   data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)
#
#   return (from_train_ids_path, to_train_ids_path,
#           from_dev_ids_path, to_dev_ids_path,
#           from_vocab_path, to_vocab_path)
