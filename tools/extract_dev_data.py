import os


def extract(source_path, target_file, interval=1000):
  if os.path.exists(source_path):
    f_reader = open(source_path, 'r', encoding="UTF-8")
    f_writer = open(target_file, 'w', encoding="UTF-8")
    counter = 1
    for line in f_reader:
      if counter % interval == 0:
        f_writer.write(line)
      counter += 1
  else:
    print("no this file")
    

extract(r"D:\Data\对话语料\chat_corpus-master\twitter_en.txt\chat_A.txt",
        r"D:\Data\对话语料\chat_corpus-master\twitter_en.txt\dev_chat_A.txt")
extract(r"D:\Data\对话语料\chat_corpus-master\twitter_en.txt\chat_Q.txt",
        r"D:\Data\对话语料\chat_corpus-master\twitter_en.txt\dev_chat_Q.txt")
extract(r"D:\Data\对话语料\chat_corpus-master\movie_subtitles_en.txt\movie_subtitles_en_A.txt",
        r"D:\Data\对话语料\chat_corpus-master\movie_subtitles_en.txt\dev_movie_subtitles_en_A.txt")
extract(r"D:\Data\对话语料\chat_corpus-master\movie_subtitles_en.txt\movie_subtitles_en_Q.txt",
        r"D:\Data\对话语料\chat_corpus-master\movie_subtitles_en.txt\dev_movie_subtitles_en_Q.txt")
extract(r"D:\Data\UNv1.0.en-zh\en-zh\UNv1.0.en-zh.en",
        r"D:\Data\UNv1.0.en-zh\en-zh\dev_UNv1.0.en-zh.en")
extract(r"D:\Data\UNv1.0.en-zh\en-zh\UNv1.0.en-zh.zh",
        r"D:\Data\UNv1.0.en-zh\en-zh\dev_UNv1.0.en-zh.zh")
extract(r"D:\Data\对话语料\stc_weibo_train_post",
        r"D:\Data\对话语料\dev_stc_weibo_train_post")
extract(r"D:\Data\对话语料\stc_weibo_train_response",
        r"D:\Data\对话语料\dev_stc_weibo_train_response")