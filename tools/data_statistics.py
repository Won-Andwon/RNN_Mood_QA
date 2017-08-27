import os


def count_line_sum(filepath):
  if os.path.exists(filepath):
    count = 0
    f_reader = open(filepath, 'r', encoding="UTF-8")
    for line in f_reader:
      count += 1
    print(count)
    return count
  else:
    print("no this file.")
    

# count_line_sum(r"D:\Data\dia\train_record_Q.txt")
count_line_sum(r"D:\Data\dialogue\stc_weibo_train_post")
count_line_sum(r"D:\Data\dialogue\stc_weibo_train_response")
# count_line_sum(r"D:\Data\对话语料\chat_corpus-master\twitter_en.txt\chat_Q.txt")
# count_line_sum(r"D:\Data\对话语料\chat_corpus-master\movie_subtitles_en.txt\movie_subtitles_en_Q.txt")
# 88256
# 4435959
# 377265
# 221616