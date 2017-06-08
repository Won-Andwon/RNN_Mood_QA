import os

QA_path = r"D:\Data\对话语料\chat_corpus-master\twitter_en.txt"
QA_name = "chat.txt"
filename_Q = "chat_Q.txt"
filename_A = "chat_A.txt"

file_QA = os.path.join(QA_path, QA_name)
file_Q = os.path.join(QA_path, filename_Q)
file_A = os.path.join(QA_path, filename_A)

if os.path.exists(file_QA):
  f_reader = open(file_QA, 'r', encoding="UTF-8")
  Q_writer = open(file_Q, 'w', encoding="UTF-8")
  A_writer = open(file_A, 'w', encoding="UTF-8")
  i = 1
  # try:
  for line in f_reader:
    
    # newline = bytearray(line)
    # for x in range(len(newline)):
    #   if newline[x] > 127:
    #     newline[x] = 0
    #
    # if i % 2 == 0:
    #   A_writer.write(str(newline, encoding = "utf-8"))
    # else:
    #   Q_writer.write(str(newline, encoding = "utf-8"))
    if i % 2 == 0:
      A_writer.write(line)
    else:
      Q_writer.write(line)
    i += 1
  # except:
  #   print(line)
  f_reader.close()
  Q_writer.close()
  A_writer.close()
else:
  print("no this file!")