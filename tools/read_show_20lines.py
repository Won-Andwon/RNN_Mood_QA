import os


def read_show_some_lines(filepath, mode='r', encoding='UTF-8', lines=20):
  if mode not in ['r', 'rb']:
    print("need read mode.")
    return
  if not os.path.exists(filepath):
    print("No this file!")
  else:
    file_reader = open(filepath, mode=mode, encoding=encoding)
    linescounter = 0
    for line in file_reader:
      if linescounter < lines:
        print(line.strip('\n'))
      else:
        break
      linescounter += 1
      
    file_reader.close()
    
read_show_some_lines(r"D:\Data\UNv1.0.en-zh\en-zh\UNv1.0.en-zh.zh", lines=10)
print()
read_show_some_lines(r"D:\Data\UNv1.0.en-zh\en-zh\UNv1.0.en-zh.en", lines=10)