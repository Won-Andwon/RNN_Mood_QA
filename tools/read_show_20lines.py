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


def show_same_lines(filepath, whichline=0, mode='r', encoding="UTF-8", show_lines=20):
    res = []
    if mode not in ['r', 'rb']:
        print("need read mode.")
        return
    if not os.path.exists(filepath):
        print("No this file!")
    else:
        file_reader = open(filepath, mode=mode, encoding=encoding)
        linescounter = 0
        shown = 0
        source_line = ''
        for line in file_reader:
            line = line.strip('\n')
            if linescounter == whichline:
                source_line = line
                res.append(linescounter)
                print(source_line)
                shown += 1
            if linescounter > whichline and shown < show_lines:
                if source_line == line:
                    res.append(linescounter)
                    print(line)
                    shown += 1
            linescounter += 1
    return res


def show_lines_by_no(no, filepath, mode='r', encoding="UTF-8"):
    if mode not in ['r', 'rb']:
        print("need read mode.")
        return
    if not os.path.exists(filepath):
        print("No this file!")
    else:
        file_reader = open(filepath, mode=mode, encoding=encoding)
        linescounter = 0
        for line in file_reader:
            if linescounter in no:
                print(line.strip('\n'))
            linescounter += 1


def count_lines(filepath, mode='r', encoding="UTF-8"):
    if mode not in ['r', 'rb']:
        print("need read mode.")
        return
    linescounter = 0
    if not os.path.exists(filepath):
        print("No this file!")
    else:
        file_reader = open(filepath, mode=mode, encoding=encoding)
        for line in file_reader:
            linescounter += 1
    return linescounter

# read_show_some_lines(r"D:\Data\dialogue\stc_weibo_train_post", lines=10)
# print()
# read_show_some_lines(r"D:\Data\dialogue\stc_weibo_train_response", lines=10)
#
# no = show_same_lines(r"D:\Data\dialogue\stc_weibo_train_post", whichline=4)
# show_lines_by_no(no, r"D:\Data\dialogue\stc_weibo_train_response")
# read_show_some_lines(r"D:\Data\dialogue\cornell movie-dialogs corpus\movie_lines.txt")

# read_show_some_lines(r"D:\Data\dialogue\OpenSubData\OpenSubData")

# read_show_some_lines(r"D:\Data\text8\text8")

# read_show_some_lines(r"D:\Data\dia\vocabulary_vector40000.txt",lines=2)
# print(count_lines(r"D:\Data\dia\vocabulary_vector40000.txt"))
# print(count_lines(r"D:\Data\dia\vocabulary40000.txt"))

# read_show_some_lines(r'D:\Data\dia\train_record_Q.txt.zh.vcb.ids40000', lines=5)
#
# read_show_some_lines(r'D:\Data\dia\train_record_A.txt.zh.vcb.ids40000', lines=5)

# print(count_lines(r"D:\Data\dialogue\weibo\stc_weibo_train_post"))
# print(count_lines(r"D:\Data\dialogue\weibo\stc_weibo_train_response"))
# print(count_lines(r"D:\Data\dia\train_middle_Q.txt"))
# print(count_lines(r"D:\Data\dia\train_record_Q.txt"))
# print(count_lines(r"D:\Data\dia\train_record_Q.txt"))

# read_show_some_lines(r"D:\Data\dia\bin_group.txt", lines=13)
