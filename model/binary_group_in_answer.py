"""
无论哪种吧，二元组（很多情况）总是要的
可作为先验概率
"""
import os
import data_utils

import numpy as np
from six.moves import xrange

path = r"D:\Data\dia\train_record_A.txt.zh.vcb.ids40000"

if not os.path.exists(path):
    print("No this file!")
else:
    all_binary_group = {}
    word_times = {}
    file_reader = open(path, mode="r", encoding="UTF-8")
    for line in file_reader:
        line_ids = [int(x) for x in line.strip().split()]
        if len(line_ids) < 1:
            continue
        id_list = [data_utils.GO_ID]
        id_list.extend(line_ids)
        id_list.append(data_utils.EOS_ID)
        
        nums = len(id_list)
        for i in range(nums):
            x = id_list[i]
            if x in word_times:
                word_times[x] += 1
            else:
                word_times[x] = 1
        
        for i in range(nums - 1):
            x = (id_list[i], id_list[i + 1])
            if x in all_binary_group:
                all_binary_group[x] += 1
            else:
                all_binary_group[x] = 1
    
    group_by_id = {}
    for each in all_binary_group.keys():
        if each[0] in group_by_id:
            group_by_id[each[0]].append((each[1], all_binary_group[each]))
        else:
            t_dic = [(each[1], all_binary_group[each])]
            group_by_id[each[0]] = t_dic

    word2id, id2word = data_utils.initialize_vocabulary(r'D:\Data\dia\vocabulary40000.txt')
    word_num = len(id2word)

    f_writer = open(r'D:\Data\dia\bin_group.txt', 'w', encoding="UTF-8")
    v_writer = open(r'D:\Data\dia\bin_group_matrix.txt', 'w', encoding="UTF-8")
    w_writer = open(r'D:\Data\dia\bin_group_next.txt', 'w', encoding="UTF-8")
    
    for i in xrange(word_num):
        if i in group_by_id:
            hind_list = sorted(group_by_id[i])
            for each in hind_list:
                f_writer.write(str(each[0]))
                f_writer.write(":")
                f_writer.write(str(each[1] / word_times[i]))
                f_writer.write(" ")
                w_writer.write(str(each[0])+" ")
            f_writer.write("\n")
            w_writer.write("\n")
        else:
            f_writer.write("\n")
            w_writer.write("\n")
    # vec = np.zeros([40000], dtype=np.float32)
    # for i in xrange(word_num):
    #     if i in group_by_id:
    #         hind_list = sorted(group_by_id[i])
    #         for each in hind_list:
    #             vec[each[0]] = each[1] / word_times[i]
    #     for p in vec:
    #         v_writer.write(str(p) + " ")
    #     v_writer.write("\n")
    #     vec = np.zeros([40000], dtype=np.float32)
    # c = 0
    # print(word_times[1])
    # sums = 0
    # for each in sorted(group_by_id[1]):
    #     print(each)
    #     c += 1
    #     sums += each[1]
    #
    # print(c)
    # print(sums)

