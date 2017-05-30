import os
import time

if not os.path.exists(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\消息记录简版.txt'):
    # 去除多余的、无意义的信息
    f_reader = open(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\全部消息记录o.txt', 'r', encoding='UTF-8')
    f_writer = open(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\消息记录（去除冗余信息）.txt', 'w', encoding='UTF-8')

    for line in f_reader:
        line = line.replace('(本消息由您的好友通过手机QQ发送，体验手机QQ请登录： http://mobile.qq.com/c )', '')
        line = line.replace('[表情]', '')
        line = line.replace('[图片]', '')
        line = line.replace('(来自手机QQ: http://mobile.qq.com/v/ )', '')
        line = line.replace('(来自手机QQ： http://mobile.qq.com/v/ )', '')
        line = line.replace('(来自手机QQ: http://mobile.qq.com )', '')
        line = line.replace('(来自手机QQ： http://mobile.qq.com/v/ )', '')
        line = line.replace('您好，我现在有事不在，一会再和您联系。', '')
        line = line.replace('(手机QQ可以视频聊天啦！ http://mobile.qq.com/v/ )', '')
        line = line.replace('(来自手机QQ： http://mobile.qq.com/c )', '')
        line = line.replace('(来自手机QQ2012 [Android]:语音对讲，高效沟通！)', '')
        line = line.replace('[QQ红包]我发了一个“口令红包”，升级手机QQ最新版就能抢啦！', '')
        line = line.replace('[QQ红包]我发了一个“口令红包”，请使用新版手机QQ查收红包。', '')
        line = line.replace('[QQ红包]请使用新版手机QQ查收红包。', '')
        line = line.replace('(来自手机QQ2012 [iPhone]:语音对讲，高效沟通！)', '')
        line = line.replace('(来自手机QQ2012 [Android] )', '')
        line = line.replace('(来自手机QQ2012 [iPhone] )', '')
        line = line.replace('[闪照]请使用新版手机QQ查看闪照。', '')
        line = line.replace('(本消息由您的好友通过手机3GQQ发送，体验3GQQ请登录: http://3g.qq.com)', '')
        line = line.replace('[QQ红包]请使用最新QQ PC版查收红包。', '')
        line = line.replace('[QQ红包]我发了一个“口令红包”，请使用最新QQ PC版查收红包。', '')
        f_writer.write(line)

    f_reader.close()
    f_writer.close()

    # 去除url （分享） 系统撤回
    f_reader = open(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\消息记录（去除冗余信息）.txt', 'r', encoding='UTF-8')
    f_writer = open(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\消息记录（去除冗余信息）2.txt', 'w', encoding='UTF-8')

    for line in f_reader:
        # 都没发现 写入
        if line.find('撤回了一条消息') == -1 and \
                        line.find('://') == -1:
            f_writer.write(line)

    f_reader.close()
    f_writer.close()

else:
    f_reader = open(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\消息记录简版.txt', 'r', encoding='UTF-8')
    f_writer = open(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\消息记录QA.txt', 'w', encoding='UTF-8')
    f_writer2 = open(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\消息记录A.txt', 'w', encoding='UTF-8')
    f_writer1 = open(r'C:\Users\Won Andwon\Desktop\研究内容\消息记录（对话语料）\消息记录Q.txt', 'w', encoding='UTF-8')

    # 常用代码 判断日期是否符合格式 以及是否是日期
    def is_valid_date(ymd, hms=None):
        try:
            if hms:
                timearray = time.strptime(ymd + ' ' + hms, "%Y-%m-%d %H:%M:%S")
            else:
                timearray = time.strptime(ymd, "%Y-%m-%d")
            return time.mktime(timearray)
        except:
            return False


    lineA = ""
    lineQ = ""
    QA = -1
    thisQA = -1
    timestamp = -1
    thistimestamp = -1

    for line in f_reader:

        # 跳过的行
        # 空行不能跳啊 有的时候回答或问句（中的一行）就是一个空白
        # if line == '\n':
        #     continue
        if line == '================================================================\n':
            continue
        if not line.find("消息分组:") == -1:
            continue
        if not line.find("消息对象:") == -1:
            # 换对象了 处理最后一次对话
            # 太长或空白不写
            if len(lineQ) > 150:
                lineQ = ""
            if len(lineA) > 150:
                lineA = ""
            if not lineA == "" or not lineQ == "":
                f_writer.write("Q:" + lineQ + '\n')
                f_writer1.write(lineQ + '\n')
                f_writer.write("A:" + lineA + '\n')
                f_writer2.write(lineA + '\n')
            QA = -1
            thisQA = -1
            timestamp = -1
            thistimestamp = -1
            continue

        # 去除前后换行
        line = line.strip('\n')
        # 判断是否是对话信息行 若是更新信息 若不是则为紧跟着的语句
        temp_line = line.strip('\n').split(' ')
        if len(temp_line) == 3 and is_valid_date(temp_line[0], temp_line[1]):
            thistimestamp = int(is_valid_date(temp_line[0], temp_line[1]))  # 秒级
            if temp_line[2] == "伊葭漪":
                thisQA = 1
            else:
                thisQA = 0
        else:
            # 此句我说 上句我说
            if thisQA == 1 and QA == 1:
                # 两句间超过6小时了
                if (thistimestamp - timestamp) > (6 * 3600):
                    # 太长或空白不写
                    if len(lineQ) > 150:
                        lineQ = ""
                    if len(lineA) > 150:
                        lineA = ""
                    if not lineA == "" or not lineQ == "":
                        f_writer.write("Q:" + lineQ + '\n')
                        f_writer1.write(lineQ + '\n')
                        f_writer.write("A:" + lineA + '\n')
                        f_writer2.write(lineA + '\n')
                    lineQ = ""
                    lineA = line
                # 接续
                else:
                    if lineA == "":
                        lineA = line
                    else:
                        if lineA[-1] == '。' or lineA[-1] == '.':
                            lineA += line
                        else:
                            lineA += ("。" + line)

                QA = thisQA
                timestamp = thistimestamp
            # 此句我说 上句他说（或没人说）
            elif thisQA == 1 and not QA == 1:
                lineA = line
                QA = thisQA
                timestamp = thistimestamp
            # 这句他人说
            if thisQA == 0:
                # 上句他人说
                if QA == 0:
                    # 超过6小时
                    if (thistimestamp - timestamp) > (6 * 3600):
                        # 太长或空白不写
                        if len(lineQ) > 150:
                            lineQ = ""
                        if len(lineA) > 150:
                            lineA = ""
                        if not lineA == "" or not lineQ == "":
                            f_writer.write("Q:" + lineQ + '\n')
                            f_writer1.write(lineQ + '\n')
                            f_writer.write("A:" + lineA + '\n')
                            f_writer2.write(lineA + '\n')
                        lineQ = line
                        lineA = ""
                    else:
                        if lineQ == "":
                            lineQ = line
                        else:
                            if lineQ[-1] == '。' or lineQ[-1] == '.':
                                lineQ += line
                            else:
                                lineQ += ("。" + line)
                    QA = thisQA
                    timestamp = thistimestamp
                if QA == 1:
                    # 此句对面说 上句我说
                    # 太长或空白不写
                    if len(lineQ) > 150:
                        lineQ = ""
                    if len(lineA) > 150:
                        lineA = ""
                    if not lineA == "" or not lineQ == "":
                        f_writer.write("Q:" + lineQ + '\n')
                        f_writer1.write(lineQ + '\n')
                        f_writer.write("A:" + lineA + '\n')
                        f_writer2.write(lineA + '\n')
                    lineQ = line
                    lineA = ""

                    QA = thisQA
                    timestamp = thistimestamp

    f_reader.close()
    f_writer.close()
    f_writer1.close()
    f_writer2.close()
