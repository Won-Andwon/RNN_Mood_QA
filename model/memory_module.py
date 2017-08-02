# 如果熵增有逆过程，那一定是意识、理智、思维的产生和进化
# 宇宙绝大部分都是平衡的，熵增或许也有与之对应的过程，从信息量的变化的角度，或能推算产生智能需要耗费的“代价”

# 大脑没有乘法器，激活、抑制应为“累加”的结果
# 大脑不求和
# 大脑的识别应该更为灵活和分布式 比如 n个神经元 每个神经元有m种激活状态 那么，可以产生m的n次方个编码
# 本代码模拟神经元（机器） “陈泳昌”作为一个词汇 激活负责此词汇的神经元 在大脑中 应该是 (i, j){第ni个神经元在mj状态}
# 通过神经元之间的激活与抑制（递质传递） 以及阈值（接受器， acceptor）作用 确定下一时刻的神经整体网络状态（即联想、回忆等）
# 本文 每个词汇（字）用神经元特别表示 彼此之间的联系由“贡献度”表示，即a激活时，下一时刻对b的激活的贡献程度
# b的最终激活程度 由所有相连神经元的贡献度累加 贡献度为永久记忆
# 一个时刻后的激活状态，传入工作记忆
# 实际上不就是 one-hot编码么
# 记忆 编码-存储-提取 干扰与消退
# 重复是抑制遗忘，而不是学习
# 外界信息 瞬时记忆 短时记忆 长时记忆 永久记忆
# 陈述性记忆 运动记忆 环境记忆 情感记忆
# 语义记忆 情景记忆 程序记忆 自动记忆 情绪记忆 通路 提取（长时）记忆
# 海马回（短->长） 大脑皮层 松果体 杏仁核 纹状体
# 暂时储存装置 工作记忆


from __future__ import division

import tensorflow as tf
import numpy as np

import data_utils


# 矩阵编号i行,j列 第 memory_matrix_size * i + j 个单词
class memory_module(object):
  def __init__(self,
               memory_matrix_size=80,
               evaporation_rate=0.95,
               ndtype=np.float32,
               dtype=tf.float32):
    self.memory_matrix_size = memory_matrix_size
    self.memory_matrix = np.zeros([self.memory_matrix_size, self.memory_matrix_size],
                                  ndtype)
    self.link_memory_matrix_size = memory_matrix_size * memory_matrix_size
    self.link_memory_matrix = np.zeros([self.link_memory_matrix_size, self.link_memory_matrix_size],
                                       ndtype)
    self.evaporation_rate = evaporation_rate
  
  # window 示例 [1,..., 10]
  def step(self, window):
    for x in window:
      x = int(x)
      self.memory_matrix[x // self.memory_matrix_size, x % self.memory_matrix_size] += 1.0
    self.memory_matrix *= self.evaporation_rate

model = memory_module()
window_size = 10
with open(r"D:\Data\dia\train_record_A.txt.ids6400") as file:
  for line in file:
    line = line.strip('\n').split()
    groupnum = len(line) // window_size
    for i in range(groupnum):
      model.step(line[i*window_size:(i+1)*window_size-1])
    model.step(line[groupnum*window_size:])

print(model.memory_matrix)

