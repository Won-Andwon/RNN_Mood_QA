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
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
indics = model.memory_matrix.argmax()
print(indics)

model.memory_matrix[indics // 80, indics % 80] = 0.0
