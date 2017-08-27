# 借鉴部分较多 保留权利声明
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Sequence-to-sequence model (no attention mechanism).
new sample output project
only one bucket (30 50)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import data_utils
import logic_module_together
import math
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops




class Seq2SeqModel(object):
    def __init__(self,
                 vocab_size=40000,
                 buckets=(16, 16),
                 size=512,
                 num_layers=3,
                 max_gradient_norm=5.0,
                 batch_size=20,
                 learning_rate=0.5,
                 learning_rate_decay_factor=0.9,
                 num_samples_word=100,
                 num_samples=200,
                 forward_only=False,
                 sentence_length=16,
                 save_path=None):
        self.save_path = save_path
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.vocab_size = vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.global_step = tf.Variable(0, trainable=False)
        self.neo_size = size
        self.num_layers = num_layers
        self.max_gradient_norm = max_gradient_norm
        self.num_samples_word = num_samples_word
        self.num_samples = num_samples
        self.forward_only = forward_only
        self.sentence_length = sentence_length
        self.build_graph()

        # def sampled_softmax_loss(self,
        #                          weights,
        #                          biases,
        #                          labels,
        #                          inputs,
        #                          batch_sample_word_id_list,
        #                          num_sampled,
        #                          num_classes,
        #                          num_true=1,
        #                          sampled_values=None,
        #                          remove_accidental_hits=True,
        #                          partition_strategy="mod",
        #                          name="my_sampled_softmax_loss"):
        #     with ops.name_scope(name, values=[weights, biases, inputs, labels]):
        #         for each in xrange(labels.shape[0]):
        #             one_label = labels[each]
        #             one_input = inputs[each]
        #             one_sample_word_id_list = batch_sample_word_id_list[each]
        #             if one_label in one_sample_word_id_list:
        #
        #
        #     return 0.0

        # if not isinstance(weights, list):
        #     weights = [weights]
        #
        # with ops.name_scope(name, "compute_sampled_logits",
        #                     weights + [biases, inputs, labels]):
        #     if labels.dtype != tf.int64:
        #         labels = math_ops.cast(labels, tf.int64)
        #
        #
        #     # Sample the negative labels.
        #     #   sampled shape: [num_sampled] tensor
        #     #   true_expected_count shape = [batch_size, 1] tensor
        #     #   sampled_expected_count shape = [num_sampled] tensor
        #     if sampled_values is None:
        #         sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
        #             true_classes=labels,
        #             num_true=num_true,
        #             num_sampled=num_sampled,
        #             unique=True,
        #             range_max=num_classes)
        #     # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        #     # pylint: disable=unpacking-non-sequence
        #     sampled, true_expected_count, sampled_expected_count = sampled_values
        #     # pylint: enable=unpacking-non-sequence
        #
        #     # labels_flat is a [batch_size * num_true] tensor
        #     # sampled is a [num_sampled] int tensor
        #     all_ids = array_ops.concat([labels_flat, sampled], 0)
        #
        #     # weights shape is [num_classes, dim]
        #     all_w = embedding_ops.embedding_lookup(
        #         weights, all_ids, partition_strategy=partition_strategy)
        #     all_b = embedding_ops.embedding_lookup(biases, all_ids)
        #     # true_w shape is [batch_size * num_true, dim]
        #     # true_b is a [batch_size * num_true] tensor
        #     true_w = array_ops.slice(
        #         all_w, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))
        #     true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
        #
        #     # inputs shape is [batch_size, dim]
        #     # true_w shape is [batch_size * num_true, dim]
        #     # row_wise_dots is [batch_size, num_true, dim]
        #     dim = array_ops.shape(true_w)[1:2]
        #     new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
        #     row_wise_dots = math_ops.multiply(
        #         array_ops.expand_dims(inputs, 1),
        #         array_ops.reshape(true_w, new_true_w_shape))
        #     # We want the row-wise dot plus biases which yields a
        #     # [batch_size, num_true] tensor of true_logits.
        #     dots_as_matrix = array_ops.reshape(row_wise_dots,
        #                                        array_ops.concat([[-1], dim], 0))
        #     true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
        #     true_b = array_ops.reshape(true_b, [-1, num_true])
        #     true_logits += true_b
        #
        #     # Lookup weights and biases for sampled labels.
        #     #   sampled_w shape is [num_sampled, dim]
        #     #   sampled_b is a [num_sampled] float tensor
        #     sampled_w = array_ops.slice(
        #         all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        #     sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])
        #
        #     # inputs has shape [batch_size, dim]
        #     # sampled_w has shape [num_sampled, dim]
        #     # sampled_b has shape [num_sampled]
        #     # Apply X*W'+B, which yields [batch_size, num_sampled]
        #     sampled_logits = math_ops.matmul(
        #         inputs, sampled_w, transpose_b=True) + sampled_b
        #
        #     if remove_accidental_hits:
        #         acc_hits = candidate_sampling_ops.compute_accidental_hits(
        #             labels, sampled, num_true=num_true)
        #         acc_indices, acc_ids, acc_weights = acc_hits
        #
        #         # This is how SparseToDense expects the indices.
        #         acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
        #         acc_ids_2d_int32 = array_ops.reshape(
        #             math_ops.cast(acc_ids, tf.int32), [-1, 1])
        #         sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
        #                                           "sparse_indices")
        #         # Create sampled_logits_shape = [batch_size, num_sampled]
        #         sampled_logits_shape = array_ops.concat(
        #             [array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)],
        #             0)
        #         if sampled_logits.dtype != acc_weights.dtype:
        #             acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
        #         sampled_logits += sparse_ops.sparse_to_dense(
        #             sparse_indices,
        #             sampled_logits_shape,
        #             acc_weights,
        #             default_value=0.0,
        #             validate_indices=False)
        #
        #     # Subtract log of Q(l), prior probability that l appears in sampled.
        #     true_logits -= math_ops.log(true_expected_count)
        #     sampled_logits -= math_ops.log(sampled_expected_count)
        #
        #     # Construct output logits and labels. The true labels/logits start at col 0.
        #     out_logits = array_ops.concat([true_logits, sampled_logits], 1)
        #     # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        #     # of ones. We then divide by num_true to ensure the per-example labels sum
        #     # to 1.0, i.e. form a proper probability distribution.
        #     out_labels = array_ops.concat([
        #         array_ops.ones_like(true_logits) / num_true,
        #         array_ops.zeros_like(sampled_logits)
        #     ], 1)

        # sampled_losses = nn_ops.softmax_cross_entropy_with_logits(labels=out_labels,
        #                                                           logits=out_logits)
        # # sampled_losses is a [batch_size] tensor.
        # return sampled_losses

    def sequence_loss_by_single(self, stc_logits, stc_targets,
                                stc_weights, batch_sample_word_id,
                                name=None):
        # print(stc_logits)
        # print(stc_targets)
        # print(stc_weights)
        # print(batch_sample_word_id)
        if len(stc_targets) != len(stc_logits) \
                or len(stc_weights) != len(stc_logits) \
                or len(batch_sample_word_id) != len(stc_logits):
            raise ValueError("Lengths of logits, weights, and targets must be the same "
                             "%d, %d, %d, %d."
                             "" % (len(stc_logits), len(stc_weights), len(stc_targets), len(batch_sample_word_id)))
        with ops.name_scope(name, "sequence_loss_by_example",
                            stc_logits + stc_targets + stc_weights + batch_sample_word_id):
            log_perp_list = []
            for logit, target, weight, word_ids in zip(
                    stc_logits, stc_targets, stc_weights, batch_sample_word_id):
                # print(logit)
                # print(target)
                # print(weight)
                # print(word_ids)
                # logit = tf.expand_dims(logit, 1)
                # print(logit)
                ex_logit = tf.expand_dims(logit, 1)
                tar_w = embedding_ops.embedding_lookup(
                    self.full_connect_w, target, partition_strategy="mod")
                tar_w = tf.expand_dims(tar_w, 1)
                tar_b = embedding_ops.embedding_lookup(
                    self.full_connect_b, target)
                tar_b = tf.expand_dims(tar_b, 1)
                # print(tf.reduce_sum(tf.multiply(ex_logit, tar_w), axis=-1).shape)
                true_logits = tf.reduce_sum(tf.multiply(ex_logit, tar_w), axis=-1) + tar_b
                # print(tar_w, tar_b, true_logits)
                all_w = embedding_ops.embedding_lookup(
                    self.full_connect_w, word_ids, partition_strategy="mod")
                all_b = embedding_ops.embedding_lookup(
                    self.full_connect_b, word_ids)

                # print(ex_logit.shape)
                # print(all_w.shape)
                # print(all_b.shape)
                # print(tf.reduce_sum(tf.multiply(ex_logit, all_w), axis=-1).shape)
                sample_logits = tf.reduce_sum(tf.multiply(ex_logit, all_w), axis=-1) + all_b
                # print(all_w, all_b, sample_logits)
                # print(true_logits, sample_logits)
                logits = array_ops.concat([true_logits, sample_logits], 1)
                # print(logits)
                labels = array_ops.concat(
                    [array_ops.ones_like(true_logits), array_ops.zeros_like(sample_logits)], 1)
                # print(labels)
                crossent = nn_ops.softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
                log_perp_list.append(crossent * weight)

            log_perps = math_ops.add_n(log_perp_list)
            # print(log_perps)
            total_size = math_ops.add_n(stc_weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
            # print(log_perps)
        return log_perps

    def sequence_loss(self, stc_logits, stc_targets,
                      stc_weights, batch_sample_word_id,
                      name=None):
        with ops.name_scope(name, "sequence_loss",
                            stc_logits + stc_targets + stc_weights + batch_sample_word_id):
            cost = math_ops.reduce_sum(
                self.sequence_loss_by_single(stc_logits=stc_logits,
                                             stc_targets=stc_targets,
                                             stc_weights=stc_weights,
                                             batch_sample_word_id=batch_sample_word_id))
            batch_size = array_ops.shape(stc_targets[0])[0]
            return cost / math_ops.cast(batch_size, cost.dtype)

    def build_graph(self):
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * self.learning_rate_decay_factor)

        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(self.neo_size) for _ in range(self.num_layers)])
        else:
            cell = tf.contrib.rnn.GRUCell(self.neo_size)

        self.full_connect_w = tf.get_variable("out_w", [self.vocab_size, self.neo_size], dtype=tf.float32)
        transpose_full_connect_w = tf.transpose(self.full_connect_w)
        self.full_connect_b = tf.get_variable("out_b", [self.vocab_size], dtype=tf.float32)
        output_projection = (transpose_full_connect_w, self.full_connect_b)

        # def sampled_loss(labels, logits):
        #     labels = tf.reshape(labels, [-1, 1])
        #     # tf.nn.sampled_softmax_loss()
        #     return tf.nn.sampled_softmax_loss(
        #         weights=full_connect_w,
        #         biases=full_connect_b,
        #         labels=labels,
        #         inputs=logits,
        #         # batch_sample_word_id_list=part_vocab_inputs,
        #         num_sampled=self.num_samples,
        #         num_classes=self.vocab_size)
        #
        # softmax_loss_function = sampled_loss

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        self.part_vocab_inputs = []

        for i in xrange(self.sentence_length):
            self.encoder_inputs.append(
                tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in xrange(self.sentence_length + 1):
            self.decoder_inputs.append(
                tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(
                tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
            self.part_vocab_inputs.append(
                tf.placeholder(shape=[None, self.num_samples + self.num_samples_word - 1],
                               dtype=tf.int32, name="part_vocab{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if self.forward_only:
            self.outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs[:self.sentence_length],
                self.decoder_inputs[:self.sentence_length],
                cell,
                num_encoder_symbols=self.vocab_size,
                num_decoder_symbols=self.vocab_size,
                embedding_size=self.neo_size,
                output_projection=output_projection,
                feed_previous=False,
                dtype=tf.float32)
            self.losses = self.sequence_loss(
                stc_logits=self.outputs,
                stc_targets=targets[:self.sentence_length],
                stc_weights=self.target_weights[:self.sentence_length],
                batch_sample_word_id=self.part_vocab_inputs[:self.sentence_length])

            if output_projection is not None:
                self.outputs = [tf.matmul(output, output_projection[0]) + output_projection[1]
                                for output in self.outputs]
        else:
            self.outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs[:self.sentence_length],
                self.decoder_inputs[:self.sentence_length],
                cell,
                num_encoder_symbols=self.vocab_size,
                num_decoder_symbols=self.vocab_size,
                embedding_size=self.neo_size,
                output_projection=output_projection,
                feed_previous=False,
                dtype=tf.float32)
            # 句子长度 个 output组成 outputs 其中 每个output是一个 [batch, 神经元个数]

            self.losses = self.sequence_loss(
                stc_logits=self.outputs,
                stc_targets=targets[:self.sentence_length],
                stc_weights=self.target_weights[:self.sentence_length],
                batch_sample_word_id=self.part_vocab_inputs[:self.sentence_length])

        params = tf.trainable_variables()
        if not self.forward_only:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

            gradients = tf.gradients(self.losses, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.gradient_norms = norm
            self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, batch_word_ids, forward_only):
        encoder_size, decoder_size = self.sentence_length, self.sentence_length

        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            input_feed[self.part_vocab_inputs[l].name] = batch_word_ids[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.gradient_norms,  # Gradient norm.
                           self.losses]  # Loss for this batch.
        else:
            output_feed = [self.losses]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, word_data, only_one=False):
        strict = -1 if only_one else 0
        data_size = len(data)
        another_size = len(word_data)
        if data_size != another_size:
            raise ValueError("读取的文件行数不一致。")

        encoder_size, decoder_size = self.sentence_length, self.sentence_length
        encoder_inputs, decoder_inputs = [], []
        sample_word_inputs = []

        for _ in xrange(self.batch_size):
            try:
                line_no = random.randint(0, data_size - 1)
                encoder_input, decoder_input = data[line_no]
                # print(len(encoder_input), len(decoder_input))
                while len(encoder_input) == 0 \
                        or len(encoder_input) > encoder_size \
                        or len(decoder_input) == strict \
                        or len(decoder_input) > (decoder_size - 1):
                    line_no = random.randint(0, data_size - 1)
                    encoder_input, decoder_input = data[line_no]
                sample_word_input = word_data[line_no]
            except IndexError:
                print("data's length is %d, but index is %d! Original length is %d"
                      % (len(data), line_no, data_size))
                raise IndexError
            if len(sample_word_input) != self.num_samples_word:
                print("有个句子给的推荐不符合长度。")
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

            # Sample words list input will be changed for longer
            t_w_list = random.sample(range(self.vocab_size), self.num_samples_word + self.num_samples)
            for w in sample_word_input:
                if w in t_w_list:
                    t_w_list.remove(w)
            t_w_list = t_w_list[:self.num_samples]

            sample_word_inputs.append(sample_word_input + t_w_list)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        batch_word_list = []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            sample_word_list = []
            for batch_idx in xrange(self.batch_size):
                # print(batch_idx, len(sample_word_inputs[batch_idx]))
                temp_sample_word_inputs = sample_word_inputs[batch_idx][:]
                if length_idx < (decoder_size - 1):
                    key_word = temp_sample_word_inputs[length_idx + 1]
                    if key_word in sample_word_inputs[batch_idx]:
                        temp_sample_word_inputs.remove(key_word)
                    else:
                        temp_sample_word_inputs = \
                            temp_sample_word_inputs[:(self.num_samples + self.num_samples_word - 1)]
                else:
                    temp_sample_word_inputs = \
                        temp_sample_word_inputs[:(self.num_samples + self.num_samples_word - 1)]
                if len(temp_sample_word_inputs) != self.num_samples + self.num_samples_word - 1:
                    raise ValueError("截取候选词汇出了问题。%d and %d." % (
                        len(temp_sample_word_inputs), self.num_samples + self.num_samples_word - 1))
                sample_word_list.append(temp_sample_word_inputs)
            batch_word_list.append(np.array(sample_word_list, dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_word_list

    def next_generator(self, sess, sentence_ids, answer_ids, sample_word_list):
        encoder_inputs, decoder_inputs, target_weights, word_sample = self.get_batch(
            [(sentence_ids, answer_ids)], [sample_word_list], only_one=True)
        _, _, output_logits = self.step(sess, encoder_inputs, decoder_inputs,
                                        target_weights, word_sample, True)
        return output_logits[len(answer_ids)][0]

    def generator(self, sess, sentence, word2id, sample_word_list, select="ALL", answer_ids=None, mask_list=None):
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_text(sentence), word2id)
        if answer_ids is None:
            encoder_inputs, decoder_inputs, target_weights, word_sample = self.get_batch(
                [(token_ids, [])], [sample_word_list], only_one=True)
        else:
            encoder_inputs, decoder_inputs, target_weights, word_sample = self.get_batch(
                [(token_ids, answer_ids)], [sample_word_list], only_one=True)

        _, _, output_logits = self.step(sess, encoder_inputs, decoder_inputs,
                                        target_weights, word_sample, True)
        if select == "ALL":
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            outputs = []
            if mask_list is None:
                for logit in output_logits:
                    logit = logit[0]
                    outputs.append((-logit).argsort()[0:30])
            else:
                for i in xrange(len(output_logits)):
                    logit = output_logits[i][0]
                    mask = np.zeros([40000], dtype=np.float32)
                    for each in mask_list[i]:
                        mask[each[0]] = 1
                    logit = np.multiply(logit, mask)
                    # mask_id = []
                    # for each in mask:
                    #     mask_id.append(each[0])
                    # for i in xrange(len(logit)):
                    #     if i not in mask_id:
                    #         logit[i] = 0
                    outputs.append((-logit).argsort()[0:30])

        elif select == "SAMPLE":
            outputs = []
            for logit in output_logits:
                logit = logit[0]
                new_logit = []

                first = (-logit).argsort()[0:5000]
                selected = []
                for i in first:
                    if i in sample_word_list:
                        selected.append(i)
                outputs.append(selected)
                # print(logit)
        else:
            raise ValueError(print("ALL or SAMPLE."))
        return outputs


def train_third_module(from_train_id, to_train_id, word_sample_path, vocab_path, para=None):
    if para is None:
        from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
        to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
        word_sample_path = r"/home/chenyongchang/data/dia/train_middle_Q.txt"
        data_set = data_utils.read_data_pure(from_train_id, to_train_id)
        data_word = data_utils.read_data_one_file(word_sample_path)
        word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
        vocabulary_size = min(len(id2word), 40000)
        seq2seq_dir = os.path.join(r'/home/chenyongchang/data/dia/', "seq2seq")
        saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
    else:
        from_train_id = from_train_id
        to_train_id = to_train_id
        word_sample_path = word_sample_path
        data_set = data_utils.read_data(from_train_id, to_train_id, sentence_size=para.cnn_sentence_size)
        data_word = data_utils.read_data_one_file(word_sample_path)
        word2id, id2word = data_utils.initialize_vocabulary(vocab_path)
        vocabulary_size = min(len(id2word), para.vocabulary_size)
        seq2seq_dir = os.path.join(para.data_dir, "seq2seq")
        saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
    with tf.Session() as sess:
        model = Seq2SeqModel(vocab_size=40000, save_path=seq2seq_dir)
        ckpt = tf.train.get_checkpoint_state(seq2seq_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        # Read data into buckets and compute their sizes.
        print("Reading data and get into training.")

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, word_sample = model.get_batch(data_set, data_word)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, word_sample, False)
            step_time += (time.time() - start_time) / 20
            loss += step_loss / 20
            current_step += 1
            if current_step % 20 == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("[Seq2Seq model] global step %d learning rate %.6f step-time %.6f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                # model.saver.save(sess, saved_model, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                # for bucket_id in xrange(len(_buckets)):
                #     if len(dev_set[bucket_id]) == 0:
                #         print("  eval: empty bucket %d" % (bucket_id))
                #         continue
                #     encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                #         dev_set, bucket_id)
                #     _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                #                                  target_weights, bucket_id, True)
                #     eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                #         "inf")
                #     print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                # sys.stdout.flush()


if __name__ == "__main__":
    # Seq2SeqModel()
    train_third_module(None, None, None, None, None)

    # from_train_id = r"/home/chenyongchang/data/dia/train_record_Q.txt.zh.vcb.ids40000"
    # to_train_id = r"/home/chenyongchang/data/dia/train_record_A.txt.zh.vcb.ids40000"
    vector_path = r"/home/chenyongchang/data/dia/vocabulary_vector40000.txt"
    # word_sample_path = r"/home/chenyongchang/data/dia/train_middle_Q.txt"
    # data_set = data_utils.read_data_pure(from_train_id, to_train_id)
    # data_word = data_utils.read_data_one_file(word_sample_path)
    word2id, id2word = data_utils.initialize_vocabulary(r'/home/chenyongchang/data/dia/vocabulary40000.txt')
    vocabulary_size = min(len(id2word), 40000)
    cnn_vocabulary_size = len(id2word)
    id2frq = data_utils.get_vocab_frq(r"/home/chenyongchang/data/dia/vocabulary40000_frequency.txt",
                                      88256)
    vec_set = data_utils.read_id2vec(vector_path, vocabulary_size)
    pos_vec_list = logic_module_together.build_position_vec(32, 200)
    cnn_dir = os.path.join(r'/home/chenyongchang/data/dia/', "bucket_cnn_tgth")
    statistics_path = os.path.join(cnn_dir, "label_material")
    cnn_saved_model = os.path.join(cnn_dir, "cnn_tgth_model.ckpt")
    seq2seq_dir = os.path.join(r'/home/chenyongchang/data/dia/', "seq2seq")
    saved_model = os.path.join(seq2seq_dir, "seq2seq_model.ckpt")
    id_mask_path = r"/home/chenyongchang/data/dia/bin_group.txt"
    id_mask = data_utils.read_mask_data(id_mask_path)
    sentence_list = [
        # "说那么多连一开始分享的快乐都没有了",
        # "我嗓子疼，随便吃点流食",
        "那我先去睡咯",
        # "聊的才开心呢",
        # "就陪父母看电视聊电视,然后散步走走,看景色",
        # "我觉得很一般",
        # "我们已经是好友啦，一起来聊天吧！",
        # "果然我们的想法都不一样感受也不容易理解",
        # "说那么多连一开始分享的快乐都没有了",
        # "刚买了一点点小零食 回来准备比赛"
    ]
    sample_list = []
    with tf.Session() as cnn_sess:
        cnn_model = logic_module_together.CNNModel(
            session=cnn_sess,
            vocabulary_size=cnn_vocabulary_size,
            save_path=cnn_dir,
            id2frq=id2frq,
            saved_model=r'/home/chenyongchang/data/dia/bucket_cnn_tgth/cnn_tgth_model.ckpt-200001')
        if os.path.exists(statistics_path):
            gene_data = data_utils.read_gene_data(statistics_path)
            cnn_model.statistic_words(gene_data=gene_data)
        for sentence in sentence_list:
            id_list = cnn_model.generator(
                sentence, word2id=word2id, id2word=id2word, id2vec_set=vec_set, pos=pos_vec_list)
            sample_list.append(list(id_list))

    with tf.Graph().as_default():
        with tf.Session() as seq2seq_sess:
            seq2seq_model = Seq2SeqModel(vocab_size=40000, save_path=seq2seq_dir, batch_size=1,
                                         forward_only=True)
            ckpt = tf.train.get_checkpoint_state(seq2seq_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                seq2seq_model.saver.restore(seq2seq_sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                seq2seq_sess.run(tf.global_variables_initializer())

            # sys.stdout.write("> ")
            # sys.stdout.flush()
            for i in xrange(len(sentence_list)):
                answer_ids = [word2id["抱抱"], word2id["午安"], word2id["宝贝儿"]]
                #
                # mask_list = [id_mask[data_utils.GO_ID]]
                # for ids in answer_ids:
                #     mask_list.append(id_mask[ids])
                # len_mask = len(mask_list)
                # if len_mask < seq2seq_model.sentence_length:
                #     mask_list.extend([id_mask[data_utils.GO_ID]] * (seq2seq_model.sentence_length - len_mask))

                res_id_list = seq2seq_model.generator(
                    seq2seq_sess, sentence_list[i], word2id, sample_list[i],
                    select="SAMPLE", answer_ids=answer_ids, mask_list=None)
                for each in res_id_list:
                    print(("Answer to %s : " % sentence_list[i]) + " ".join(
                        [tf.compat.as_str(id2word[output]) for output in each]))
                    #     while sentence:
                    #         # Get token-ids for the input sentence.
                    #         token_ids = data_utils.sentence_to_token_ids(tf.compat.as_text(sentence), en_vocab)
                    #         # Which bucket does it belong to?
                    #         bucket_id = len(_buckets) - 1
                    #         for i, bucket in enumerate(_buckets):
                    #             if bucket[0] >= len(token_ids):
                    #                 bucket_id = i
                    #                 break
                    #         else:
                    #             logging.warning("Sentence truncated: %s", sentence)
                    #
                    #         # Get a 1-element batch to feed the sentence to the model.
                    #         encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    #             {bucket_id: [(token_ids, [])]}, bucket_id)
                    #         # Get output logits for the sentence.
                    #         _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                    #                                          target_weights, bucket_id, True)
                    #         # This is a greedy decoder - outputs are just argmaxes of output_logits.
                    #         outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                    #         # If there is an EOS symbol in outputs, cut them at that point.
                    #         if data_utils.EOS_ID in outputs:
                    #             outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    #         # Print out French sentence corresponding to outputs.
                    #         print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
                    #         print("> ", end="")
                    #         sys.stdout.flush()
                    #         sentence = sys.stdin.readline()
