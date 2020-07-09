# Copyright 2017 Google Inc. All Rights Reserved.
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
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

from lib import vocab_utils


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
    pass


class CueWordBatchedInput(
    collections.namedtuple("CueWordBatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "first_half_target_input", "first_half_target_output",
                            "source_sequence_length",
                            "target_sequence_length", "first_half_target_sequence_length"))):
    pass

class GraphBatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input","e_source", "e_target_input", "edge", "vertices", "ver_len",
                            "target_output","e_target_output", "source_sequence_length",
                            "target_sequence_length"))):
    pass




def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 sub_token_num=1,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 shuffle = True,
                 reshuffle_each_iteration=True,
                 use_char_encode=False):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  if shuffle:
      src_tgt_dataset = src_tgt_dataset.shuffle(
          output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, (tf.size(src) // sub_token_num), tf.size(tgt_in)),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.cast(tf.minimum(num_buckets, bucket_id), tf.int64)

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
   tgt_seq_len) = (batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)

def get_type1_graph_iterator(
                 src_dataset,
                 tgt_dataset,
                 edge_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 sub_token_num=1,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 shuffle = True,
                 reshuffle_each_iteration=True,
                 use_char_encode=False):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, edge_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  if shuffle:
      src_tgt_dataset = src_tgt_dataset.shuffle(
          output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt, edge: (
          tf.string_split([src]).values, tf.string_split([tgt]).values, tf.string_split([edge]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt, edge: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))


  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt, edge: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                                  tf.cast(tgt_vocab_table.lookup(tgt), tf.int32),
                                  tf.string_to_number(edge, tf.int32),
                        ),

        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)


  def edge_to_matrix(edge, nodes):
      node_num = tf.size(nodes) + 1
      indices = tf.to_int64(tf.reshape(edge, [-1, 2]))
      values = tf.tile([1e10], [tf.size(indices) // 2])
      dense_shape = [node_num, node_num]
      sparse_tmp = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
      res = tf.sparse_tensor_to_dense(sparse_tmp) - 1e10
      return res

  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt, edge: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0),
                        edge_to_matrix(edge, src),
                             ),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out, edge: (
            src, tgt_in, tgt_out, (tf.size(src) // sub_token_num), tf.size(tgt_in), edge),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([]),  # tgt_len
            tf.TensorShape([None, None]),  # src_edge
        ),
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0, # tgt_len -- unused
            -1e10,
        ))

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len, edge):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.cast(tf.minimum(num_buckets, bucket_id), tf.int64)

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
   tgt_seq_len, edges) = (batched_iter.get_next())
  return GraphBatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      vertices=None,
      ver_len=None,
      e_source=None,
      e_target_input=None,
      e_target_output=None,
      edge=edges,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)

def get_type3_graph_iterator(
                 src_dataset,
                 tgt_dataset,
                 tgt_out_dataset,
                 vertex_datast,
                 vertex_len_dataset,
                 edge_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 vertex_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 sub_token_num=1,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 shuffle = True,
                 reshuffle_each_iteration=True,
                 use_char_encode=False):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, tgt_out_dataset, vertex_datast, vertex_len_dataset, edge_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  if shuffle:
      src_tgt_dataset = src_tgt_dataset.shuffle(
          output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt, tgt_out, vertex, v_len, edge: (
          tf.string_split([src]).values, tf.string_split([tgt]).values, tf.string_split([tgt_out]).values, tf.string_split([vertex]).values,
          tf.string_split([v_len]).values, tf.string_split([edge]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt, tgt_out, vertex, v_len, edge: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))


  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt, tgt_out,  vertex, v_len, edge: (
                                    tf.cast(src_vocab_table.lookup(src), tf.int32),
                                  tf.cast(tgt_vocab_table.lookup(tgt), tf.int32),
                                  tf.cast(tgt_vocab_table.lookup(tgt_out), tf.int32),
                                  tf.cast(vertex_vocab_table.lookup(src), tf.int32),
                                  tf.cast(vertex_vocab_table.lookup(tgt), tf.int32),
                                  tf.cast(vertex_vocab_table.lookup(tgt_out), tf.int32),
                                  tf.cast(vertex_vocab_table.lookup(vertex), tf.int32),
                                  tf.string_to_number(v_len, tf.int32),
                                  tf.string_to_number(edge, tf.int32),
                        ),

        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)


  def edge_to_matrix(edge, nodes):
      node_num = tf.size(nodes) + 1
      indices = tf.to_int64(tf.reshape(edge, [-1, 2]))
      values = tf.tile([1e10], [tf.size(indices) // 2])
      dense_shape = [node_num, node_num]
      sparse_tmp = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
      res = tf.sparse_tensor_to_dense(sparse_tmp) - 1e10
      return res

  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt, tgt_out, e_src, e_tgt, e_tgt_out, vertex, v_len, edge: (
                        src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt_out, [tgt_eos_id]), 0),
                        e_src,
                        tf.concat(([0], e_tgt), 0),
                        tf.concat((e_tgt_out, [0]), 0),
                        vertex,
                        tf.reshape(v_len, [-1]),
                        edge_to_matrix(edge, vertex),
                             ),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out, e_src, e_tgt, e_tgt_out, vertex, v_len, edge: (
            src, tgt_in, tgt_out,  e_src, e_tgt, e_tgt_out, (tf.size(src) // sub_token_num), tf.size(tgt_in), vertex, v_len, edge),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([]),  # tgt_len
            tf.TensorShape([None]),  # vertex
            tf.TensorShape([None]),  # v_len
            tf.TensorShape([None, None]),  # src_edge
        ),
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,
            0,
            0,
            0,  # src_len -- unused
            0, # tgt_len -- unused
            0, # --unk
            0,
            -1e10,
        ))

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3,unused_12, unused_22, unused_32, src_len, tgt_len, vertex, v_len, edge):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.cast(tf.minimum(num_buckets, bucket_id), tf.int64)

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids,
   e_src, e_tgt, e_tgt_out,
   src_seq_len,
   tgt_seq_len, vertices, ver_len,  edges) = (batched_iter.get_next())
  return GraphBatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      e_source= e_src,
      e_target_input = e_tgt,
      e_target_output = e_tgt_out,
      vertices= vertices,
      ver_len= ver_len,
      edge=edges,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)


def get_cue_word_iterator(
                 src_dataset,
                 tgt_dataset,
                 first_half_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 sub_token_num=1,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 shuffle = True,
                 reshuffle_each_iteration=True,
                 use_char_encode=False):


  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, first_half_dataset))
  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)

  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  if shuffle:
      src_tgt_dataset = src_tgt_dataset.shuffle(
          output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt, htgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values, tf.string_split([htgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt, htgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt, htgt: (src[:src_max_len], tgt, htgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  if tgt_max_len:
    # htgt 的长度一定保证了小于这个值
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt, htgt: (src, tgt[:tgt_max_len], htgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt, htgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(htgt), tf.int32),
                                ),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt, htgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0),
                        tf.concat(([tgt_sos_id], htgt), 0),
                        tf.concat((htgt, [tgt_eos_id]), 0),
                              ),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out, htgt_in, htgt_out: (
            src, tgt_in, tgt_out, htgt_in, htgt_out, tf.div(tf.size(src),sub_token_num), tf.size(tgt_in), tf.size(htgt_in)),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([]),  # tgt_len
            tf.TensorShape([])),  # htgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3,unused_4, unused_5, src_len, tgt_len, unused_6):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.cast(tf.minimum(num_buckets, bucket_id), tf.int64)

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, 
   htgt_input_ids, htgt_output_ids,
   src_seq_len, tgt_seq_len, htgt_seq_len) = (batched_iter.get_next())
  return CueWordBatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      first_half_target_input=htgt_input_ids,
      first_half_target_output=htgt_output_ids,
      first_half_target_sequence_length=htgt_seq_len,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)

