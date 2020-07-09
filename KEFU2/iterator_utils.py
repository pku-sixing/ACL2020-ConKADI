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

from KEFU2 import knowledge_utils
import tensorflow as tf

from lib import vocab_utils


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input", "fact", "fact_length",
                            "source_entity", "target_input_entity",
                            "target_output", "source_sequence_length", "target_sequence_length", "cue_fact", "neg_fact"))):
    pass

def get_iterator(src_dataset,
                 tgt_in_dataset,
                 tgt_out_dataset,
                 cue_fact_dataset,
                 neg_fact_dataset,
                 fact_dataset,
                 vocab_table,
                 meta_vocab_table,
                 entity_vocab_table,
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
                 shuffle=True,
                 reshuffle_each_iteration=True):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  src_eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)
  ent_pad_id = tf.cast(vocab_table.lookup(tf.constant(knowledge_utils.PAD_ENTITY)), tf.int32)
  tgt_sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_in_dataset, tgt_out_dataset, fact_dataset, cue_fact_dataset, neg_fact_dataset))
  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)

  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  if shuffle:
      print('Shuffling data is enabled')
      src_tgt_dataset = src_tgt_dataset.shuffle(
          output_buffer_size, random_seed, reshuffle_each_iteration)


  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out, fact, cue, neg: (
          tf.string_split([src]).values,
          tf.string_split([tgt_in]).values, tf.string_split([tgt_out]).values,
          tf.string_split([fact]).values, tf.string_split([cue]).values, tf.string_split([neg]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out, fact, cue, neg: (src, tgt_in, tgt_out, fact, cue, tf.random_shuffle(neg)[:tf.minimum(tf.size(cue), tf.size(neg))]),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt_in, tgt_out, fact, cue, neg: tf.logical_and(tf.size(src) > 0, tf.size(tgt_in) > 0))


  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out, fact, cue, neg: (src[:src_max_len], tgt_in, tgt_out, fact, cue, neg),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out, fact, cue, neg: (src, tgt_in[:tgt_max_len], tgt_out[:tgt_max_len], fact, cue, neg),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out, fact, cue, neg: (
                          tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(vocab_table.lookup(tgt_in), tf.int32), # union for input embedding
                          tf.cast(meta_vocab_table.lookup(tgt_out), tf.int32), # relative for output lables

                          tf.cast(entity_vocab_table.lookup(src), tf.int32),
                          tf.cast(entity_vocab_table.lookup(tgt_in), tf.int32), # union for input embedding

                          tf.string_to_number(fact, tf.int32),
                          tf.string_to_number(cue, tf.int32),
                          tf.string_to_number(neg, tf.int32),
                         ),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out, entity_src, entity_tgt_in, fact, cue, neg: (
                            src,
                            tf.concat(([tgt_sos_id], tgt_in), 0),
                            tf.concat((tgt_out, [tgt_eos_id]), 0),
                            entity_src,
                            tf.concat(([0], entity_tgt_in), 0),
                            fact,
                            cue,
                            neg,
                            ),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out, entity_src, entity_tgt_in,fact, cue, neg: (
            src,  tgt_in, tgt_out,entity_src, entity_tgt_in, fact, cue, neg, tf.div(tf.size(src), sub_token_num), tf.size(tgt_in), tf.size(fact)),
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
            tf.TensorShape([None]),  # fact
            tf.TensorShape([None]),  # cue
            tf.TensorShape([None]),  # neg
            tf.TensorShape([]),  # src_len
            tf.TensorShape([]),  # tgt_len
            tf.TensorShape([]),  # fact_len


        ),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,
            0,
            ent_pad_id,  # fact
            0,  # cue
            0,  # neg
            0,  # src_len -- unused
            0,  # src_len -- unused
            0,

        ))  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3,unused_4, entity_src, entity_tgt_in, cue, neg, src_len, tgt_len, fact_len):
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
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)

  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids,entity_src_ids, entity_tgt_in_ids, fact_ids, cue_id, neg_id, src_seq_len,
   tgt_seq_len, fact_len) = (batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_entity=entity_src_ids,
      target_input_entity=entity_tgt_in_ids,
      fact=fact_ids,
      source_sequence_length=src_seq_len,
      fact_length=fact_len,
      target_sequence_length=tgt_seq_len,
      cue_fact=cue_id,
      neg_fact=neg_id,
  )

