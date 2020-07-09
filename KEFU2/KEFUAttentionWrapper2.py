# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _zero_state_tensors

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest


class HGFUAttentionWrapperState(
    collections.namedtuple("HGFUAttentionWrapperState",
                           ("cell_state", "attention", "time", "last_id",  "alignments",  "model_selector_openness",
                            "copy_alignments", "fact_alignments", "fact_memory_alignments", "decoding_mask",
                            "alignment_history", "attention_state",
                            ))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
    - `attention_state`: A single or tuple of nested objects
       containing attention mechanism state for each attention mechanism.
       The objects may contain Tensors or TensorArrays.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    The new state fields' shape must match original state fields' shape. This
    will be validated, and original fields' shape will be propagated to new
    fields.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    def with_same_shape(old, new):
      """Check and set new tensor's shape."""
      if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
        return tensor_util.with_same_shape(old, new)
      return new

    return nest.map_structure(
        with_same_shape,
        self,
        super(HGFUAttentionWrapperState, self)._replace(**kwargs))



class AttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               std_cell,
               cue_cell,
               cue_inputs,
               fact_candidates,
               lengths_for_fact_candidates,
               kgp_initial_goals,
               attention_mechanism,
               encoder_memory=None,
               encoder_memory_len=None,
               attention_layer_size=None,
               k_openness_history=False,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               common_word_projection=None,
               entity_predict_mode=False,
               copy_predict_mode=False,
               balance_gate=True,
               cue_fact_mode=0,
               cue_fact_mask=False,
               vocab_sizes=None,
               name=None,
               sim_dim=64,
               mid_projection_dim=1280,
               binary_decoding=False,
               attention_layer=None):
    """Construct the `AttentionWrapper`.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: A list of `AttentionMechanism` instances or a single
        instance.
      attention_layer_size: A list of Python integers or a single Python
        integer, the depth of the attention (output) layer(s). If None
        (default), use the context as attention at each time step. Otherwise,
        feed the context and cell output into the attention layer to generate
        attention at each time step. If attention_mechanism is a list,
        attention_layer_size must be a list of the same length. If
        attention_layer is set, this must be None.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the behavior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      name: Name to use when creating ops.
      attention_layer: A list of `tf.layers.Layer` instances or a
        single `tf.layers.Layer` instance taking the context and cell output as
        inputs to generate attention at each time step. If None (default), use
        the context as attention at each time step. If attention_mechanism is a
        list, attention_layer must be a list of the same length. If
        attention_layers_size is set, this must be None.

    Raises:
      TypeError: `attention_layer_size` is not None and (`attention_mechanism`
        is a list but `attention_layer_size` is not; or vice versa).
      ValueError: if `attention_layer_size` is not None, `attention_mechanism`
        is a list, and its length does not match that of `attention_layer_size`;
        if `attention_layer_size` and `attention_layer` are set simultaneously.
    """

    if entity_predict_mode or copy_predict_mode:
        assert common_word_projection is not None and vocab_sizes is not None

    self.balance_gate = balance_gate
    if vocab_sizes is not None:
        self._common_vocab_size, self._copy_vocab_size , self._entity_vocab_size = vocab_sizes
        self._vocab_sizes = vocab_sizes

    self._cue_fact_mode = cue_fact_mode
    self._cue_fact_mask = cue_fact_mask
    self._entity_predict_mode = entity_predict_mode
    self._copy_predict_mode = copy_predict_mode
    self._fact_candidates = fact_candidates
    self._lengths_for_fanct_candidates = lengths_for_fact_candidates
    self._batch_size = tf.shape(cue_inputs)[0]
    self._kg_initial_goals = kgp_initial_goals
    if kgp_initial_goals is not None:
        self.decoding_mask_template = kgp_initial_goals
    else:
        self.decoding_mask_template = tf.reduce_max(fact_candidates, -1)
    self._common_word_projection = common_word_projection
    self._encoder_memory = encoder_memory
    self._encoder_memory_len = encoder_memory_len
    self._sim_vec_dim = sim_dim

    self.mid_projection_dim = mid_projection_dim
    self._binary_decoding = binary_decoding

    if copy_predict_mode:
        self._transformed_encoder_memory = tf.layers.dense(self._encoder_memory, units=self._sim_vec_dim,
                                                           activation=tf.nn.tanh,
                                                           name='encoder_memory_transformed')


    super(AttentionWrapper, self).__init__(name=name)

    self.k_openness_history = k_openness_history
    rnn_cell_impl.assert_like_rnncell("cell", std_cell)
    rnn_cell_impl.assert_like_rnncell("cell", cue_cell)
    if isinstance(attention_mechanism, (list, tuple)):
      self._is_multi = True
      attention_mechanisms = attention_mechanism
      for attention_mechanism in attention_mechanisms:
        if not isinstance(attention_mechanism, attention_wrapper.AttentionMechanism):
          raise TypeError(
              "attention_mechanism must contain only instances of "
              "AttentionMechanism, saw type: %s"
              % type(attention_mechanism).__name__)
    else:
      self._is_multi = False
      if not isinstance(attention_mechanism, attention_wrapper.AttentionMechanism):
        raise TypeError(
            "attention_mechanism must be an AttentionMechanism or list of "
            "multiple AttentionMechanism instances, saw type: %s"
            % type(attention_mechanism).__name__)
      attention_mechanisms = (attention_mechanism,)

    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None and attention_layer is not None:
      raise ValueError("Only one of attention_layer_size and attention_layer "
                       "should be set")

    if attention_layer_size is not None:
      attention_layer_sizes = tuple(
          attention_layer_size
          if isinstance(attention_layer_size, (list, tuple))
          else (attention_layer_size,))
      if len(attention_layer_sizes) != len(attention_mechanisms):
        raise ValueError(
            "If provided, attention_layer_size must contain exactly one "
            "integer per attention_mechanism, saw: %d vs %d"
            % (len(attention_layer_sizes), len(attention_mechanisms)))
      self._attention_layers = tuple(
          layers_core.Dense(
              attention_layer_size,
              name="attention_layer",
              use_bias=False,
              dtype=attention_mechanisms[i].dtype)
          for i, attention_layer_size in enumerate(attention_layer_sizes))
      self._attention_layer_size = sum(attention_layer_sizes)
    elif attention_layer is not None:
      self._attention_layers = tuple(
          attention_layer
          if isinstance(attention_layer, (list, tuple))
          else (attention_layer,))
      if len(self._attention_layers) != len(attention_mechanisms):
        raise ValueError(
            "If provided, attention_layer must contain exactly one "
            "layer per attention_mechanism, saw: %d vs %d"
            % (len(self._attention_layers), len(attention_mechanisms)))
      self._attention_layer_size = sum(
          layer.compute_output_shape(
              [None,
               std_cell.output_size + mechanism.values.shape[-1].value])[-1].value
          for layer, mechanism in zip(
              self._attention_layers, attention_mechanisms))
    else:
      self._attention_layers = None
      self._attention_layer_size = sum(
          attention_mechanism.values.get_shape()[-1].value
          for attention_mechanism in attention_mechanisms)

    self._std_cell = std_cell
    self._cue_cell = cue_cell
    self._cue_inputs = cue_inputs
    self._attention_mechanisms = attention_mechanisms
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._alignment_history = alignment_history
    with ops.name_scope(name, "AttentionWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            final_state_tensor.shape[0].value
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing AttentionWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            self._batch_size_checks(state_batch_size, error_message)):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  def _batch_size_checks(self, batch_size, error_message):
    return [check_ops.assert_equal(batch_size,
                                   attention_mechanism.batch_size,
                                   message=error_message)
            for attention_mechanism in self._attention_mechanisms]

  def _item_or_tuple(self, seq):
    """Returns `seq` as tuple or the singular element.

    Which is returned is determined by how the AttentionMechanism(s) were passed
    to the constructor.

    Args:
      seq: A non-empty sequence of items or generator.

    Returns:
       Either the values in the sequence as a tuple if AttentionMechanism(s)
       were passed to the constructor as a sequence or the singular element.
    """
    t = tuple(seq)
    if self._is_multi:
      return t
    else:
      return t[0]

  @property
  def output_size(self):
    if self._output_attention:
      return self._attention_layer_size
    else:
      if self._common_word_projection is not None:
          return sum(self._vocab_sizes)

      return self._std_cell.output_size

  @property
  def state_size(self):
    """The `state_size` property of `AttentionWrapper`.
    Returns:
      An `AttentionWrapperState` tuple containing shapes used by this object.
    """
    return HGFUAttentionWrapperState(
        cell_state=self._std_cell.state_size,
        time=tensor_shape.TensorShape([]),
        last_id=tf.shape(tf.zeros([self._batch_size], dtype=tf.float32)),
        attention=self._attention_layer_size,
        model_selector_openness=tensor_shape.TensorShape([]),
        copy_alignments=tensor_shape.TensorShape([]),
        decoding_mask=tf.shape(self.decoding_mask_template),
        fact_alignments=tensor_shape.TensorShape([]),
        fact_memory_alignments=tensor_shape.TensorShape([]),
        alignments=self._item_or_tuple(
            a.alignments_size for a in self._attention_mechanisms),
        attention_state=self._item_or_tuple(
            a.state_size for a in self._attention_mechanisms),
        alignment_history=self._item_or_tuple(
            a.alignments_size if self._alignment_history else ()
            for a in self._attention_mechanisms))  # sometimes a TensorArray

  def zero_state(self, batch_size, dtype):
    """Return an initial (zero) state tuple for this `AttentionWrapper`.

    **NOTE** Please see the initializer documentation for details of how
    to call `zero_state` if using an `AttentionWrapper` with a
    `BeamSearchDecoder`.

    Args:
      batch_size: `0D` integer tensor: the batch size.
      dtype: The internal state data type.

    Returns:
      An `AttentionWrapperState` tuple containing zeroed out tensors and,
      possibly, empty `TensorArray` objects.

    Raises:
      ValueError: (or, possibly at runtime, InvalidArgument), if
        `batch_size` does not match the output size of the encoder passed
        to the wrapper object at initialization time.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._std_cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          self._batch_size_checks(batch_size, error_message)):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      initial_alignments = [
          attention_mechanism.initial_alignments(batch_size, dtype)
          for attention_mechanism in self._attention_mechanisms]


      step_scores2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      step_scores3 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      step_scores4 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      step_scores5 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

      return HGFUAttentionWrapperState(
          model_selector_openness=step_scores5,
          copy_alignments=step_scores2,
          fact_alignments=step_scores3,
          decoding_mask=tf.ones_like(self.decoding_mask_template),
          fact_memory_alignments=step_scores4,
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          last_id=tf.zeros([self._batch_size], dtype=tf.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._item_or_tuple(initial_alignments),
          attention_state=self._item_or_tuple(
              attention_mechanism.initial_state(batch_size, dtype)
              for attention_mechanism in self._attention_mechanisms),
          alignment_history=self._item_or_tuple(
              tensor_array_ops.TensorArray(
                  dtype,
                  size=0,
                  dynamic_size=True,
                  element_shape=alignment.shape)
              if self._alignment_history else ()
              for alignment in initial_alignments))

  def call(self, inputs, state):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `AttentionWrapperState`
         containing the state calculated at this time step.

    Raises:
      TypeError: If `state` is not an instance of `AttentionWrapperState`.
    """
    if not isinstance(state, HGFUAttentionWrapperState):
      raise TypeError("Expected state to be instance of AttentionWrapperState. "
                      "Received type %s instead."  % type(state))


    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.

    cell_state = state.cell_state
    model_selector_openness = state.model_selector_openness
    copy_alignments = state.copy_alignments
    fact_alignments = state.fact_alignments
    fact_memory_alignments = state.fact_memory_alignments
    batch_size = tf.shape(self._lengths_for_fanct_candidates)[0]
    next_last_id = state.last_id
    decoding_mask = state.decoding_mask

    # [batch, dim]
    std_cell_inputs = tf.concat([self._cell_input_fn(inputs, state.attention)], -1)
    with tf.variable_scope('std_gru'):
        outputs_std, cell_state = self._std_cell(std_cell_inputs, cell_state)
        cell_output = outputs_std
        next_cell_state = cell_state


    fact_embedding = self._fact_candidates
    maximium_candidate_num = tf.shape(fact_embedding)[1]
    # [batch, embed]

    dynamic_inputs_1 = tf.concat([outputs_std, inputs], -1)

    entity_update_scores_p1 = tf.layers.dense(dynamic_inputs_1, units=self._sim_vec_dim,
                                              activation=tf.nn.tanh,
                                               name='entity_query_projection')
    entity_update_scores_p2 =self._fact_candidates
    entity_update_scores_p1 = tf.expand_dims(entity_update_scores_p1, 1)
    entity_update_scores_p1 = tf.tile(entity_update_scores_p1, [1, maximium_candidate_num, 1])
    entity_update_scores = tf.reduce_sum(entity_update_scores_p1 * entity_update_scores_p2, -1)
    entity_update_mask = tf.sequence_mask(self._lengths_for_fanct_candidates, dtype=tf.float32)
    entity_update_mask = (1.0 - entity_update_mask) * -1e10
    entity_update_scores += entity_update_mask
    entity_logits = entity_update_scores


    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    with ops.control_dependencies(
        self._batch_size_checks(cell_batch_size, error_message)):
      cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    if self._is_multi:
      previous_attention_state = state.attention_state
      previous_alignment_history = state.alignment_history
    else:
      previous_attention_state = [state.attention_state]
      previous_alignment_history = [state.alignment_history]

    all_alignments = []
    all_attentions = []
    all_attention_states = []
    maybe_all_histories = []
    for i, attention_mechanism in enumerate(self._attention_mechanisms):
      attention, alignments, next_attention_state = attention_wrapper._compute_attention(
          attention_mechanism, cell_output, previous_attention_state[i],
          self._attention_layers[i] if self._attention_layers else None)
      alignment_history = previous_alignment_history[i].write(
          state.time, alignments) if self._alignment_history else ()

      all_attention_states.append(next_attention_state)
      all_alignments.append(alignments)
      all_attentions.append(attention)
      maybe_all_histories.append(alignment_history)

    attention = array_ops.concat(all_attentions, 1)


    cell_output_org = cell_output
    common_word_inputs = tf.concat([cell_output, attention, inputs], -1)
    if self.mid_projection_dim > -1:
        common_word_inputs = tf.layers.dense(common_word_inputs, self.mid_projection_dim, tf.nn.elu)
    common_word_logits = self._common_word_projection(common_word_inputs)
    common_probs = tf.nn.softmax(common_word_logits, -1)

    if self._entity_predict_mode or self._copy_predict_mode:
        selector_mask = [1.0, 1.0, 1.0]
        if self._copy_predict_mode:
            # encoder_memroy [batch, seq_len, embedding]
            batch_num = tf.shape(self._encoder_memory)[0]
            max_encoder_len = tf.shape(self._encoder_memory)[1]
            # => [batch,seq_lem, embedding]
            copy_input_p1 = tf.concat([cell_output, attention, inputs], -1)

            copy_layer1_p1 = tf.layers.dense(copy_input_p1, units=self._sim_vec_dim, activation=tf.nn.tanh,
                                          name='copy_query')
            copy_layer1_p2 = self._transformed_encoder_memory

            copy_layer1_p1 = tf.tile(tf.expand_dims(copy_layer1_p1, 1), [1, max_encoder_len, 1])

            copy_logits = tf.reduce_sum(copy_layer1_p1 * copy_layer1_p2, -1)

            copy_mask = tf.sequence_mask(self._encoder_memory_len, dtype=tf.float32)
            copy_mask = (1.0 - copy_mask) * -1e10
            copy_logits += copy_mask
            copy_probs = tf.nn.softmax(copy_logits, -1)

            # Padding to fix len

            padding_num = self._copy_vocab_size - tf.reduce_max(self._encoder_memory_len)
            padding_probs = tf.zeros([batch_num, padding_num])
            copy_probs = tf.concat([copy_probs, padding_probs], -1)

        else:
            copy_logits = tf.ones([batch_size, self._copy_vocab_size], tf.float32) * -1e10
            copy_probs = tf.zeros([batch_size, self._copy_vocab_size], tf.float32)
            selector_mask[1] = 0.0
        if self._entity_predict_mode:
            entity_logits = entity_logits
            # 一半一半
            if self._kg_initial_goals is not None:
                if self.balance_gate is True:
                    balance_selector_input = tf.concat([cell_output_org, attention, inputs], -1)
                    balance_selector = tf.layers.dense(balance_selector_input, 1, activation=tf.nn.sigmoid, name='entity_balance_selector')
                    entity_probs = tf.nn.softmax(entity_logits, -1) * balance_selector + self._kg_initial_goals * ( 1.0 - balance_selector)

                    if self.k_openness_history:
                        fact_alignments = fact_alignments.write(state.time, entity_probs)
                else:
                    entity_probs = tf.nn.softmax(entity_logits, -1) * 0.5 + self._kg_initial_goals * 0.5
            else:
                entity_probs = tf.nn.softmax(entity_logits, -1)

            entity_probs = tf.minimum(entity_probs, decoding_mask)
        else:
            selector_mask[2] = 0
            entity_probs = tf.zeros([batch_size, self._entity_vocab_size], tf.float32)


        mode_selector_input = tf.concat([cell_output_org, attention, inputs], -1)
        layer1 = tf.layers.dense(mode_selector_input, self._sim_vec_dim, use_bias=True, activation=tf.nn.relu,
                                 name='selector_l1')

        common_selector = tf.layers.dense(layer1, 1, use_bias=False, name='common_selector') + ((1.0 - selector_mask[0]) * -1e10)
        copy_selector =  tf.layers.dense(layer1, 1, use_bias=False, name='copy_selector')+ ((1.0 - selector_mask[1]) * -1e10)
        entity_selector = tf.layers.dense(layer1, 1, use_bias=False, name='entity_selector') + ((1.0 - selector_mask[2]) * -1e10)

        common_selector = tf.exp(common_selector)
        copy_selector = tf.exp(copy_selector)
        entity_selector = tf.exp(entity_selector)

        model_selector_openness = model_selector_openness.write(state.time,
                                                                    tf.concat([common_selector, copy_selector,
                                                                               entity_selector],
                                                                              -1))

        exp_sum = common_selector + copy_selector + entity_selector



        common_selector = common_selector / exp_sum
        copy_selector = copy_selector / exp_sum
        entity_selector = entity_selector / exp_sum

        if self._binary_decoding:
            new_common_selector = tf.where(tf.greater_equal(common_selector, entity_selector), common_selector+entity_selector, tf.zeros_like(common_selector))
            new_entity_selector = tf.where(tf.greater(entity_selector, common_selector), common_selector+entity_selector, tf.zeros_like(entity_selector))
            entity_probs = entity_probs * new_entity_selector
            common_probs = common_probs * new_common_selector
            copy_probs = copy_probs * copy_selector
        else:
            entity_probs = entity_probs * entity_selector
            common_probs = common_probs * common_selector
            copy_probs = copy_probs * copy_selector


        cell_output = tf.concat([common_probs, copy_probs, entity_probs], -1)

        if self._cue_fact_mask:
            max_id = tf.argmax(cell_output, -1)
            # [batch]
            is_entity = tf.greater_equal(max_id, self._common_vocab_size + self._copy_vocab_size)
            # [batch]
            abs_id = tf.maximum(max_id - self._common_vocab_size - self._copy_vocab_size, 0)
            # [batch, fact_num]
            mask_used_fact = tf.one_hot(abs_id, maximium_candidate_num, on_value=1e-20, off_value=1.0)
            mask_used_fact = tf.where(is_entity, mask_used_fact, tf.ones_like(mask_used_fact))
            decoding_mask = tf.minimum(mask_used_fact, decoding_mask)



        next_id = tf.to_int32(tf.argmax(cell_output, -1))
        next_last_id = tf.to_int32(next_id)
        def safe_log(inX):
            return tf.log(inX + 1e-20)

        cell_output = safe_log((cell_output))

    next_state = HGFUAttentionWrapperState(
        last_id=next_last_id,
        model_selector_openness=model_selector_openness,
        copy_alignments=copy_alignments,
        fact_alignments=fact_alignments,
        fact_memory_alignments=fact_memory_alignments,
        time=state.time + 1,
        cell_state=next_cell_state,
        decoding_mask=decoding_mask,
        attention=attention,
        attention_state=self._item_or_tuple(all_attention_states),
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(maybe_all_histories))




    return cell_output, next_state
