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
                           ("cell_state", "attention", "time", "alignments", "k_openness",
                            "alignment_history", "attention_state"))):
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
               attention_mechanism,
               attention_layer_size=None,
               k_openness_history=False,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               name=None,
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
      return self._cell.output_size

  @property
  def state_size(self):
    """The `state_size` property of `AttentionWrapper`.
    Returns:
      An `AttentionWrapperState` tuple containing shapes used by this object.
    """
    return HGFUAttentionWrapperState(
        cell_state=self._std_cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        k_openness=tensor_shape.TensorShape([]),
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

      step_scores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      # step_scores = step_scores.write(0, -1.0)
      return HGFUAttentionWrapperState(
          k_openness=step_scores,
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
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
    std_cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    k_openness = state.k_openness
    inputs_for_cue = tf.concat([state.attention, self._cue_inputs], axis=-1)
    with tf.variable_scope('std_gru'):
        outputs_std, cell_state= self._std_cell(std_cell_inputs, cell_state)
    with tf.variable_scope('cue_gru'):
        outputs_cue, cue_cell_state = self._cue_cell(inputs_for_cue, cell_state)

    transformed_hy = tf.layers.dense(outputs_std, units=self._std_cell.state_size, activation=tf.nn.tanh,
                                     use_bias=False, name='FusionGate_HY')
    transformed_hw = tf.layers.dense(outputs_cue, units=self._cue_cell.state_size, activation=tf.nn.tanh,
                                     use_bias=False, name='FusionGate_HW')
    k = tf.layers.dense(tf.concat([transformed_hy, transformed_hw], -1), units=self._cue_cell.state_size, activation=tf.nn.sigmoid,
                        use_bias=False, name='FusionGate_k')

    if self.k_openness_history:
        k_his = tf.reduce_mean(k, axis=-1)
        k_openness = k_openness.write(state.time, k_his)
    cell_output = k * outputs_std + (1.0 - k) * outputs_cue
    next_cell_state = k * cell_state + (1.0 - k) * cue_cell_state


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
    next_state = HGFUAttentionWrapperState(
        k_openness=k_openness,
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        attention_state=self._item_or_tuple(all_attention_states),
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(maybe_all_histories))

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state


class KEFUWrapperState(

  collections.namedtuple("KEFUWrapperState", ("cell_state", "time_step", "fusion_gate"))):

  def clone(self, **kwargs):
    def with_same_shape(old, new):
      """Check and set new tensor's shape."""

      if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
        return tensor_util.with_same_shape(old, new)

      return new

    return nest.map_structure(

      with_same_shape,

      self,

      super(KEFUWrapperState, self)._replace(**kwargs))

class KEFUWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, cell_ke, std_input_dim, entity_dim, input_entity_ids, cue_fact_embedding,
                 cue_fact_mode=False, force_mode=False, initial_cell_state=None,
                 name=None, export_gate_history=False):
      """
      Args:
          cell:
          encoder_states:
          encoder_input_ids:
          tgt_vocab_size:
          gen_vocab_size:
          encoder_state_size:
          initial_cell_state:
      """
      super(KEFUWrapper, self).__init__(name=name)

      self._cell = cell
      self._cell_ke = cell_ke
      self._initial_cell_state = initial_cell_state
      self._std_input_dim = std_input_dim
      self._entity_dim = entity_dim
      self._gate_history = export_gate_history

      self._force_mode = force_mode
      self._input_entity_ids = input_entity_ids

      self._cue_fact_embedding = cue_fact_embedding
      self._cue_fact_mode = cue_fact_mode



    def __call__(self, inputs, state, scope=None):
      if not isinstance(state, KEFUWrapperState):
        raise TypeError("Expected state to be instance of KEFUWrapper. "
                        "Received type %s instead." % type(state))

      cell_state = state.cell_state
      time_step = state.time_step
      fusion_gate = state.fusion_gate

      std_inputs = tf.slice(inputs, [0, 0], [-1, self._std_input_dim])
      if self._cue_fact_mode:
          ke_inputs = tf.concat([inputs, self._cue_fact_embedding], -1)
      else:
          ke_inputs = inputs  # tf.slice(inputs, [0, self._std_input_dim], [-1, -1])
      _, std_cell_state = self._cell(std_inputs, cell_state, scope)
      _, ke_cell_state = self._cell_ke(ke_inputs, cell_state, scope)

      transformed_hy = tf.layers.dense(std_cell_state, units=self._cell.state_size,
                                       use_bias=False, name='FusionGate_X')
      transformed_hw = tf.layers.dense(ke_cell_state, units=self._cell.state_size,
                                       use_bias=False, name='FusionGate_KE')

      k = tf.layers.dense(tf.concat([transformed_hy, transformed_hw, cell_state], -1), units=64,
                          activation=tf.nn.tanh, name='FusionGate_k1')
      k = tf.layers.dense(k, units=1,
                          activation=tf.sigmoid, name='FusionGate_k2')

      # k = tf.expand_dims(k, -1)

      fusion_gate = fusion_gate.write(time_step, tf.reduce_mean(tf.squeeze(k),-1))

      cell_state = k * std_cell_state + (1.0 - k) * ke_cell_state

      outputs = cell_state

      state = KEFUWrapperState(cell_state=cell_state, time_step=state.time_step + 1,fusion_gate=fusion_gate)

      return outputs, state

    @property
    def state_size(self):
      """size(s) of state(s) used by this cell.

          It can be represented by an Integer, a TensorShape or a tuple of Integers
          or TensorShapes.
      """
      return KEFUWrapperState(cell_state=self._cell.state_size
                               , time_step=tf.TensorShape([]),
                               fusion_gate=tensor_shape.TensorShape([])
      )

    @property
    def output_size(self):
      """Integer or TensorShape: size of outputs produced by this cell."""
      return self._cell.output_size

    def zero_state(self, batch_size, dtype):
      with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
        if self._initial_cell_state is not None:
          cell_state = self._initial_cell_state
        else:
          cell_state = self._cell.zero_state(batch_size, dtype)
        step_scores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        return KEFUWrapperState(cell_state=cell_state, fusion_gate=step_scores,
                                 time_step=tf.zeros([], dtype=tf.int32,))


