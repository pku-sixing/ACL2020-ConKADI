
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util
import tensorflow as tf


class FactAwareWrapperState(

  collections.namedtuple("FactAwareWrapperState", ("cell_state", "last_ids", "time_step", "selector_state"))):

  def clone(self, **kwargs):
    def with_same_shape(old, new):
      """Check and set new tensor's shape."""

      if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
        return tensor_util.with_same_shape(old, new)

      return new

    return nest.map_structure(

      with_same_shape,

      self,

      super(FactAwareWrapperState, self)._replace(**kwargs))

class FactAwareWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, encoder_states, common_words_proejction,
                 vocab_size, entity_size, initial_matching_score, facts_embedding,
                 encoder_state_size=None, initial_cell_state=None,
                 mode=None, name=None, binary_selector=False, dynamic_entity_score=False,output_size=None):
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
      super(FactAwareWrapper, self).__init__(name=name)

      self._mode = mode
      self._cell = cell
      self._encoder_states = encoder_states
      self._batch_size = tf.shape(self._encoder_states)[0]
      self._vocab_size = vocab_size
      self._entity_size = entity_size
      self._output_size = output_size
      if encoder_state_size is None:
        encoder_state_size = self._encoder_states.shape[-1].value
        if encoder_state_size is None:
          raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
      self._encoder_state_size = encoder_state_size
      self._initial_cell_state = initial_cell_state
      self._projection = common_words_proejction
      self._initial_matching_score = initial_matching_score
      self._binary_selector = binary_selector
      self._facts_embedding = facts_embedding
      self._dynamic_update = dynamic_entity_score
      self._fast_dynamic_updating = True


    def __call__(self, inputs, state, scope=None):
      if not isinstance(state, FactAwareWrapperState):
        raise TypeError("Expected state to be instance of FactAwareWrapperState. "
                        "Received type %s instead." % type(state))

      cell_state = state.cell_state
      last_ids = state.last_ids
      selector_state = state.selector_state

      # Generation Mode
      outputs, cell_state = self._cell(inputs, cell_state, scope)
      common_words_logits = self._projection(outputs)

      # fact_embedding: [batch, max_candidate_num, embed]
      # outputs: [batch, outputs]
      fact_embedding = self._facts_embedding
      entity_update_logits = tf.layers.dense(fact_embedding, units=self._output_size,  use_bias=False, name='fact_transform')
      entity_update_logits = tf.matmul(entity_update_logits, tf.expand_dims(outputs, -1))
      entity_update_logits = tf.squeeze(entity_update_logits, -1)


      common_words_logits = tf.concat([common_words_logits, entity_update_logits], -1)
      outputs = common_words_logits
      last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
      state = FactAwareWrapperState(cell_state=cell_state, last_ids=last_ids,
                                time_step=state.time_step + 1, selector_state=selector_state,
                                )
      return outputs, state

    @property
    def state_size(self):
      """size(s) of state(s) used by this cell.

          It can be represented by an Integer, a TensorShape or a tuple of Integers
          or TensorShapes.
      """
      return FactAwareWrapperState(cell_state=self._cell.state_size,
                               last_ids=tf.TensorShape([])
                               ,time_step=tf.TensorShape([])
                               ,selector_state=tf.TensorShape([]))

    @property
    def output_size(self):
      """Integer or TensorShape: size of outputs produced by this cell."""
      return self._vocab_size + self._entity_size

    def zero_state(self, batch_size, dtype):
      with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
        if self._initial_cell_state is not None:
          cell_state = self._initial_cell_state
        else:
          cell_state = self._cell.zero_state(batch_size, dtype)
        last_ids = tf.zeros([batch_size], tf.int32)
        selector_state = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        return FactAwareWrapperState(cell_state=cell_state,
                                 last_ids=last_ids,
                                 time_step=tf.zeros([], dtype=tf.int32,),
                                 selector_state=selector_state,
                                 )
