import tensorflow as tf
from lib import model_helper
from lib import utils
from lib import vocab_utils
from KEFU2 import KEFUAttentionWrapper2
from lib.layers.beam_search import BeamSearchDecoder


class Model:

    def __init__(self, inputs, hparams, mode):
        vocab_size = hparams['tgt_vocab_size']

        self.mode = mode
        self.dropout = inputs['dropout']
        self._word2entity = inputs['word2entity']
        self._entity2word = inputs['entity2word']
        self._fact_distribution = hparams.get('fact_distribution', False)

        id2type = tf.constant([0] * vocab_size + [1] * hparams['copy_token_nums'] + [2] * hparams['entity_token_nums'], dtype=tf.int32)
        self._embedding_id2type = tf.one_hot(id2type, depth=3, dtype=tf.float32)
        self._inputs_for_encoder = inputs['inputs_for_encoder']
        self._inputs_for_decoder = inputs['inputs_for_decoder']
        self._outputs_for_decoder = inputs['outputs_for_decoder']
        self._outputs_type_for_decoder = tf.nn.embedding_lookup(id2type, self._outputs_for_decoder)
        self.batch_size = tf.to_float(tf.shape(self._inputs_for_encoder)[0])


        self._cue_fact_idx = tf.squeeze(inputs['cue_fact'], -1)
        self._neg_fact_idx = tf.squeeze(inputs['neg_fact'], -1)

        self._lengths_for_fact_candidate = inputs['lengths_for_facts']

        self._lengths_for_encoder = inputs['lengths_for_encoder']
        self._lengths_for_decoder = inputs['lengths_for_decoder']


        self._embedding_vocab = inputs['embedding_vocab']
        self._embedding_entity = inputs['embedding_entity']
        self._embedding_fact = inputs['embedding_fact']


        self._fact_entity_in_response = inputs['fact_entity_in_response']
        self._fact_entity_in_post = inputs['fact_entity_in_post']
        self._fact_candidate = inputs['inputs_for_facts']

        self._max_fact_num = tf.shape(self._fact_candidate)[-1]

        # => [batch, fact_num]
        zero_mask = tf.one_hot(tf.zeros([self.batch_size], dtype=tf.int32)
        ,depth=vocab_size, on_value=0.0, off_value = 1.0, dtype = tf.float32)
        word_bow = tf.reduce_max(tf.one_hot(self._outputs_for_decoder, depth=vocab_size,
                                                  on_value=1.0, off_value=0.0,
                                                  dtype=tf.float32), 1)
        self._golden_word_bow = tf.minimum(word_bow,zero_mask)

        # => [batch, fact_num]
        zero_mask = tf.one_hot(tf.zeros([self.batch_size], dtype=tf.int32)
                               , depth=self._max_fact_num,
                               on_value=0.0, off_value=1.0,
                               dtype=tf.float32)
        golden_fact_bow = tf.reduce_max(tf.one_hot(inputs['cue_fact'], depth=self._max_fact_num,
                                                         on_value=1.0, off_value=0.0,
                                                         dtype=tf.float32), 1)
        self._golden_fact_bow = tf.minimum(golden_fact_bow, zero_mask)

        self._cue_fact_abs_idx = tf.squeeze(tf.batch_gather(inputs['inputs_for_facts'], inputs['cue_fact']), -1)

        if hparams.get('add_word_embedding_to_fact', False):
            response_words = tf.nn.embedding_lookup(self._entity2word, self._fact_entity_in_response)
            response_embedding = tf.nn.embedding_lookup(self._embedding_vocab, response_words)

            post_words = tf.nn.embedding_lookup(self._entity2word, self._fact_entity_in_post)
            post_embedding = tf.nn.embedding_lookup(self._embedding_vocab, post_words)

            self._embedding_fact = tf.concat([self._embedding_fact, post_embedding, response_embedding], axis=-1)


        self._fact_candidate_embedding = tf.nn.embedding_lookup(self._embedding_fact, self._fact_candidate)
        self._cue_fact_embedding = tf.nn.embedding_lookup(self._embedding_fact, self._cue_fact_abs_idx)

        # inputs
        self._input_embeddings_for_encoder = tf.nn.embedding_lookup(self._embedding_vocab, self._inputs_for_encoder)
        self._input_entities_for_encoder = inputs['entity_inputs_for_encoder']
        self._input_entity_embeddings_for_encoder = tf.nn.embedding_lookup(self._embedding_entity, self._input_entities_for_encoder)

        self._input_embeddings_for_decoder = tf.nn.embedding_lookup(self._embedding_vocab, self._inputs_for_decoder)
        self._input_entities_for_decoder = inputs['entity_inputs_for_decoder']
        self._input_entity_embeddings_for_decoder = tf.nn.embedding_lookup(self._embedding_entity, self._input_entities_for_decoder)

        self.fact_projection = tf.layers.dense(self._fact_candidate_embedding, units=hparams.get("sim_dim", 64),
                                               activation=tf.nn.tanh,
                                               name='dynamic_fact_projection')

        self._projection_layer = tf.layers.Dense(vocab_size, use_bias=False)
        self.predict_count = tf.reduce_sum(self._lengths_for_decoder)

        self.src_vocab_table = inputs['src_vocab_table']
        self.tgt_vocab_table = inputs['tgt_vocab_table']
        self.reverse_target_vocab_table = inputs['reverse_target_vocab_table']


        self.hparams = hparams

        self.global_step = tf.Variable(0, trainable=False)
        self.epoch_step = tf.Variable(0, trainable=False)
        self.next_epoch = tf.assign_add(self.epoch_step, 1)

        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        # warm-up
        self.learning_rate = model_helper.get_learning_rate_warmup(self.learning_rate, self.global_step, hparams)
        # decay
        self.learning_rate = model_helper.get_learning_rate_decay(self.learning_rate, self.global_step, hparams)

        self.create_model()
        if self.mode == model_helper.TRAIN:
            self.create_update_op()


        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=2)
        self.ppl_saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hparams['num_keep_ckpts'])




    def create_model(self, name='flexka'):

        def safe_log(y):
            return tf.log(tf.clip_by_value(y, 1e-9, tf.reduce_max(y)))

        hparams = self.hparams
        with tf.variable_scope(name) as scope:
            encoder_outputs, encoder_states = self.create_encoder(self._input_embeddings_for_encoder,
                                                                  self._input_entity_embeddings_for_encoder,
                                                                  self._lengths_for_encoder,
                                                                  )

            if self._fact_distribution:
                self.knowledge_fusion = None
                max_candidate_num = tf.shape(self._fact_candidate_embedding)[1]
                fact_embedding_projection = self.fact_projection
                prior_inputs = tf.concat(encoder_states, -1)
                prior_projection = tf.layers.dense(prior_inputs, units=hparams.get("sim_dim", 64), activation=tf.nn.tanh,
                                                   use_bias=True, name='prior_distribution')
                prior_projection = tf.expand_dims(prior_projection, 1)
                prior_projection = tf.tile(prior_projection, [1, max_candidate_num, 1])
                prior_scores = tf.reduce_sum(prior_projection * fact_embedding_projection, -1)
                fact_seq_mask = tf.sequence_mask(self._lengths_for_fact_candidate, dtype=tf.float32)
                unk_mask = tf.sequence_mask(tf.ones_like(self._lengths_for_fact_candidate), maxlen=max_candidate_num, dtype=tf.float32)
                fact_mask = (1.0 - fact_seq_mask) * -1e10 + unk_mask * -1e10
                prior_scores += fact_mask
                prior_distribution = tf.nn.softmax(prior_scores)


                if self.mode == model_helper.TRAIN:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        decoder_encoder_outputs, decoder_encoder_states = self.create_encoder(self._input_embeddings_for_decoder,
                                                                              self._input_entity_embeddings_for_decoder,
                                                                              self._lengths_for_decoder)
                    post_inputs = tf.concat(decoder_encoder_states + encoder_states, -1)
                    post_projection = tf.layers.dense(post_inputs, units=hparams.get("sim_dim", 64),  activation=tf.nn.tanh,
                                                       use_bias=True, name='post_distribution')
                    post_projection = tf.expand_dims(post_projection, 1)
                    post_projection = tf.tile(post_projection, [1, max_candidate_num, 1])
                    post_scores = tf.reduce_sum(post_projection * fact_embedding_projection, -1)
                    post_scores += fact_mask
                    post_distribution = tf.nn.softmax(post_scores)
                    self.knowledge_distribution = post_distribution

                    if hparams.get('knowledge_fusion', "none") == 'initDecoder':
                        self.knowledge_fusion = tf.reduce_sum(self._fact_candidate_embedding * tf.expand_dims(post_distribution, -1), 1)

                    kld_loss = post_distribution * safe_log(post_distribution / tf.clip_by_value(prior_distribution, 1e-9, 1.0)) #* fact_seq_mask
                    kld_loss = tf.reduce_mean(kld_loss, -1)
                    self.kld_loss = tf.reduce_sum(kld_loss) / self.batch_size

                    knowledge_bow_loss = - tf.reduce_sum(self._golden_fact_bow * safe_log(self.knowledge_distribution),
                                                         -1) / tf.maximum(tf.reduce_sum(self._golden_fact_bow, -1), 1)
                    self.knowledge_bow_loss = tf.reduce_sum(knowledge_bow_loss) / self.batch_size

                else:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        decoder_encoder_outputs, decoder_encoder_states = self.create_encoder(
                            self._input_embeddings_for_decoder,
                            self._input_entity_embeddings_for_decoder,
                            self._lengths_for_decoder)
                    post_inputs = tf.concat(decoder_encoder_states + encoder_states, -1)
                    post_projection = tf.layers.dense(post_inputs, units=hparams.get("sim_dim", 64),
                                                      activation=tf.nn.tanh,
                                                      use_bias=True, name='post_distribution')
                    post_projection = tf.expand_dims(post_projection, 1)
                    post_projection = tf.tile(post_projection, [1, max_candidate_num, 1])
                    post_scores = tf.reduce_sum(post_projection * fact_embedding_projection, -1)
                    post_scores += fact_mask
                    self.post_knowledge_distribution = tf.nn.softmax(post_scores)


                    self.knowledge_distribution = prior_distribution
                    if hparams.get('knowledge_fusion', "none") == 'initDecoder':
                        self.knowledge_fusion = tf.reduce_sum(self._fact_candidate_embedding * tf.expand_dims(prior_distribution, -1), 1)

                if self.knowledge_fusion is not None:
                    self.knowledge_fusion = tf.nn.dropout(self.knowledge_fusion, keep_prob=1.0 - self.dropout)


            else:
                self.kld_loss = tf.constant(0.0)
                self.knowledge_distribution = None
                self.knowledge_fusion = None
                self.knowledge_bow_loss = tf.constant(0.0)



            logits, sampled_id, scores = self.create_decoder(encoder_outputs, encoder_states,
                                         )
            self.logits = logits
            self.scores = scores
            if self.mode != model_helper.INFER:

                loss = self.compute_loss(logits, self._outputs_for_decoder, self._lengths_for_decoder,
                                         unk_helper=hparams.get("unk_helper", True))
                self.train_loss = tf.reduce_sum(loss) / self.batch_size

                self._train_update_loss = self.train_loss
                teach_force_loss = self.compute_loss(self.selector_logits, self._outputs_type_for_decoder,
                                                     self._lengths_for_decoder, unk_helper=False)
                self.teach_force_loss = tf.reduce_sum(teach_force_loss) / self.batch_size

                if hparams.get("teach_force", False):
                    self._train_update_loss += self.teach_force_loss * hparams.get("teach_force_rate", 0.5)
                else:
                    pass

                if self._fact_distribution:
                    self._train_update_loss += self.kld_loss

                if hparams.get("knowledge_bow_loss", False):
                    self._train_update_loss += self.knowledge_bow_loss * hparams.get("knowledge_bow_loss")

                if self.hparams.get('word_bow_loss', 0.0) > 0.0:
                    self._train_update_loss += self.word_bow_loss * self.hparams.get('word_bow_loss')


                self._cue_fact_loss = tf.constant(0.0)


            else:
                self.sampled_id = self.reverse_target_vocab_table.lookup(
                    tf.to_int64(sampled_id))

        # Print vars
        utils.print_out('-------------Trainable Variables------------------')
        for var in tf.trainable_variables():
            utils.print_out(var)


    def create_update_op(self):
        hparams = self.hparams
        # Optimizer
        if hparams['optimizer'] == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif hparams['optimizer'] == "adam":
            opt = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s" % hparams.optimizer)

        params = tf.trainable_variables()

        gradients = tf.gradients(
            self._train_update_loss ,
            params,
            colocate_gradients_with_ops=hparams['colocate_gradients_with_ops'])

        clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
            gradients, max_gradient_norm=hparams['max_gradient_norm'], safe_clip=hparams['safe_clip'])
        self.grad_norm_summary = grad_norm_summary
        self.grad_norm = grad_norm

        self.update = opt.apply_gradients(
            zip(clipped_grads, params), global_step=self.global_step)



    def create_encoder(self, seq_inputs, entity_inputs, lengths, name='encoder'):
        """

        :param inputs:  [batch,time,dimension]
        :param lengths:  [batch]
        :param hparams: hparams
        :return:
        """
        hparams = self.hparams
        mode = self.mode
        num_layers = hparams['encoder_num_layers']
        cell_type = hparams['cell_type']
        num_units = hparams['num_units']
        forget_bias = hparams['forget_bias']
        embed_dim = hparams['embed_dim']
        dropout = self.dropout


        with tf.variable_scope(name) as scope:
            inputs_for_std = seq_inputs
            inputs_for_fact = entity_inputs
            inputs = tf.concat([inputs_for_std, inputs_for_fact], axis=-1)

            def create_kefu_cell(name):
                 return  model_helper.create_cell(cell_type, num_units, forget_bias, dropout, mode)


            with tf.variable_scope('Knowledge_RNN'):

                cell_fw = create_kefu_cell('KEFU_FW')
                cell_bw = create_kefu_cell('KEFU_BW')

                cell_fw = tf.contrib.rnn.DeviceWrapper(cell_fw, '/gpu:0')
                cell_bw = tf.contrib.rnn.DeviceWrapper(cell_bw, '/gpu:1')

                utils.print_out('Creating bi_directional RNN Encoder, num_layers=%s, cell_type=%s, num_units=%d' %
                                (num_layers, cell_type, num_units))
                bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    inputs,
                    dtype=tf.float32,
                    sequence_length=lengths,
                    time_major=False,
                    swap_memory=True)
                encoder_outputs = tf.concat(bi_encoder_outputs, -1)
                encoder_state = [x for x in bi_encoder_state]

            return encoder_outputs, encoder_state


    def _prepare_beam_search_decoder_inputs(
            self, beam_width, memory, source_sequence_length, encoder_state):
        memory = tf.contrib.seq2seq.tile_batch(
            memory, multiplier=beam_width)
        source_sequence_length = tf.contrib.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        batch_size = self.batch_size * beam_width
        return memory, source_sequence_length, encoder_state, batch_size

    def create_attention_mechanism(self, attention_option, num_units, memory,
                                   source_sequence_length, mode):
        """Create attention mechanism based on the attention_option."""
        del mode  # unused

        # Mechanism
        if attention_option == "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units, memory, memory_sequence_length=source_sequence_length)
        elif attention_option == "scaled_luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units,
                memory,
                memory_sequence_length=source_sequence_length,
                scale=True)
        elif attention_option == "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units, memory, memory_sequence_length=source_sequence_length)
        elif attention_option == "normed_bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units,
                memory,
                memory_sequence_length=source_sequence_length,
                normalize=True)
        else:
            raise ValueError("Unknown attention option %s" % attention_option)

        return attention_mechanism

    def _dense_layer_helper(self, n, input, name):
        with tf.variable_scope(name):
            last_input = input
            for i in range(n - 1):
                last_input = tf.layers.dense(last_input, units=self.hparams['embed_dim'],
                                             use_bias=True, activation=tf.nn.relu, name='DNN_%d' % i)
            last_input = tf.layers.dense(last_input, units=1, use_bias=False,
                                         name='DNN_%d' % (i+1))
            last_input = tf.squeeze(last_input, axis=-1)
            return last_input



    def create_decoder(self, encoder_outputs, encoder_states, name='decoder'):
        hparams = self.hparams
        mode = self.mode
        sim_dim = self.hparams.get("sim_dim", 64)

        lengths = self._lengths_for_decoder
        copy_embedding_transform_fn = None

        copy_embedding_transform_fn = tf.layers.Dense(units=hparams['embed_dim'], name='copy_embedding_transformation')
        copy_fn_var_scope = tf.get_variable_scope()

        if self.mode == model_helper.TRAIN and hparams.get("multi_decoder_input", False):
            # Common Words
            embedding_list = []
            common_word_embedding = self._input_embeddings_for_decoder
            embedding_list.append(common_word_embedding)

            if hparams.get("copy_predict_mode", False):

                decoder_input_idx = self._inputs_for_decoder
                not_common_words = tf.greater_equal(decoder_input_idx, hparams['tgt_vocab_size'])
                not_entity_words = tf.less(decoder_input_idx, hparams['tgt_vocab_size'] + hparams['copy_token_nums'])
                is_copy_words = not_common_words & not_entity_words
                is_copy_mask = tf.cast(is_copy_words, tf.float32)

                copy_idx = decoder_input_idx - hparams['tgt_vocab_size']
                copy_idx = tf.maximum(copy_idx, 0)
                copy_idx = tf.minimum(copy_idx, hparams['copy_token_nums'] - 1)

                src_idx = self._inputs_for_encoder
                batch_size = tf.shape(src_idx)[0]
                max_src_len = tf.shape(src_idx)[1]
                max_tgt_len = tf.shape(copy_idx)[1]

                offset = tf.range(batch_size) * max_src_len
                offset = tf.expand_dims(offset, -1)
                offset = tf.tile(offset, [1, max_tgt_len])
                offset_copy_idx = copy_idx + offset

                flatten_encoder_outputs = tf.reshape(encoder_outputs, [-1, tf.shape(encoder_outputs)[-1]])
                copy_embedding = tf.nn.embedding_lookup(flatten_encoder_outputs, offset_copy_idx)
                copy_embedding = tf.reshape(copy_embedding, [batch_size, max_tgt_len, hparams.get("num_units")*2])
                copy_embedding = copy_embedding_transform_fn(copy_embedding)

                # common_word_idx = tf.where(is_copy_mask, copy_to_word_idx, common_word_idx)
                is_copy_mask = tf.expand_dims(is_copy_mask, -1)
                copy_embedding = copy_embedding * is_copy_mask

                embedding_list.append(copy_embedding)

            if hparams.get("entity_predict_mode", False):
                embedding_list.append(self._input_entity_embeddings_for_decoder)

            targets_in_embedding = tf.concat(embedding_list, -1)
        else:
            targets_in_embedding = self._input_embeddings_for_decoder


        with tf.variable_scope(name) as scope:
            num_layers = hparams['decoder_num_layers']
            cell_type = hparams['cell_type']
            num_units = hparams['num_units']
            forget_bias = hparams['forget_bias']
            dropout = self.dropout
            maximum_iterations = tf.reduce_max(self._lengths_for_encoder) * 2

            # Create RNN Cell
            with tf.variable_scope('std_rnn'):
                cell_list = [model_helper.create_cell(
                    unit_type=cell_type,
                    num_units=num_units,
                    forget_bias=forget_bias,
                    dropout=dropout,
                    mode=mode,
                ) for x in range(num_layers)]

            if num_layers > 1:
                cell_std = tf.contrib.rnn.MultiRNNCell(cell_list)
            else:
                cell_std = cell_list[0]


            if hparams.get("decoder_num_layers") == hparams.get("encoder_num_layers") and hparams.get("pass_raw_encoder_state", False):

                decoder_initial_state = []
                for i in range(num_layers):
                    decoder_initial_state.append(encoder_states[i])

                if num_layers > 1:
                    decoder_initial_state = tuple(decoder_initial_state)
                else:
                    decoder_initial_state = decoder_initial_state[0]
            else:
                decoder_initial_state = []
                if self.knowledge_fusion is None:
                    concatenated_encoder_states = tf.nn.dropout(tf.concat(encoder_states, -1), keep_prob=1.0 - dropout)
                else:
                    concatenated_encoder_states = tf.nn.dropout(tf.concat(encoder_states + [self.knowledge_fusion], -1), keep_prob=1.0 - dropout)

                if self.hparams.get('word_bow_loss', 0.0) > 0.0:
                    def safe_log(y):
                        return tf.log(tf.clip_by_value(y, 1e-9, tf.reduce_max(y)))
                    if self.hparams.get('word_bow_loss_type_2', False) is False:
                        common_word_inputs = tf.layers.dense(concatenated_encoder_states, self.hparams.get("mid_projection_dim"), tf.nn.elu,
                                                name = 'word_bow_predictor_1')
                        word_logits = tf.layers.dense(common_word_inputs, self.hparams.get("tgt_vocab_size"),
                                                      use_bias=False, name='word_bow_predictor_2')
                    else:
                        word_logits = tf.layers.dense(self.knowledge_fusion, self.hparams.get("tgt_vocab_size"),
                                                      use_bias=False, name='word_bow_predictor_2')
                    word_probs =  tf.nn.softmax(word_logits)
                    word_bow_loss = - tf.reduce_sum(self._golden_word_bow * safe_log(word_probs),
                                                         -1) / tf.maximum(tf.reduce_sum(self._golden_word_bow , -1), 1)
                    self.word_bow_loss = tf.reduce_sum(word_bow_loss) / self.batch_size
                else:
                    self.word_bow_loss = tf.constant(0.0)


                for i in range(num_layers):
                    init_out = tf.layers.dense(concatenated_encoder_states, num_units, activation=tf.nn.tanh,
                                        use_bias=False, name='std_transformer_%d' % i)
                    decoder_initial_state.append(init_out)


                if num_layers > 1:
                    decoder_initial_state = tuple(decoder_initial_state)
                else:
                    decoder_initial_state = decoder_initial_state[0]


            with tf.variable_scope('cue_rnn'):
                cell_list = [model_helper.create_cell(
                    unit_type=cell_type,
                    num_units=num_units,
                    forget_bias=forget_bias,
                    dropout=dropout,
                    mode=mode,
                ) for x in range(num_layers)]


            _batch_size = tf.shape(self._fact_candidate)[0]
            _fact_num = tf.shape(self._fact_candidate)[1]
            fact_projection = self.fact_projection


            if num_layers > 1:
                cell_cue = tf.contrib.rnn.MultiRNNCell(cell_list)
            else:
                cell_cue = cell_list[0]



            # Attention
            assert hparams['attention'] is not None
            memory = encoder_outputs
            if (self.mode == model_helper.INFER and
                    hparams['infer_mode'] == "beam_search"):
                memory, source_sequence_length, decoder_initial_state, batch_size = (
                    self._prepare_beam_search_decoder_inputs(
                        hparams["beam_width"], memory, self._lengths_for_encoder,
                        decoder_initial_state))

                if hparams.get('kefu_decoder', False):
                    _lengths_for_fact_candidate = tf.contrib.seq2seq.tile_batch(
                        self._lengths_for_fact_candidate, multiplier=hparams['beam_width'])
                    _fact_candidate_embedding = tf.contrib.seq2seq.tile_batch(
                        fact_projection, multiplier=hparams['beam_width'])
                    _cue_input_embedding = tf.contrib.seq2seq.tile_batch(self._cue_fact_embedding, multiplier=hparams['beam_width'])

                    fact_entity_idx = tf.contrib.seq2seq.tile_batch(self._fact_candidate, multiplier=hparams['beam_width'])
                    encoder_memory = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=hparams['beam_width'])
                    encoder_memory_len  = tf.contrib.seq2seq.tile_batch(self._lengths_for_encoder, multiplier=hparams['beam_width'])
                    if self.knowledge_distribution is not  None:
                        knowledge_distribution = tf.contrib.seq2seq.tile_batch(self.knowledge_distribution, multiplier=hparams['beam_width'])
                    else:
                        knowledge_distribution = self.knowledge_distribution


            else:
                fact_entity_idx = self._fact_candidate
                _lengths_for_fact_candidate = self._lengths_for_fact_candidate
                _fact_candidate_embedding = fact_projection
                source_sequence_length = self._lengths_for_encoder
                batch_size = self.batch_size
                _cue_input_embedding = self._cue_fact_embedding
                encoder_memory = encoder_outputs
                encoder_memory_len = self._lengths_for_encoder
                knowledge_distribution =  self.knowledge_distribution

            attention_mechanism = self.create_attention_mechanism(
                hparams["attention"], num_units, memory, source_sequence_length, self.mode)

            generate_probs_in_cell = hparams.get('kefu_decoder', True) and (hparams.get("entity_predict_mode", False) or hparams.get("copy_predict_mode", False))

            if generate_probs_in_cell:
                common_word_projection = self._projection_layer
            else:
                common_word_projection = None

            # Only generate alignment in greedy INFER mode.
            alignment_history = (self.mode == model_helper.INFER and
                                 hparams["infer_mode"] != "beam_search")

            k_openness_history = self.mode == model_helper.INFER

            if hparams.get('kefu_decoder', False):
                if hparams.get("use_dynamic_knowledge_distribution", True) is False:
                    knowledge_distribution = None
                cell_fw = KEFUAttentionWrapper2.AttentionWrapper(
                    cell_std,
                    cell_cue,
                    _cue_input_embedding,
                    _fact_candidate_embedding,
                    _lengths_for_fact_candidate,
                    knowledge_distribution,
                    attention_mechanism,
                    mid_projection_dim=hparams.get("mid_projection_dim_for_commonword", hparams.get("mid_projection_dim", 1280)),
                    cue_fact_mode=hparams.get("cue_fact", False),
                    cue_fact_mask=self.mode == model_helper.INFER,
                    encoder_memory=encoder_memory,
                    encoder_memory_len=encoder_memory_len,
                    balance_gate=hparams.get("balance_gate", True),
                    entity_predict_mode=hparams.get('entity_predict_mode', False),
                    copy_predict_mode=hparams.get('copy_predict_mode', False),
                    vocab_sizes=(hparams['tgt_vocab_size'], hparams['copy_token_nums'], hparams['entity_token_nums']),
                    common_word_projection=common_word_projection,
                    attention_layer_size=num_units,
                    alignment_history=alignment_history,
                    k_openness_history=k_openness_history,
                    output_attention=hparams["output_attention"],
                    sim_dim=sim_dim,
                    name="attention")
            else:

                cell_fw = tf.contrib.seq2seq.AttentionWrapper(
                    cell_std,
                    attention_mechanism,
                    attention_layer_size=num_units,
                    alignment_history=alignment_history,
                    output_attention=hparams["output_attention"],
                    name="attention")

            batch_size = tf.to_int32(batch_size)
            decoder_initial_state = cell_fw.zero_state(batch_size, tf.float32).clone(
                cell_state=decoder_initial_state)

            # Train or Eval
            if mode != tf.contrib.learn.ModeKeys.INFER:
                utils.print_out(
                    'Creating Training RNN Decoder, num_layers=%s, cell_type=%s, num_units=%d' %
                    (num_layers, cell_type, num_units))
                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    targets_in_embedding, lengths,
                    time_major=False)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell_fw,
                    helper,
                    decoder_initial_state)

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=False,
                    swap_memory=True,
                    scope=scope)

                rnn_outputs = outputs.rnn_output
                if generate_probs_in_cell:
                    logits = rnn_outputs
                else:
                    logits = self._projection_layer(rnn_outputs)

                if hparams.get("cue_fact", False):
                    self._cue_fact_loss = final_context_state.cue_fact_openness
                else:
                    self._cue_fact_loss = tf.constant(0.0)
                sampled_id = None
                scores = tf.no_op()

                self.selector_logits = tf.transpose(final_context_state.model_selector_openness.stack(), [1,0,2])


            else:
                utils.print_out(
                    'Creating Infer RNN Decoder, num_layers=%s, cell_type=%s, num_units=%d' %
                    (num_layers, cell_type, num_units))

                infer_mode = hparams["infer_mode"]
                utils.print_out('Infer mode : %s' % infer_mode)

                start_token = tf.cast(self.tgt_vocab_table.lookup(tf.constant(vocab_utils.SOS)),
                                     tf.int32)
                end_token = tf.cast(self.tgt_vocab_table.lookup(tf.constant(vocab_utils.EOS)),
                                     tf.int32)

                start_tokens = tf.fill([tf.shape(self._inputs_for_encoder)[0]], start_token)



                def embedding_fn_multi(input_idx,fact_entity_idx = fact_entity_idx, copy_embedding_transform_fn=copy_embedding_transform_fn):
                    common_word_idx = input_idx
                    embedding_list = [] # Reverse
                    # Common Copy Entity
                    if hparams.get("entity_predict_mode", False):
                        # entity mode
                        relative_entity_idx = input_idx - hparams.get('src_vocab_size') - hparams.get('copy_token_nums')
                        is_entity = tf.greater_equal(relative_entity_idx, 0)
                        is_entity_mask = tf.cast(is_entity, tf.float32)
                        relative_entity_idx = tf.maximum(0, relative_entity_idx)

                        # [batch, fact_len]
                        fact_entity_idx = fact_entity_idx
                        batch_size = tf.shape(input_idx)[0]
                        max_fact_num = tf.shape(fact_entity_idx)[1]
                        flatten_fact_idx = tf.reshape(fact_entity_idx, [-1])

                        offset = tf.expand_dims(tf.range(batch_size), -1) * max_fact_num
                        relative_entity_idx = relative_entity_idx + offset
                        fact_idx = tf.nn.embedding_lookup(flatten_fact_idx, relative_entity_idx)
                        entity_idx = tf.nn.embedding_lookup(self._fact_entity_in_response, fact_idx)
                        entity2word_idx = tf.nn.embedding_lookup(self._entity2word, entity_idx)

                        common_word_idx = tf.where(is_entity, entity2word_idx, common_word_idx)

                        tmp_common_word_idx = common_word_idx
                        if hparams.get("copy_predict_mode", False) is False:
                            common_word_idx_to_entity_idx = tf.nn.embedding_lookup(self._word2entity, tmp_common_word_idx)
                            entity_embedding = tf.nn.embedding_lookup(self._embedding_entity, common_word_idx_to_entity_idx)
                            embedding_list.append(entity_embedding)

                    if hparams.get("copy_predict_mode", False):
                        src_idx = self._inputs_for_encoder
                        max_src_len = tf.shape(src_idx)[1]
                        batch_size = tf.shape(input_idx)[0]

                        isnot_common_words = tf.greater_equal(input_idx, hparams['tgt_vocab_size'])
                        isnot_entity_words = tf.less(input_idx,
                                                     hparams['tgt_vocab_size'] + max_src_len)
                        is_copy_words = isnot_common_words & isnot_entity_words
                        is_copy_mask = tf.cast(is_copy_words, tf.float32)

                        copy_idx = input_idx - hparams['tgt_vocab_size']
                        copy_idx = tf.maximum(copy_idx, 0)
                        copy_idx = tf.minimum(copy_idx, max_src_len - 1)

                        max_tgt_len = tf.shape(copy_idx)[1]

                        offset = tf.range(batch_size) * max_src_len
                        offset = tf.expand_dims(offset, -1)
                        offset = tf.tile(offset, [1, max_tgt_len])
                        offset_copy_idx = copy_idx + offset

                        flatten_src_idx = tf.reshape(src_idx, [-1])
                        flatten_encoder_outputs = tf.reshape(encoder_outputs, [-1, tf.shape(encoder_outputs)[-1]])
                        copy_to_word_idx = tf.nn.embedding_lookup(flatten_src_idx, offset_copy_idx)
                        copy_embedding = tf.nn.embedding_lookup(flatten_encoder_outputs, offset_copy_idx)
                        copy_embedding = tf.reshape(copy_embedding,
                                                    [batch_size, max_tgt_len, hparams.get("num_units") * 2])
                        with tf.variable_scope(copy_fn_var_scope):
                            copy_embedding = copy_embedding_transform_fn(copy_embedding)

                        common_word_idx = tf.where(is_copy_words, copy_to_word_idx, common_word_idx)
                        tmp_common_word_idx = common_word_idx
                        if hparams.get("entity_predict_mode", False):
                            common_word_idx_to_entity_idx = tf.nn.embedding_lookup(self._word2entity,
                                                                                   tmp_common_word_idx)
                            entity_embedding = tf.nn.embedding_lookup(self._embedding_entity,
                                                                      common_word_idx_to_entity_idx)
                            embedding_list.append(entity_embedding)
                        is_copy_mask = tf.expand_dims(is_copy_mask, -1)
                        copy_embedding = copy_embedding * is_copy_mask
                        embedding_list.append(copy_embedding)

                    embedding_list.append(tf.nn.embedding_lookup(self._embedding_vocab, common_word_idx))

                    if hparams.get('add_token_type_feature', False):
                        embedding_list.append(tf.nn.embedding_lookup(self._embedding_id2type, input_idx))
                    # Must
                    embedding_list.reverse()
                    return tf.concat(embedding_list, -1)


                def embedding_fn(x, fact_entity_idx=fact_entity_idx):
                    if hparams.get('entity_predict_mode', False):
                        # > 0 is_entity else is word or copy token [0,500]
                        relative_entity_idx = x - hparams.get('src_vocab_size') - hparams.get('copy_token_nums')

                        is_entity = tf.greater(relative_entity_idx, 0)
                        relative_entity_idx = tf.maximum(0, relative_entity_idx)
                        # [batch, fact_len]
                        fact_entity_idx = fact_entity_idx
                        # Cast relative idx to right idx
                        batch_size  = tf.shape(fact_entity_idx)[0]
                        max_fact_num = tf.shape(fact_entity_idx)[1]
                        fact_entity_idx = tf.reshape(fact_entity_idx, [-1,1])

                        # batch_range
                        offset = tf.expand_dims(tf.range(batch_size), -1) * max_fact_num
                        relative_entity_idx = relative_entity_idx + offset
                        relative_entity_idx = tf.reshape(relative_entity_idx, tf.shape(x))

                        entity_idx = tf.nn.embedding_lookup(fact_entity_idx, relative_entity_idx)

                        entity_idx = tf.squeeze(entity_idx, -1)

                        entity_idx = tf.nn.embedding_lookup(self._fact_entity_in_response, entity_idx)
                        entity2word_idx = tf.nn.embedding_lookup(self._entity2word, entity_idx)

                        x = tf.where(is_entity, entity2word_idx, x)

                    return tf.nn.embedding_lookup(self._embedding_vocab, x)

                if infer_mode == "greedy":
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding_fn, start_tokens, end_token)
                if infer_mode == "beam_search":
                    beam_width = hparams["beam_width"]
                    beam_decoder_fn = BeamSearchDecoder
                    if generate_probs_in_cell:
                        projection_layer = None
                    else:
                        projection_layer = self._projection_layer
                    if hparams.get("multi_decoder_input", False):
                        my_embedding_fn = embedding_fn_multi
                    else:
                        my_embedding_fn = embedding_fn
                    my_decoder = beam_decoder_fn(
                        cell=cell_fw,
                        embedding=my_embedding_fn,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=projection_layer,
                        coverage_penalty_weight=hparams.get('coverage_penalty_weight', 0),
                        diverse_decoding_rate=hparams.get('diverse_decoding_rate', 0),
                        length_penalty_weight=hparams.get('length_penalty_weight', 0)

                    )
                else:
                    raise ValueError("Unknown infer_mode '%s'", infer_mode)

                if infer_mode != 'beam_search':
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell_fw,
                        helper,
                        decoder_initial_state,
                        output_layer=projection_layer  # applied per timestep
                    )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=maximum_iterations,
                    output_time_major=False,
                    swap_memory=True,
                    scope=scope)

                if infer_mode == "beam_search":
                    # sampled_id [batch_id,length,beam_id]
                    sampled_id = outputs.predicted_ids
                    logits = tf.no_op()
                    scores = outputs.beam_search_decoder_output.scores
                    # first dim is set to the beam_id
                    sampled_id = tf.transpose(sampled_id, [2, 0, 1])

                    #mapped_sampled_id = sampled_id
                    scores = tf.transpose(scores, [2, 0, 1])

                    if hparams.get('kefu_decoder', False):
                        self.mode_selector = final_context_state.cell_state.model_selector_openness
                        self.fact_alignments = final_context_state.cell_state.fact_alignments
                        if hparams.get('cue_fact', False):
                            self.k_openness = final_context_state.cell_state.k_openness
                        else:
                            self.k_openness = tf.constant(0.0)
                        self.copy_alignments = final_context_state.cell_state.copy_alignments
                        self.fact_alignments = final_context_state.cell_state.fact_alignments

                        if hparams.get("fact_memory_read", False):
                            self.fact_memory_alignments = final_context_state.cell_state.fact_memory_alignments
                        else:
                            self.fact_memory_alignments = tf.no_op()
                    else:
                        self.debug = tf.no_op()

                else:
                    logits = outputs.rnn_output
                    sampled_id = outputs.sample_id
                    scores = outputs.scores
                    sampled_id = tf.expand_dims(sampled_id, 0)
                    scores = tf.expand_dims(scores, 0)

            return logits, sampled_id, scores


    def _softmax_cross_entropy_loss(
            self, logits, labels):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        return crossent

    def compute_loss(self, logits, target_output, target_sequence_length, unk_helper=True):
        """Compute optimization loss."""
        max_time = target_output.shape[1].value or tf.shape(target_output)[1]

        crossent = self._softmax_cross_entropy_loss(
            logits, target_output)

        target_weights = tf.sequence_mask(target_sequence_length, max_time, dtype=tf.float32)
        if unk_helper:
            is_unk = tf.equal(target_output, 0)
            unk_val = tf.cast(is_unk, tf.float32)
            unk_val = unk_val / tf.reduce_sum(unk_val, keep_dims=True)
            unk_weights = unk_val * target_weights
            target_weights = tf.where(is_unk, unk_weights, target_weights)
        loss = crossent * target_weights
        return loss


