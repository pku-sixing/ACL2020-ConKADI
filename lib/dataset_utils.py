import tensorflow as tf
from KEFU2 import kefu_knowledge_utils


def create_flexka2_iterator(hparams, is_eval=False):
    tf_entity_embed, tf_entity_fact_embedding, \
    tf_entity_source, tf_entity_target, entity_vocab, \
    reverse_entity_vocab, word2entity, entity2word, \
    tf_fact_idf = kefu_knowledge_utils.load_entity_vocab(
        hparams)

    iterator, skip_count_placeholder, src_file_placeholder, tgt_in_file_placeholder, \
    tgt_out_file_placeholder, fact_file_placeholder, cue_fact_placeholder, neg_fact_placeholder, \
    src_vocab_table, tgt_vocab_table, reverse_tgt_vocab_table \
        = kefu_knowledge_utils.create_kefu_iterator_from_file(hparams, is_eval=is_eval)

    inputs = dict()
    inputs['word2entity'] = word2entity
    inputs['entity2word'] = entity2word

    inputs['inputs_for_encoder'] = iterator.source
    inputs['lengths_for_encoder'] = iterator.source_sequence_length
    inputs['entity_inputs_for_encoder'] = iterator.source_entity
    inputs['entity_inputs_for_decoder'] = iterator.target_input_entity

    inputs['inputs_for_decoder'] = iterator.target_input
    inputs['outputs_for_decoder'] = iterator.target_output
    inputs['lengths_for_decoder'] = iterator.target_sequence_length

    inputs['inputs_for_facts'] = iterator.fact
    inputs['lengths_for_facts'] = iterator.fact_length
    inputs['fact_entity_in_response'] = tf_entity_target
    inputs['fact_entity_in_post'] = tf_entity_source
    inputs['fact_idf'] = tf_fact_idf

    inputs['cue_fact'] = iterator.cue_fact
    inputs['neg_fact'] = iterator.neg_fact

    inputs['embedding_vocab'] = tf.get_variable('embedding_encoder',
                                                shape=[hparams['src_vocab_size'], hparams['embed_dim']])
    inputs['embedding_entity'] = tf_entity_embed
    inputs['embedding_fact'] = tf_entity_fact_embedding
    inputs['src_vocab_table'] = src_vocab_table
    inputs['tgt_vocab_table'] = tgt_vocab_table
    inputs['reverse_target_vocab_table'] = reverse_tgt_vocab_table

    inputs['dropout'] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name='dropout')

    def init_func(sess, prefix=""):
        tgt_out_suffix = ""
        if hparams.get('tgt_out_suffix', None) is not None:
            tgt_out_suffix = "." + hparams.get('tgt_out_suffix')
        sess.run(iterator.initializer, feed_dict={skip_count_placeholder: 0,
                                                  src_file_placeholder: hparams['%ssrc_file' % prefix],
                                                  tgt_in_file_placeholder: hparams['%stgt_file' % prefix],
                                                  tgt_out_file_placeholder: hparams[
                                                                                '%stgt_file' % prefix] + tgt_out_suffix,
                                                  fact_file_placeholder: hparams['%sfact_file' % prefix],
                                                  cue_fact_placeholder: hparams['%scue_fact_file' % (prefix)],
                                                  neg_fact_placeholder: hparams['%sneg_fact_file' % (prefix)],
                                                  })

    inputs['init_fn'] = init_func

    return inputs
