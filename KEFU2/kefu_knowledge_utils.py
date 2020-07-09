import numpy as np
import tensorflow as tf
from lib import utils
from lib import model_helper
from lib import vocab_utils
from KEFU2 import iterator_utils
from tensorflow.python.ops import lookup_ops
# #UNK
# #N
# #PE
# #NH
# #NT
UNK_ENTITY = '#UNK'
NONE_ENTITY = '#N'
PAD_ENTITY = '#PE'
NOT_HEAD_ENTITY = '#NH'
NOT_TAIL_ENTITY = '#NT'

#NF
#PR
#NR
NONE_RELATION = '#NF'
PAD_RELATION = '#PR'
NOT_TBD = '#NR'


def load_entity_vocab(hparams):
    """
    Currently same as GenDS.knowledge_utils.load_entity_vocab
    :param hparams:
    :return:
    """
    word2entity_dict_path = hparams['word2entity_dict_path']
    entity2word_dict_path = hparams['entity2word_dict_path']
    entity_dict_path = hparams['entity_path']
    relation_dict_path = hparams['relation_path']
    entity_embed_path = hparams['entity_embedding_path']
    relation_embed_path = hparams['relation_embedding_path']
    embed_dim = hparams['entity_dim']
    utils.print_out("load entity dict from %s" % entity_dict_path)

    entity_vocab = lookup_ops.index_table_from_file(entity_dict_path, default_value=0)
    reverse_entity_vocab = lookup_ops.index_to_string_table_from_file(entity_dict_path, default_value=UNK_ENTITY)
    padding_entity_list = [UNK_ENTITY, NONE_ENTITY, PAD_ENTITY, NOT_HEAD_ENTITY, NOT_TAIL_ENTITY]
    padding_relation_list = [NONE_RELATION, PAD_RELATION, NOT_TBD]

    # word2entity
    with open(word2entity_dict_path, encoding='utf-8') as fin:
        word2entities = np.array([int(x.strip('\n')) for x in fin.readlines()], dtype=np.int32)
        word2entities = tf.get_variable('word2entities', dtype=tf.int32, initializer=word2entities, trainable=False)

    # entity2word
    with open(entity2word_dict_path, encoding='utf-8') as fin:
        entity2word = np.array([int(x.strip('\n')) for x in fin.readlines()], dtype=np.int32)
        entity2word = tf.get_variable('entity2words', dtype=tf.int32, initializer=entity2word, trainable=False)

    entity_list = []
    relation_list = []

    entity_dict = dict()
    relation_dict = dict()

    # check
    with open(entity_dict_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)
            entity_dict[e] = i
    for i in range(len(padding_entity_list)):
        assert padding_entity_list[i] == entity_list[i]

    with open(relation_dict_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            e = line.strip()
            relation_list.append(e)
            relation_dict[e] = i
    for i in range(len(padding_relation_list)):
        assert padding_relation_list[i] == relation_list[i]

    print("Loading entity vectors...")
    entity_embed = []
    with open(entity_embed_path, 'r+', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if '\t' not in line:
                s = line.strip().split(' ')
            else:
                s = line.strip().split('\t')
            entity_embed.append([float(x) for x in s])

    print("Loading relation vectors...")
    relation_embed = []
    with open(relation_embed_path, 'r+', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if '\t' not in line:
                s = line.strip().split(' ')
            else:
                s = line.strip().split('\t')
            relation_embed.append([float(x) for x in s])

    entity_embed = np.array(entity_embed, dtype=np.float32)
    relation_embed = np.array(relation_embed, dtype=np.float32)

    entity_embed = tf.get_variable('entity_embed', dtype=tf.float32, initializer=entity_embed, trainable=False)
    relation_embed = tf.get_variable('relation_embed', dtype=tf.float32, initializer=relation_embed, trainable=False)
    entity_embed = tf.reshape(entity_embed, [-1, embed_dim])
    relation_embed = tf.reshape(relation_embed, [-1, embed_dim])

    padding_entity_embedding = tf.get_variable('entity_padding_embed', [len(padding_entity_list), embed_dim],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
    padding_relation_embedding = tf.get_variable('relation_padding_embed', [len(padding_relation_list), embed_dim],
                                                 dtype=tf.float32,
                                                 initializer=tf.zeros_initializer())
    tf_entity_embed = tf.concat([padding_entity_embedding, entity_embed], axis=0)
    tf_relation_embed = tf.concat([padding_relation_embedding, relation_embed], axis=0)
    tf_entity_embed = tf.layers.dense(tf_entity_embed, hparams['entity_dim'], use_bias=False,
                                      name='entity_embedding_transformer')
    tf_relation_embed = tf.layers.dense(tf_relation_embed, hparams['entity_dim'], use_bias=False,
                                        name='relation_embedding_transformer')

    tf_entity_embed = tf.concat([tf_entity_embed, tf_relation_embed], axis=0)

    # Facts
    utils.print_out('Loading facts')
    fact_dict_path = hparams['fact_path']
    entity_fact = []
    entity_source = []
    entity_target = []

    fact_idf = []
    with open(fact_dict_path, encoding='utf-8') as fin:
        lines = fin.readlines()
        utils.print_out('Total Entity-Fact : %d' % len(lines))
        print(lines[0].strip('\n').split())
        for line in lines:
            items = line.strip('\n').split()
            # 0:entity_in_post, 1:entity_in_response, 2 head, 3 relation, 4 tail 5/6/7 score
            for i in [0, 1, 2, 4]:
                items[i] = int(entity_dict.get(items[i], 0))
            items[3] = int(relation_dict.get(items[3])) + len(entity_dict)  # relation 和 entity共用一个列表
            entity_fact.append(items[2:5])
            entity_source.append(items[0])
            entity_target.append(items[1])  # uni ids

            if len(items) > 5:
                idf = [float(items[5]), float(items[6]), float(items[7])]
            else:
                idf = [0.0, 0.0, 0.0]
            fact_idf.append(idf)

    fact_idf = np.array(fact_idf, dtype=np.float32)
    entity_fact = np.array(entity_fact, dtype=np.int32)
    entity_source = np.array(entity_source, dtype=np.int32)
    entity_target = np.array(entity_target, dtype=np.int32)
    entity_fact = np.reshape(entity_fact, [len(lines), 3])
    entity_source = np.reshape(entity_source, [len(lines)])
    entity_target = np.reshape(entity_target, [len(lines)])

    tf_fact_idf = tf.constant(value=fact_idf, dtype=np.float32)
    tf_entity_fact = tf.constant(value=entity_fact, dtype=np.int32)
    tf_entity_source = tf.constant(value=entity_source, dtype=np.int32)
    tf_entity_target = tf.constant(value=entity_target, dtype=np.int32)
    tf_entity_fact_embedding = tf.nn.embedding_lookup(tf_entity_embed, tf_entity_fact)
    # index by context id
    tf_entity_fact_embedding = tf.reshape(tf_entity_fact_embedding, [-1, 3 * hparams['entity_dim']])

    return tf_entity_embed, tf_entity_fact_embedding, tf_entity_source, tf_entity_target, entity_vocab, \
           reverse_entity_vocab, word2entities, entity2word, tf_fact_idf



def create_kefu_iterator_from_file(hparams, is_eval=False):

    src_vocab_table, tgt_vocab_table = model_helper.create_vocab_from_file(hparams['src_vocab'], hparams['tgt_vocab'], hparams['share_vocab'])
    entity_vocab_table, _ = model_helper.create_vocab_from_file(hparams['entity_path'], hparams['entity_path'], hparams['share_vocab'])
    # 特殊符号的表

    tgt_meta_vocab_talbe, _ = model_helper.create_vocab_from_file(hparams['meta_tgt_vocab'], hparams['meta_tgt_vocab'], True)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        hparams['meta_tgt_vocab'], default_value=vocab_utils.UNK)


    src_file_placeholder = tf.placeholder(dtype=tf.string, shape=[], name='src_placeholder')
    tgt_in_file_placeholder = tf.placeholder(dtype=tf.string, shape=[], name='tgt_in_placeholder')
    tgt_out_file_placeholder = tf.placeholder(dtype=tf.string, shape=[], name='tgt_out_placeholder')
    fact_file_placeholder = tf.placeholder(dtype=tf.string, shape=[], name='fact_placeholder')
    cue_fact_placeholder = tf.placeholder(dtype=tf.string, shape=[], name='cue_fact_placeholder')
    neg_fact_placeholder = tf.placeholder(dtype=tf.string, shape=[], name='neg_fact_placeholder')


    src_dataset = tf.data.TextLineDataset(src_file_placeholder)
    cue_fact_dataset = tf.data.TextLineDataset(cue_fact_placeholder)
    neg_fact_dataset = tf.data.TextLineDataset(neg_fact_placeholder)
    tgt_in_dataset = tf.data.TextLineDataset(tgt_in_file_placeholder)
    tgt_out_dataset = tf.data.TextLineDataset(tgt_out_file_placeholder)
    fact_dataset = tf.data.TextLineDataset(fact_file_placeholder)
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
    num_buckets = hparams['num_buckets'] if not is_eval else 1
    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_in_dataset,
        tgt_out_dataset,
        cue_fact_dataset,
        neg_fact_dataset,
        fact_dataset,
        src_vocab_table,
        tgt_meta_vocab_talbe,
        entity_vocab_table,
        random_seed=hparams['random_seed'],
        num_buckets=num_buckets,
        batch_size=hparams['batch_size'],
        sos=vocab_utils.SOS,
        eos=vocab_utils.EOS,
        src_max_len=hparams['src_max_len'],
        tgt_max_len=hparams['tgt_max_len'],
        shuffle=not is_eval,
        skip_count=skip_count_placeholder)



    return iterator, skip_count_placeholder, src_file_placeholder, tgt_in_file_placeholder,\
           tgt_out_file_placeholder,fact_file_placeholder, cue_fact_placeholder, neg_fact_placeholder,\
           src_vocab_table, tgt_vocab_table, reverse_tgt_vocab_table



