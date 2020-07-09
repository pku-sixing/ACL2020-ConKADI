import numpy as np
import tensorflow as tf
from lib import utils
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

# NF
# PR
# NR
NONE_RELATION = '#NF'
PAD_RELATION = '#PR'
NOT_TBD = '#NR'


def load_knowledge_graph(hparams):
    """
    :param hparams:
    :return:
    """
    entity_dict_path = hparams['entity_path']
    relation_dict_path = hparams['relation_path']
    utils.print_out("load entity dict from %s" % entity_dict_path)
    utils.print_out("load relation dict from %s" % relation_dict_path)

    entity_embed_path = hparams['entity_embedding_path']
    relation_embed_path = hparams['relation_embedding_path']

    embed_dim = hparams['entity_dim']

    entity_vocab = lookup_ops.index_table_from_file(entity_dict_path, default_value=0)
    reverse_entity_vocab = lookup_ops.index_to_string_table_from_file(entity_dict_path, default_value=UNK_ENTITY)
    padding_entity_list = [UNK_ENTITY, NONE_ENTITY, PAD_ENTITY, NOT_HEAD_ENTITY, NOT_TAIL_ENTITY]
    padding_relation_list = [NONE_RELATION, PAD_RELATION, NOT_TBD]

    entity_list = []
    relation_list = []

    entity_dict = dict()
    relation_dict = dict()

    #  保证位置正确
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
    tf_entity_embed = tf.layers.dense(tf_entity_embed, hparams['entity_dim'], use_bias=False, name='entity_embedding')
    tf_relation_embed = tf.layers.dense(tf_relation_embed, hparams['entity_dim'], use_bias=False,
                                        name='relation_embedding')
    tf_entity_embed = tf.concat([tf_entity_embed, tf_relation_embed], axis=0)

    # Facts
    utils.print_out('Loading facts')
    fact_dict_path = hparams['fact_path']
    entity_fact = []
    entity_target = []
    with open(fact_dict_path, encoding='utf-8') as fin:
        lines = fin.readlines()
        utils.print_out('Total Entity-Fact : %d' % len(lines))
        for line in lines:
            items = line.strip('\n').split()
            for i in [0, 1, 3]:
                items[i] = int(entity_dict.get(items[i], 0))
            items[2] = int(relation_dict.get(items[2])) + len(entity_dict)  # realtion和 entity共用一个列表
            entity_fact.append(items[1:])
            entity_target.append(items[0])  # uni ids
    entity_fact = np.array(entity_fact, dtype=np.int32)
    entity_target = np.array(entity_target, dtype=np.int32)
    entity_fact = np.reshape(entity_fact, [len(lines), 3])
    entity_target = np.reshape(entity_target, [len(lines)])
    tf_entity_fact = tf.constant(value=entity_fact, dtype=np.int32)
    tf_entity_target = tf.constant(value=entity_target, dtype=np.int32)

    tf_entity_fact_embedding = tf.nn.embedding_lookup(tf_entity_embed, tf_entity_fact)
    tf_entity_fact_embedding = tf.reshape(tf_entity_fact_embedding, [-1, 3 * hparams['entity_dim']])

    return tf_entity_embed, tf_entity_fact_embedding, tf_entity_target, entity_vocab, reverse_entity_vocab
