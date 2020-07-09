import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from KEFU2.kefu_model2 import Model
from lib import model_helper
from lib import config_parser
from lib.status_counter import Status
import time
import os
import tensorflow as tf
import argparse
from lib import utils
from lib import dataset_utils

parser = argparse.ArgumentParser()

parser.add_argument("--coverage_penalty_weight", type=float, default=0.0, help="coverage_penalty")
parser.add_argument("--diverse_decoding_rate", type=float, default=0.0, help="diverse_decoding_rate")
parser.add_argument("--length_penalty_weight", type=float, default=0.0, help="length_penalty_weight")

parser.add_argument("-c", "--config_path", type=str, help="config json path")
parser.add_argument("-t", "--test", type=bool, default=False, help="config json path")
parser.add_argument("-b", "--beam", type=int, default=-1, help="beam search width")
args = parser.parse_args()


def train():
    # Load config
    hparams = config_parser.load_and_restore_config(args.config_path, verbose=True)
    out_dir = hparams['model_path']
    eval_file = os.path.join(out_dir, 'eval_out.txt')

    status_per_steps = hparams['status_per_steps']
    status_counter = Status(status_per_steps)

    # dataset iterator
    dataset = dataset_utils.create_flexka2_iterator(hparams, is_eval=False)
    model = Model(dataset, hparams, model_helper.TRAIN)
    dropout = dataset['dropout']

    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:

        # Initialize or restore a model
        step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        dataset['init_fn'](sess)
        epoch_start_time = time.time()
        while utils.should_stop(epoch, step, hparams) is False:
            try:
                teach_force_loss, kld_loss, knowledge_bow_loss, word_bow_loss, lr, _, loss, step, epoch, predict_count, batch_size \
                    = sess.run([
                    model.teach_force_loss, model.kld_loss, model.knowledge_bow_loss, model.word_bow_loss,
                    model.learning_rate, model.update, model.train_loss,
                    model.global_step, model.epoch_step,
                    model.predict_count, model.batch_size],
                    feed_dict={dropout: hparams['dropout'], model.learning_rate: hparams['learning_rate']})

                # print(sess.run(model.debug))
                ppl = utils.safe_exp(loss * batch_size / predict_count)
                status_counter.add_record({'ppl': ppl, 'loss': loss, 'mode_loss': teach_force_loss,
                                           'word_bow_loss': word_bow_loss,
                                           'knowledge_bow_loss': knowledge_bow_loss,
                                           'kld_loss': kld_loss * 1000000, 'lr': lr, 'count': predict_count}, step,
                                          epoch)

            except tf.errors.InvalidArgumentError as e:
                print('Found Inf or NaN global norm')
                raise e
            except tf.errors.OutOfRangeError:
                utils.print_out('epoch %d is finished,  step %d' % (epoch, step))
                sess.run([model.next_epoch])
                # Save Epoch
                model.saver.save(
                    sess,
                    os.path.join(out_dir, "seq2seq.ckpt"),
                    global_step=model.global_step)
                utils.print_out('Saved model to -> %s' % out_dir)

                # EVAL on Dev/Test Set:
                for prefix in ['valid_', 'test_']:
                    dataset['init_fn'](sess, prefix)
                    eval_loss = []
                    eval_count = []
                    eval_batch = []
                    while True:
                        try:
                            loss, predict_count, batch_size, batch_size = sess.run(
                                [model.train_loss, model.predict_count, model.batch_size, model.batch_size],
                                feed_dict={dropout: 0.0})
                            eval_loss.append(loss)
                            eval_count.append(predict_count)
                            eval_batch.append(batch_size)
                        except tf.errors.OutOfRangeError as e:
                            pass
                            break
                    ppl = utils.safe_exp(sum(eval_loss) * sum(eval_batch) / len(eval_batch) / sum(eval_count))

                    if prefix == 'valid_':
                        utils.print_out('Eval on Dev: EVAL PPL: %.4f' % (ppl))
                        utils.eval_print(eval_file, 'Eval on Dev: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, ppl))

                        hparams['loss'].append(float(ppl))
                        hparams['epochs'].append(int(step))
                        config_parser.save_config(hparams)

                        if min(hparams['loss']) - ppl >= 0:
                            model.ppl_saver.save(
                                sess,
                                os.path.join(out_dir, 'min_ppl', "seq2seq.ckpt"),
                                global_step=model.global_step)
                            utils.print_out('Saved min_ppl model to -> %s' % out_dir)

                        if len(hparams['loss']) > 1:
                            if hparams['loss'][-1] > hparams['loss'][-2]:
                                hparams['learning_rate'] = hparams['learning_rate'] * hparams['learning_halve']
                                utils.eval_print(eval_file, 'Halved the learning rate to %f' % hparams['learning_rate'])
                                config_parser.save_config(hparams)
                    else:
                        utils.print_out('Eval on Test: EVAL PPL: %.4f' % (ppl))
                        utils.eval_print(eval_file,
                                         'Eval on Test: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, ppl))

                # NEXT EPOCH
                epoch_time = time.time() - epoch_start_time
                utils.print_time(epoch_time, 'Epoch Time:')
                epoch_time = time.time() - epoch_start_time
                epoch_time *= (hparams['num_train_epochs'] - epoch - 1)
                utils.print_time(epoch_time, 'Remaining Time:')
                epoch_start_time = time.time()

                dataset['init_fn'](sess)

        utils.print_out('model has been fully trained !')


def test():
    hparams = config_parser.load_and_restore_config(args.config_path, verbose=True)
    if args.beam != -1:
        hparams['beam_width'] = args.beam
        utils.print_out("Reset beam_width to %d" % args.beam)
    if args.beam > 10:
        hparams['batch_size'] = hparams['batch_size'] * 30 // args.beam

    hparams['length_penalty_weight'] = args.length_penalty_weight
    hparams['diverse_decoding_rate'] = args.diverse_decoding_rate
    hparams['coverage_penalty_weight'] = args.coverage_penalty_weight

    # Dataset
    dataset = dataset_utils.create_flexka2_iterator(hparams, is_eval=True)
    model = Model(dataset, hparams, model_helper.INFER)
    dropout = dataset['dropout']
    entity_word_vocab = []
    with open(hparams['fact_path'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n').split()
            items[0] = 'P:' + items[0]
            items[1] = 'E:' + items[1]
            entity_word_vocab.append(','.join(items))

    entity_set = set()
    with open(hparams['entity_path'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n')
            entity_set.add(items)

    input_srcs = []
    input_src_lens = []
    with open(hparams['%ssrc_file' % 'test_'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n')
            input_srcs.append(items)
            input_src_lens.append(len(items.split()))

    out_dir = os.path.join(hparams['model_path'], 'min_ppl')
    if os.path.exists(os.path.join(hparams['model_path'], 'decoded')) is False:
        os.mkdir(os.path.join(hparams['model_path'], 'decoded'))
    if os.path.exists(os.path.join(hparams['model_path'], 'decoded', 'fact_attention')) is False:
        os.mkdir(os.path.join(hparams['model_path'], 'decoded', 'fact_attention'))

    config_id = 'B%s_L%.1f_D%.1f_C%.1f' % \
                (hparams['beam_width'], args.length_penalty_weight, args.diverse_decoding_rate,
                 args.coverage_penalty_weight)

    beam_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s.txt' % config_id)
    top1_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.txt' % config_id)
    topk_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_topk.txt' % config_id)

    test_query_file = hparams['test_src_file']
    test_response_file = hparams['test_tgt_file']

    with open(test_query_file, 'r+', encoding='utf-8') as fin:
        queries = [x.strip('\n') for x in fin.readlines()]
    with open(test_response_file, 'r+', encoding='utf-8') as fin:
        responses = [x.strip('\n') for x in fin.readlines()]

    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:
        step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        dataset['init_fn'](sess, 'test_')

        utils.print_out('Current Epoch,Step : %s/%s, Max Epoch,Step : %s/%s' % (
        epoch, step, hparams['num_train_epochs'], hparams['num_train_steps']))
        case_id = 0
        with open(beam_out_file_path, 'w+', encoding='utf-8') as fout:
            with open(top1_out_file_path, 'w+', encoding='utf-8') as ftop1:
                with open(topk_out_file_path, 'w+', encoding='utf-8') as ftopk:
                    while True:
                        try:
                            model_selector, facts, lengts_for_facts, src_ids, sample_ids, probs, scores = sess.run(
                                [model.mode_selector, dataset['inputs_for_facts'], dataset['lengths_for_facts'],
                                 dataset['inputs_for_encoder'], model.sampled_id, model.logits, model.scores],
                                feed_dict={dropout: 0.0})
                            # print(()

                            num_responses_per_query = sample_ids.shape[0]
                            num_cases = sample_ids.shape[1]
                            for sent_id in range(num_cases):
                                fout.write('#Case : %d\n' % case_id)
                                fout.write('\tquery:\t%s\n' % queries[case_id])
                                fout.write('\tresponse:\t%s\n' % responses[case_id])

                                if hparams['beam_width'] == 1 and hparams.get(
                                        'fusion_encoder', True):
                                    input_src = input_srcs[case_id].split()
                                    for i in range(len(input_src)):
                                        if input_src[i] in entity_set:
                                            input_src[i] = input_src[i].upper()

                                for beam_id in range(num_responses_per_query):
                                    translations, score = model_helper.get_translation(sample_ids[beam_id],
                                                                                       scores[beam_id], sent_id, '</s>')
                                    new_translation = []
                                    for pid, token in enumerate(translations.split()):
                                        if token[:len('$ENT_')] == '$ENT_':
                                            relative_fact_id = int(token[len('$ENT_'):])
                                            fact = entity_word_vocab[facts[sent_id, relative_fact_id]]
                                            entity_in_response = fact.split(',')[1]
                                            new_translation.append('$' + entity_in_response)
                                        elif token[:len('$CP_')] == '$CP_':
                                            position = int(token[len('$CP_'):])
                                            new_translation.append('$C:' + input_srcs[case_id].split()[position])
                                        else:
                                            new_translation.append(token)
                                    translations = ' '.join(new_translation)
                                    fout.write('\tBeam%d\t%.4f\t%s\n' % (beam_id, score, translations))

                                    if beam_id == 0:
                                        ftop1.write('%s\n' % (
                                            translations.replace('#', '').replace('$R:', '').replace('$C:', '').replace(
                                                '$E:', '')))
                                    ftopk.write('%s\n' % (
                                        translations.replace('#', '').replace('$R:', '').replace('$C:', '').replace(
                                            '$E:', '')))
                                case_id += 1
                        except tf.errors.OutOfRangeError as e:
                            break


if args.test is False:
    train()
else:
    test()
