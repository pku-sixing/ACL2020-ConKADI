import json
import pprint
from lib import utils
import os

def load_and_restore_config(config_path, verbose=False):
    hparams = load_config(config_path, verbose=True)
    out_dir = hparams['model_path']
    utils.default_path = os.path.join(out_dir, 'log.txt')
    model_config_path = os.path.join(out_dir, 'config.json')
    eval_file = os.path.join(out_dir, 'eval_out.txt')
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    if os.path.exists(model_config_path):
        utils.print_out('reload the parameters from the %s' % model_config_path)
        loaded_hparams = load_config(model_config_path, verbose=True)
        for key in hparams.keys():
            if key not in loaded_hparams:
                utils.print_out('ADD HParam Key : %s' % key)
                loaded_hparams[key] = hparams[key]
        hparams = loaded_hparams
    return hparams

def load_config(config_path, verbose=False):
    """

    :param config_path:
    :return:  hparams
    """
    utils.print_out('load json config file from %s' % config_path)
    with open(config_path, encoding='utf-8') as fin:
        config = json.load(fin)
        if verbose:
            pprint.pprint(config)

        if 'loss' not in config:
            config['loss'] = []
            config['loss_r'] = []
            config['loss_c'] = []
            config['epochs'] = []
        if 'loss_c' not  in config:
            config['loss_c'] = []
        return config

def save_config(config, config_path=None):
    if config_path is None:
        config_path = config['model_path']+'/config.json'
    utils.print_out('save json config file to %s' % config_path)
    with open(config_path, 'w+', encoding='utf-8') as fout:
        json.dump(config, fout)