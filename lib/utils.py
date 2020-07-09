import math

default_path = None

def eval_print(eval_path, input):
    print(input)
    with open(eval_path, 'a+', encoding='utf-8') as fout:
      fout.write('%s\n' % input)

def print_out(input):
    if default_path is not None:
        with open(default_path, 'a+', encoding='utf-8') as fout:
            fout.write('%s\n' % input)
    print(input)

def print_time(time, title='Time'):

    DD = time // 3600 // 24
    time -= DD * 3600 * 24
    HH = time // 3600
    time -= HH * 3600
    MM = time / 60
    print_out('%s : %dD %dH %.2fM' % (title, DD, HH, MM))

def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def should_stop(epoch, step, hparams):
    """
    :param epoch:
    :param step:
    :param hparams:
    :return:
    """
    max_epoch, max_step = hparams['num_train_epochs'], hparams['num_train_steps']

    if max_epoch == -1:
        max_epoch = 1e10
    if max_step == -1:
        max_step = 1e10

    min_epoch, min_step = hparams['num_min_train_epochs'], hparams['num_min_train_steps']

    if step < min_step or epoch < min_epoch:
        return False
    if step >= max_step or epoch >= max_epoch:
        return True


    early_stop = hparams.get("early_stop", True)
    if early_stop:
        loss_list = hparams['loss']
        if len(loss_list) < 3:
            pass
        else:
            for i in range(2, len(loss_list)):
                if loss_list[i] > loss_list[i - 2] and loss_list[i-1] > loss_list[i - 2]:
                    print_out("Early Stop !")
                    return True
        if hparams['learning_rate'] < 1e-7:
            print_out('Early Stop , learning rate is less than 1e-7')
            return True

    return False