# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess
import sys
import math

import tensorflow as tf

from lib.evaluation_scripts import bleu
from lib.evaluation_scripts import rouge
from lib.evaluation_scripts import tokens2wordlevel
from lib.evaluation_scripts import embed

from collections import Counter
__all__ = ["evaluate"]


def evaluate(ref_file, ref_src_file, trans_file, embed_file, metric, dim=200, vocab_size=None, subword_option=None, beam_width=10):
  """Pick a metric and evaluate depending on task."""

  if metric.lower() == "embed":
    evaluation_score = embed.eval(ref_file, trans_file, embed_file, dim,
                                  subword_option)

  elif len(metric.lower()) > 4 and metric.lower()[0:4]=='bleu':
    max_order = int(metric.lower()[5:])
    evaluation_score = _bleu(ref_file, trans_file, max_order=max_order,
                             subword_option=subword_option)
  elif metric.lower()[0:len('distinct')] == 'distinct':
    max_order = int(metric.lower()[len('distinct')+1:])
    evaluation_score = _distinct(trans_file, max_order, subword_option=subword_option)
  elif metric.lower()[0:len('len')] == 'len':
    evaluation_score = seq_len(trans_file, -1, subword_option=subword_option)
  elif metric.lower() == 'entropy':
    dict_files = [ref_file, ref_src_file]
    evaluation_score = _entropy_nrg(dict_files, trans_file, subword_option=subword_option, vocab_size=vocab_size)
  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score


def _clean(sentence, subword_option):
  sentence = tokens2wordlevel.revert_from_sentence(sentence, subword_option)
  sentence = sentence.replace('#','')
  return sentence



def _distinct(trans_file,max_order=1, subword_option=None):
  """Compute Distinct Score"""

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      translations.append(line.split(" "))

  num_tokens = 0
  unique_tokens = set()
  scores = []
  for items in translations:
      local_unique_tokens = set()
      local_count = 0.0
      #print(items)
      for i in range(0, len(items) - max_order + 1):
        tmp = ' '.join(items[i:i+max_order])
        unique_tokens.add(tmp)
        num_tokens += 1
        local_unique_tokens.add(tmp)
        local_count += 1
      if local_count == 0:
        scores.append(0)
      else:
        scores.append(100*len(local_unique_tokens) / local_count)
  if num_tokens == 0:
    ratio = 0
  else:
    ratio = len(unique_tokens) / num_tokens
  return 100 * ratio, scores



def seq_len(trans_file,max_order=1, subword_option=None):
  """Compute Length Score"""
  sum = 0.0
  total = 0
  scores = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      word_len = len(line.split())
      sum += word_len
      scores.append(word_len)
      total += 1
  return sum / total, scores


def _entropy_nrg(dict_files, trans_file, vocab_size, subword_option=None):
  """Compute Entropy Score"""
  counter = Counter()
  num_tokens = 0.0
  print("entropy reference files:")
  print(dict_files)
  for dict_file in dict_files:
    # with codecs.getreader("utf-8")(tf.gfile.GFile(dict_file)) as fh:
    with open(dict_file, encoding='utf-8') as fh:
      for line in fh:
        line = _clean(line, subword_option=subword_option)
        tokens = line.split(" ")
        num_tokens += len(tokens)
        counter.update(tokens)
  entropy = 0
  vocab_counts = counter.most_common(vocab_size)
  num_vocab_count = sum([x[1] for x in vocab_counts])
  unk_count = num_tokens - num_vocab_count
  num_infer_tokens = 0

  scores1 = []
  with open(trans_file, encoding='utf-8') as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      tokens = line.split(" ")
      local_scores = []
      for item in tokens:
        if item.find('unk') > 0:
          fre = unk_count
        else:
          fre = max(1, counter[item])
        p = fre/ num_tokens
        tmp = -math.log(p, 2)
        local_scores += [tmp]
        entropy += tmp
        num_infer_tokens += 1.0
      scores1.append(sum(local_scores)/len(local_scores))
  score1 = entropy/num_infer_tokens
  return (score1, scores1)

# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file,max_order=4, subword_option=None):
  """Compute BLEU scores and handling BPE."""
  smooth = False
  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(reference_filename, "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = _clean(reference, subword_option)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  print(per_segment_references[0:10])

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      translations.append(line.split(" "))
  print(translations[0:10])

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)

  blue_scores = []
  for ref, trans in zip(per_segment_references, translations):
    tmp_bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      [ref], [trans], max_order, smooth)
    blue_scores.append(tmp_bleu_score * 100)
  return 100 * bleu_score, blue_scores


def _moses_bleu(multi_bleu_script, tgt_test, trans_file, subword_option=None):
  """Compute BLEU scores using Moses multi-bleu.perl script."""

  # TODO(thangluong): perform rewrite using python
  # BPE
  if subword_option == "bpe":
    debpe_tgt_test = tgt_test + ".debpe"
    if not os.path.exists(debpe_tgt_test):
      # TODO(thangluong): not use shell=True, can be a security hazard
      subprocess.call("cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
      subprocess.call("sed s/@@ //g %s" % (debpe_tgt_test),
                      shell=True)
    tgt_test = debpe_tgt_test
  elif subword_option == "spm":
    despm_tgt_test = tgt_test + ".despm"
    if not os.path.exists(despm_tgt_test):
      subprocess.call("cp %s %s" % (tgt_test, despm_tgt_test))
      subprocess.call("sed s/ //g %s" % (despm_tgt_test))
      subprocess.call(u"sed s/^\u2581/g %s" % (despm_tgt_test))
      subprocess.call(u"sed s/\u2581/ /g %s" % (despm_tgt_test))
    tgt_test = despm_tgt_test
  cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

  # subprocess
  # TODO(thangluong): not use shell=True, can be a security hazard
  bleu_output = subprocess.check_output(cmd, shell=True)

  # extract BLEU score
  m = re.search("BLEU = (.+?),", bleu_output)
  bleu_score = float(m.group(1))

  return bleu_score
