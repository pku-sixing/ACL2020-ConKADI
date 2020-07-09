from seq2seq import Model
from lib import utils
from lib import model_helper
from lib import config_parser
from lib.status_counter import Status
import time
import os
import tensorflow as tf
import json
import argparse
from lib import evaluation_utils
from multiprocessing.pool import Pool

stopwords = {"": 1, "have": 1, "the": 1, "saying": 1, "two": 1, "beforehand": 1, "anyone": 1, "mustn't": 1, "on": 1, "hereby": 1, "don't": 1, "selves": 1, "whenever": 1, "say": 1, "you'll": 1, "very": 1, "wouldn't": 1, "either": 1, "seeming": 1, "seven": 1, "aren't": 1, "haven't": 1, "hadn't": 1, "do": 1, "sent": 1, "is": 1, "really": 1, "particularly": 1, "'m": 1, "together": 1, "need": 1, "let's": 1, "you've": 1, "didn't": 1, "himself": 1, "for": 1, "be": 1, "which": 1, "taken": 1, "her": 1, "of": 1, "'d": 1, "later": 1, "my": 1, "how": 1, "specifying": 1, "eg": 1, "could": 1, "next": 1, "asking": 1, "least": 1, "he'll": 1, "id": 1, "most": 1, "doing": 1, "can": 1, "i'd": 1, "afterwards": 1, "regarding": 1, "your": 1, "n't": 1, "again": 1, "was": 1, "comes": 1, "go": 1, "ourselves": 1, "qv": 1, "relatively": 1, "although": 1, "cause": 1, "time": 1, "against": 1, "normally": 1, "secondly": 1, "today": 1, "respectively": 1, "yes": 1, "other": 1, "ignored": 1, "latterly": 1, "or": 1, "definitely": 1, "from": 1, "what's": 1, "trying": 1, "first": 1, "we'll": 1, "may": 1, "via": 1, "awfully": 1, "won't": 1, "second": 1, "all": 1, "whatever": 1, "those": 1, "shall": 1, "unless": 1, "these": 1, "value": 1, "ok": 1, "our": 1, "he's": 1, "therefore": 1, "novel": 1, "actually": 1, "c'mon": 1, "are": 1, "please": 1, "whereafter": 1, "them": 1, "tends": 1, "if": 1, "mostly": 1, "best": 1, "says": 1, "their": 1, "i'm": 1, "seeing": 1, "beyond": 1, "they": 1, "instead": 1, "truly": 1, "back": 1, "then": 1, "neither": 1, "day": 1, "non": 1, "tell": 1, "oh": 1, "nor": 1, "so": 1, "he": 1, "herein": 1, "elsewhere": 1, "almost": 1, "amongst": 1, "lately": 1, "below": 1, "can't": 1, "despite": 1, "love": 1, "good": 1, "liked": 1, "haha": 1, "am": 1, "will": 1, "keeps": 1, "allows": 1, "whereby": 1, "name": 1, "ltd": 1, "onto": 1, "every": 1, "once": 1, "knows": 1, "less": 1, "thereafter": 1, "she's": 1, "a": 1, "particular": 1, "sup": 1, "therein": 1, "'t": 1, "everywhere": 1, "wherein": 1, "until": 1, "wants": 1, "indicate": 1, "viz": 1, "th": 1, "often": 1, "an": 1, "et": 1, "everybody": 1, "several": 1, "let": 1, "you'd": 1, "behind": 1, "looks": 1, "over": 1, "theres": 1, "what": 1, "wasn't": 1, "happens": 1, "usually": 1, "uses": 1, "help": 1, "thanks": 1, "little": 1, "course": 1, "former": 1, "mainly": 1, "re": 1, "insofar": 1, "ours": 1, "immediate": 1, "see": 1, "last": 1, "throughout": 1, "ought": 1, "noone": 1, "tried": 1, "anything": 1, "nine": 1, "entirely": 1, "we've": 1, "she'll": 1, "wherever": 1, "formerly": 1, "she": 1, "nothing": 1, "downwards": 1, "we": 1, "hereafter": 1, "indicated": 1, "zero": 1, "might": 1, "hence": 1, "toward": 1, "co": 1, "gotten": 1, "whence": 1, "well": 1, "lol": 1, "ain't": 1, "associated": 1, "hi": 1, "un": 1, "'ll": 1, "want": 1, "itself": 1, "here's": 1, "in": 1, "done": 1, "that's": 1, "com": 1, "it'll": 1, "quite": 1, "without": 1, "since": 1, "appear": 1, "come": 1, "beside": 1, "especially": 1, "him": 1, "look": 1, "available": 1, "third": 1, "during": 1, "presumably": 1, "just": 1, "brief": 1, "someone": 1, "three": 1, "one": 1, "shouldn't": 1, "four": 1, "welcome": 1, "whom": 1, "you're": 1, "different": 1, "que": 1, "used": 1, "reasonably": 1, "unfortunately": 1, "us": 1, "isn't": 1, "too": 1, "anywhere": 1, "already": 1, "new": 1, "right": 1, "seriously": 1, "exactly": 1, "though": 1, "anybody": 1, "else": 1, "inner": 1, "c's": 1, "apart": 1, "hardly": 1, "!": 1, "saw": 1, "hers": 1, "gets": 1, "clearly": 1, "here": 1, "they've": 1, "eight": 1, "whether": 1, "twice": 1, "above": 1, "provides": 1, "more": 1, "nd": 1, "way": 1, "known": 1, "must": 1, "example": 1, "became": 1, "possible": 1, "thereby": 1, "always": 1, "i": 1, "she'd": 1, "otherwise": 1, "moreover": 1, "both": 1, "unlikely": 1, "anyways": 1, "keep": 1, "as": 1, "however": 1, "herself": 1, "between": 1, "thus": 1, "while": 1, "take": 1, "hopefully": 1, "seemed": 1, "obviously": 1, "somehow": 1, "ask": 1, "there": 1, "contains": 1, "how's": 1, "it's": 1, "placed": 1, "after": 1, "off": 1, "up": 1, "allow": 1, "somewhere": 1, "near": 1, "ie": 1, "2": 1, "something": 1, "vs": 1, "because": 1, "were": 1, "many": 1, "nobody": 1, "not": 1, "who": 1, "goes": 1, "whither": 1, "sometime": 1, "yet": 1, "specified": 1, "appropriate": 1, "maybe": 1, "came": 1, "become": 1, "whereupon": 1, "willing": 1, "described": 1, "further": 1, "nearly": 1, "still": 1, "that": 1, "concerning": 1, "contain": 1, "they're": 1, "with": 1, "than": 1, "i'll": 1, "accordingly": 1, ",": 1, "nevertheless": 1, "rd": 1, "sensible": 1, "aside": 1, "whereas": 1, "they'd": 1, "same": 1, "into": 1, "know": 1, "becoming": 1, "certain": 1, "sometimes": 1, "fifth": 1, "indeed": 1, "able": 1, "also": 1, "given": 1, "around": 1, "even": 1, "plus": 1, "believe": 1, "upon": 1, "causes": 1, "furthermore": 1, "we'd": 1, "ones": 1, "you": 1, "inasmuch": 1, "few": 1, "consider": 1, "wish": 1, "merely": 1, "but": 1, "he'd": 1, "thorough": 1, "outside": 1, "yourself": 1, "except": 1, "his": 1, "some": 1, "did": 1, "where's": 1, "doesn't": 1, "shan't": 1, "couldn't": 1, "has": 1, "when": 1, "before": 1, "would": 1, "<UNK>": 1, "sorry": 1, "alone": 1, "took": 1, "and": 1, "try": 1, "it": 1, "far": 1, "edu": 1, "tries": 1, "thats": 1, "rather": 1, "followed": 1, "been": 1, "following": 1, "corresponding": 1, "probably": 1, "perhaps": 1, "went": 1, "inward": 1, "sub": 1, "nowhere": 1, "through": 1, "such": 1, "follows": 1, "serious": 1, "etc": 1, "its": 1, "themselves": 1, "necessary": 1, "appreciate": 1, "lest": 1, "enough": 1, "seen": 1, "there's": 1, "only": 1, "got": 1, "does": 1, "gone": 1, "should": 1, "namely": 1, "looking": 1, "using": 1, "where": 1, "anyhow": 1, "forth": 1, "none": 1, "somewhat": 1, "besides": 1, "whoever": 1, "across": 1, "useful": 1, "indicates": 1, "within": 1, "needs": 1, "yours": 1, "under": 1, "hello": 1, "seem": 1, "among": 1, "no": 1, "ex": 1, "never": 1, "gives": 1, ".": 1, "having": 1, "at": 1, "theirs": 1, "changes": 1, "i've": 1, "hasn't": 1, "u": 1, "currently": 1, "why": 1, "had": 1, "cannot": 1, "to": 1, "thereupon": 1, "?": 1, "like": 1, "soon": 1, "we're": 1, "mean": 1, "when's": 1, "sure": 1, "another": 1, "going": 1, "ever": 1, "who's": 1, "certainly": 1, "they'll": 1, "hereupon": 1, "by": 1, "self": 1, "much": 1, "t's": 1, "somebody": 1, "everyone": 1, "five": 1, "get": 1, "specify": 1, "whole": 1, "likely": 1, "better": 1, "thru": 1, "people": 1, "now": 1, "cant": 1, "d": 1, "use": 1, "this": 1, "a's": 1, "meanwhile": 1, "six": 1, "out": 1, "thoroughly": 1, "thanx": 1, "me": 1, "unto": 1, "yourselves": 1, "it'd": 1, "said": 1, "old": 1, "weren't": 1, "any": 1, "each": 1, "anyway": 1, "kept": 1, "regardless": 1, "thence": 1, "down": 1, "seems": 1, "considering": 1, "make": 1, "along": 1, "various": 1, "im": 1, "per": 1, "whose": 1, "away": 1, "hither": 1, "about": 1, "regards": 1, "others": 1, "wonder": 1, "think": 1, "overall": 1, "myself": 1, "own": 1, "inc": 1, "howbeit": 1, "towards": 1, "consequently": 1, "becomes": 1, "thank": 1, "latter": 1, "greetings": 1, "why's": 1, "getting": 1, "containing": 1, "being": 1, "everything": 1, "okay": 1, "'s": 1, "ours \tourselves": 1, "according": 1}

def load_text(file_path, split=True, clean_prefix=True):
    generations = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if clean_prefix:
                line = line.replace('#', '')
                line = line.replace('$C:', '')
                line = line.replace('$R:', '')
                line = line.replace('$E:', '')
            if split:
                generations.append(line.strip('\n').split())
            else:
                generations.append(line.strip('\n'))
    return generations


def eval_entity_score(hparams, ref_tgt_file, ref_src_file, generated_file, score_file_path):
    # entity infor
    entity_list = []
    entity_dict = dict()
    entity_dict_path = hparams['entity_path']
    with open(entity_dict_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)
            entity_dict[e] = i

    # load generations
    generations = load_text(generated_file)
    refs = load_text(ref_tgt_file)

    # load facts
    facts = []
    with open(hparams['fact_path'], 'r', encoding='utf-8') as fin:
        for line in fin:
            facts.append([x.replace('#', '') for x in line.strip('\n').split()])

    # load fact idx
    fact_idx = []
    with open(hparams['test_fact_file'], 'r', encoding='utf-8') as fin:
        for line in fin:
            fact_idx.append([int(x) for x in line.strip('\n').split()])

    matched = []
    used = []
    copied = []
    num = []
    ratio = []
    recall = []
    precision = []

    with open(score_file_path, 'w+', encoding='utf-8') as fout:
        for generation, ref, idx in zip(generations, refs, fact_idx):
            entity_set = set()
            target_entity_set = set()
            source_entity_set = set()
            for i in idx:
                assert len(facts[i]) == 5
                if facts[i][0] not in stopwords:
                    source_entity_set.add(facts[i][0])
                if facts[i][1] not in stopwords:
                    target_entity_set.add(facts[i][1])
                if facts[i][1] not in stopwords:
                    entity_set.add(facts[i][1])
                if facts[i][0] not in stopwords:
                    entity_set.add(facts[i][0])

            entity_sum = 0.0
            used_entity = set()
            matched_entity = set()
            copied_entity = set()
            for word in generation:
                if word in entity_set:
                    used_entity.add(word)
                    entity_sum += 1
                if word in target_entity_set:
                    matched_entity.add(word)
                elif word in source_entity_set:
                    copied_entity.add(word)

            matched.append(len(matched_entity) / (0.0 + len(generation)))
            used.append(len(used_entity) / (0.0 + len(generation)))
            copied.append(len(copied_entity) / (0.0 + len(generation)))
            ratio.append(entity_sum / (0.0 + len(generation)))
            num.append(entity_sum)

            ref_entity = set()
            for word in ref:
                if word in entity_set:
                    ref_entity.add(word)

            if len(ref_entity) != 0:
                recall.append(len(ref_entity & used_entity) / (0.0+len(ref_entity)))
            else:
                recall.append(1.0)

            if len(used_entity) != 0:
                precision.append(len(ref_entity & used_entity) / (0.0+len(used_entity)))
            else:
                if len(ref_entity) != 0:
                    precision.append(0.0)
                else:
                    precision.append(1.0)

        return {
            "entity_matched" :            sum(matched)/len(generations), # Matched Entity Score
            "entity_used":                sum(used)/len(generations), # Used Entity Score
            "entity_copied":              sum(copied)/len(generations), # Used Entity 占比
            "entity_num":                 sum(num)/len(generations), # Used Entity 占比
            "entity_ratio":               sum(ratio)/len(generations), # Used Entity 占比
            "entity_recall":              sum(recall)/len(generations), # Recall
            "entity_precision":           sum(precision)/len(generations), # Precision
        }


def eval_std_metrics(hparams, ref_tgt_file, ref_src_file, generated_file):
    metrics = 'embed,rouge,bleu-1,bleu-2,bleu-3,bleu-4,distinct-1,distinct-2,distinct_c-1,distinct_c-2,accuracy,len,entropy'.split(
        ',')
    scores = []
    metric_num = len(metrics)
    pool = Pool(metric_num)
    jobs = []

    for metric in metrics:
        job = pool.apply_async(evaluation_utils.evaluate,(ref_tgt_file, ref_src_file, generated_file,
                                          hparams['pre_embed_file'], metric, hparams['tgt_vocab_size'],
                                                   None, None, hparams['pre_embed_dim']))
        jobs.append(job)
    pool.close()
    pool.join()

    res = dict()
    for metric, job in zip(metrics, jobs):
        score = job.get()
        if type(score) is list or type(score) is tuple:
            score = '-'.join([str(x) for x in score])
        else:
            score = str(score)
        utils.print_out('%s->%s\n' % (metric, score))
        res[metric] = score

    return res

def eval_std_metrics_st(hparams, ref_tgt_file, ref_src_file, generated_file):
    metrics = 'embed,rouge,bleu-1,bleu-2,bleu-3,bleu-4,distinct-1,distinct-2,distinct_c-1,distinct_c-2,accuracy,len'.split(
        ',')
    scores = []
    metric_num = len(metrics)


    for metric in metrics:
        score = evaluation_utils.evaluate(ref_tgt_file, ref_src_file, generated_file,
                                          hparams['pre_embed_file'], metric, dim=hparams['pre_embed_dim'])
        utils.print_out(('%s->%s\n') % (metric, score))
        if type(score) is list or type(score) is tuple:
            for x in score:
                scores.append(str(x))
        else:
            scores.append(str(score))
    metrics = ['entropy']
    for metric in metrics:
        score = evaluation_utils.evaluate(hparams['tgt_file'], hparams['src_file'], generated_file,
                                          hparams['pre_embed_file'], metric, vocab_size=hparams['tgt_vocab_size'])
        utils.print_out(('%s->%s\n') % (metric, score))
        if type(score) is list or type(score) is tuple:
            for x in score:
                scores.append(str(x))
        else:
            scores.append(str(score))
