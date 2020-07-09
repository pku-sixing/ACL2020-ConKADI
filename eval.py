from lib import utils
from lib import config_parser
import os
import time
import argparse
from lib import evaluation_utils
from multiprocessing import Pool

stopwords = {"": 1, "have": 1, "the": 1, "saying": 1, "two": 1, "beforehand": 1, "anyone": 1, "mustn't": 1, "on": 1, "hereby": 1, "don't": 1, "selves": 1, "whenever": 1, "say": 1, "you'll": 1, "very": 1, "wouldn't": 1, "either": 1, "seeming": 1, "seven": 1, "aren't": 1, "haven't": 1, "hadn't": 1, "do": 1, "sent": 1, "is": 1, "really": 1, "particularly": 1, "'m": 1, "together": 1, "need": 1, "let's": 1, "you've": 1, "didn't": 1, "himself": 1, "for": 1, "be": 1, "which": 1, "taken": 1, "her": 1, "of": 1, "'d": 1, "later": 1, "my": 1, "how": 1, "specifying": 1, "eg": 1, "could": 1, "next": 1, "asking": 1, "least": 1, "he'll": 1, "id": 1, "most": 1, "doing": 1, "can": 1, "i'd": 1, "afterwards": 1, "regarding": 1, "your": 1, "n't": 1, "again": 1, "was": 1, "comes": 1, "go": 1, "ourselves": 1, "qv": 1, "relatively": 1, "although": 1, "cause": 1, "time": 1, "against": 1, "normally": 1, "secondly": 1, "today": 1, "respectively": 1, "yes": 1, "other": 1, "ignored": 1, "latterly": 1, "or": 1, "definitely": 1, "from": 1, "what's": 1, "trying": 1, "first": 1, "we'll": 1, "may": 1, "via": 1, "awfully": 1, "won't": 1, "second": 1, "all": 1, "whatever": 1, "those": 1, "shall": 1, "unless": 1, "these": 1, "value": 1, "ok": 1, "our": 1, "he's": 1, "therefore": 1, "novel": 1, "actually": 1, "c'mon": 1, "are": 1, "please": 1, "whereafter": 1, "them": 1, "tends": 1, "if": 1, "mostly": 1, "best": 1, "says": 1, "their": 1, "i'm": 1, "seeing": 1, "beyond": 1, "they": 1, "instead": 1, "truly": 1, "back": 1, "then": 1, "neither": 1, "day": 1, "non": 1, "tell": 1, "oh": 1, "nor": 1, "so": 1, "he": 1, "herein": 1, "elsewhere": 1, "almost": 1, "amongst": 1, "lately": 1, "below": 1, "can't": 1, "despite": 1, "love": 1, "good": 1, "liked": 1, "haha": 1, "am": 1, "will": 1, "keeps": 1, "allows": 1, "whereby": 1, "name": 1, "ltd": 1, "onto": 1, "every": 1, "once": 1, "knows": 1, "less": 1, "thereafter": 1, "she's": 1, "a": 1, "particular": 1, "sup": 1, "therein": 1, "'t": 1, "everywhere": 1, "wherein": 1, "until": 1, "wants": 1, "indicate": 1, "viz": 1, "th": 1, "often": 1, "an": 1, "et": 1, "everybody": 1, "several": 1, "let": 1, "you'd": 1, "behind": 1, "looks": 1, "over": 1, "theres": 1, "what": 1, "wasn't": 1, "happens": 1, "usually": 1, "uses": 1, "help": 1, "thanks": 1, "little": 1, "course": 1, "former": 1, "mainly": 1, "re": 1, "insofar": 1, "ours": 1, "immediate": 1, "see": 1, "last": 1, "throughout": 1, "ought": 1, "noone": 1, "tried": 1, "anything": 1, "nine": 1, "entirely": 1, "we've": 1, "she'll": 1, "wherever": 1, "formerly": 1, "she": 1, "nothing": 1, "downwards": 1, "we": 1, "hereafter": 1, "indicated": 1, "zero": 1, "might": 1, "hence": 1, "toward": 1, "co": 1, "gotten": 1, "whence": 1, "well": 1, "lol": 1, "ain't": 1, "associated": 1, "hi": 1, "un": 1, "'ll": 1, "want": 1, "itself": 1, "here's": 1, "in": 1, "done": 1, "that's": 1, "com": 1, "it'll": 1, "quite": 1, "without": 1, "since": 1, "appear": 1, "come": 1, "beside": 1, "especially": 1, "him": 1, "look": 1, "available": 1, "third": 1, "during": 1, "presumably": 1, "just": 1, "brief": 1, "someone": 1, "three": 1, "one": 1, "shouldn't": 1, "four": 1, "welcome": 1, "whom": 1, "you're": 1, "different": 1, "que": 1, "used": 1, "reasonably": 1, "unfortunately": 1, "us": 1, "isn't": 1, "too": 1, "anywhere": 1, "already": 1, "new": 1, "right": 1, "seriously": 1, "exactly": 1, "though": 1, "anybody": 1, "else": 1, "inner": 1, "c's": 1, "apart": 1, "hardly": 1, "!": 1, "saw": 1, "hers": 1, "gets": 1, "clearly": 1, "here": 1, "they've": 1, "eight": 1, "whether": 1, "twice": 1, "above": 1, "provides": 1, "more": 1, "nd": 1, "way": 1, "known": 1, "must": 1, "example": 1, "became": 1, "possible": 1, "thereby": 1, "always": 1, "i": 1, "she'd": 1, "otherwise": 1, "moreover": 1, "both": 1, "unlikely": 1, "anyways": 1, "keep": 1, "as": 1, "however": 1, "herself": 1, "between": 1, "thus": 1, "while": 1, "take": 1, "hopefully": 1, "seemed": 1, "obviously": 1, "somehow": 1, "ask": 1, "there": 1, "contains": 1, "how's": 1, "it's": 1, "placed": 1, "after": 1, "off": 1, "up": 1, "allow": 1, "somewhere": 1, "near": 1, "ie": 1, "2": 1, "something": 1, "vs": 1, "because": 1, "were": 1, "many": 1, "nobody": 1, "not": 1, "who": 1, "goes": 1, "whither": 1, "sometime": 1, "yet": 1, "specified": 1, "appropriate": 1, "maybe": 1, "came": 1, "become": 1, "whereupon": 1, "willing": 1, "described": 1, "further": 1, "nearly": 1, "still": 1, "that": 1, "concerning": 1, "contain": 1, "they're": 1, "with": 1, "than": 1, "i'll": 1, "accordingly": 1, ",": 1, "nevertheless": 1, "rd": 1, "sensible": 1, "aside": 1, "whereas": 1, "they'd": 1, "same": 1, "into": 1, "know": 1, "becoming": 1, "certain": 1, "sometimes": 1, "fifth": 1, "indeed": 1, "able": 1, "also": 1, "given": 1, "around": 1, "even": 1, "plus": 1, "believe": 1, "upon": 1, "causes": 1, "furthermore": 1, "we'd": 1, "ones": 1, "you": 1, "inasmuch": 1, "few": 1, "consider": 1, "wish": 1, "merely": 1, "but": 1, "he'd": 1, "thorough": 1, "outside": 1, "yourself": 1, "except": 1, "his": 1, "some": 1, "did": 1, "where's": 1, "doesn't": 1, "shan't": 1, "couldn't": 1, "has": 1, "when": 1, "before": 1, "would": 1, "<UNK>": 1, "sorry": 1, "alone": 1, "took": 1, "and": 1, "try": 1, "it": 1, "far": 1, "edu": 1, "tries": 1, "thats": 1, "rather": 1, "followed": 1, "been": 1, "following": 1, "corresponding": 1, "probably": 1, "perhaps": 1, "went": 1, "inward": 1, "sub": 1, "nowhere": 1, "through": 1, "such": 1, "follows": 1, "serious": 1, "etc": 1, "its": 1, "themselves": 1, "necessary": 1, "appreciate": 1, "lest": 1, "enough": 1, "seen": 1, "there's": 1, "only": 1, "got": 1, "does": 1, "gone": 1, "should": 1, "namely": 1, "looking": 1, "using": 1, "where": 1, "anyhow": 1, "forth": 1, "none": 1, "somewhat": 1, "besides": 1, "whoever": 1, "across": 1, "useful": 1, "indicates": 1, "within": 1, "needs": 1, "yours": 1, "under": 1, "hello": 1, "seem": 1, "among": 1, "no": 1, "ex": 1, "never": 1, "gives": 1, ".": 1, "having": 1, "at": 1, "theirs": 1, "changes": 1, "i've": 1, "hasn't": 1, "u": 1, "currently": 1, "why": 1, "had": 1, "cannot": 1, "to": 1, "thereupon": 1, "?": 1, "like": 1, "soon": 1, "we're": 1, "mean": 1, "when's": 1, "sure": 1, "another": 1, "going": 1, "ever": 1, "who's": 1, "certainly": 1, "they'll": 1, "hereupon": 1, "by": 1, "self": 1, "much": 1, "t's": 1, "somebody": 1, "everyone": 1, "five": 1, "get": 1, "specify": 1, "whole": 1, "likely": 1, "better": 1, "thru": 1, "people": 1, "now": 1, "cant": 1, "d": 1, "use": 1, "this": 1, "a's": 1, "meanwhile": 1, "six": 1, "out": 1, "thoroughly": 1, "thanx": 1, "me": 1, "unto": 1, "yourselves": 1, "it'd": 1, "said": 1, "old": 1, "weren't": 1, "any": 1, "each": 1, "anyway": 1, "kept": 1, "regardless": 1, "thence": 1, "down": 1, "seems": 1, "considering": 1, "make": 1, "along": 1, "various": 1, "im": 1, "per": 1, "whose": 1, "away": 1, "hither": 1, "about": 1, "regards": 1, "others": 1, "wonder": 1, "think": 1, "overall": 1, "myself": 1, "own": 1, "inc": 1, "howbeit": 1, "towards": 1, "consequently": 1, "becomes": 1, "thank": 1, "latter": 1, "greetings": 1, "why's": 1, "getting": 1, "containing": 1, "being": 1, "everything": 1, "okay": 1, "'s": 1, "ours \tourselves": 1, "according": 1}


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type=str, help="config json path")
parser.add_argument("-b", "--beam", type=int, default=-1, help="beam search width")
parser.add_argument("-e", "--only_entity", type=bool, default=False, help="entity_mode")
parser.add_argument("-m", "--mmi", type=bool, default=False, help="mmi")
parser.add_argument("-s", "--binary", type=bool, default=False, help="enable binary selector")
parser.add_argument("--pre_embed_file", type=str, default="", help="enable binary selector")
parser.add_argument("--pre_embed_dim", type=int, default=-1, help="enable binary selector")
parser.add_argument("--thread", type=int, default=16, help="thread")
parser.add_argument("--rerank", type=int, default=0)


parser.add_argument("--coverage_penalty_weight", type=float, default=0.0, help="coverage_penalty")
parser.add_argument("--diverse_decoding_rate", type=float, default=0.0, help="diverse_decoding_rate")
parser.add_argument("--length_penalty_weight", type=float, default=0.0, help="length_penalty_weight")


args = parser.parse_args()



def main(args):
    hparams = config_parser.load_config(args.config_path, verbose=True)
    if args.beam != -1:
        hparams['beam_width'] = args.beam
        utils.print_out("Reset beam_width to %d" % args.beam)

    res_suffix = 'res'
    if args.pre_embed_file != '':
        hparams['pre_embed_file'] = args.pre_embed_file
        utils.print_out("Reset pre_embed_file to %s" % args.pre_embed_file)
        res_suffix = 'ores'


    if args.pre_embed_dim != -1:
        hparams['pre_embed_dim'] = args.pre_embed_dim
        utils.print_out("Reset pre_embed_dim to %s" % args.pre_embed_file)



    if args.rerank == 0:
        config_id = 'B%s_L%.1f_D%.1f_C%.1f' % (
        hparams['beam_width'], args.length_penalty_weight, args.diverse_decoding_rate, args.coverage_penalty_weight)
    else:
        config_id = 'R%s_B%s_L%.1f_D%.1f_C%.1f' % (
        args.rerank, hparams['beam_width'], args.length_penalty_weight, args.diverse_decoding_rate, args.coverage_penalty_weight)

    if os.path.exists(os.path.join(hparams['model_path'], 'decoded')) is False:
        os.mkdir(os.path.join(hparams['model_path'], 'decoded'))

    if args.binary:
        top1_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.txt.bi' % config_id)
        topk_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_topk.txt.bi' % config_id)

    else:
        top1_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.txt' % config_id)
        topk_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_topk.txt' % config_id)

        if args.rerank > 0:
            top1_out_file_path += '.mmi'
            topk_out_file_path += '.mmi'

    if os.path.exists(os.path.join(hparams['model_path'], 'decoded')) is False:
        os.mkdir(os.path.join(hparams['model_path'], 'decoded'))



    # Evalutation
    if args.binary:
        score_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.%s.bi' % (config_id, 'eres'))
    else:
        score_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.%s' % (config_id,'eres'))


    # check
    entity_list = []
    entity_dict = dict()
    entity_dict_path = hparams['entity_path']
    with open(entity_dict_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)
            entity_dict[e] = i

    # load generations
    generations = []
    with open(top1_out_file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.replace('#', '')
            line = line.replace('$C:', '')
            line = line.replace('$R:', '')
            line = line.replace('$E:', '')
            generations.append(line.strip('\n').split())

    # load refs
    refs = []
    with open(hparams['test_tgt_file'], 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.replace('#', '')
            line = line.replace('$C:', '')
            line = line.replace('$R:', '')
            line = line.replace('$E:', '')
            refs.append(line.strip('\n').split())

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

    # Compute Entity Score
    entity_recalls = []
    entity_sources_targets = []
    entity_targets = []
    with open(score_file_path, 'w+', encoding='utf-8') as fout:
        for generation, ref, idx in zip(generations, refs, fact_idx):
            # print(generation)
            # print(ref)
            entity_set = set()
            target_entity_set = set()
            for i in idx:
                assert len(facts[i]) == 5
                if facts[i][1] not in stopwords:
                    target_entity_set.add(facts[i][1])
                if facts[i][1] not in stopwords:
                    entity_set.add(facts[i][1])
                if facts[i][0] not in stopwords:
                    entity_set.add(facts[i][0])

            entity_score = 0.0
            generated_entities = set()
            matched_entities = set()
            for word in generation:
                if word in entity_set:
                    generated_entities.add(word)
                    entity_score += 1
                if word in target_entity_set:
                    matched_entities.add(word)

            entity_targets.append(len(matched_entities))
            entity_sources_targets.append(len(generated_entities))
            
            fout.write('%s\n' % ' '.join(matched_entities))
            ref_entities = set()
            for word in ref:
                if word in entity_set:
                    ref_entities.add(word)

            if len(ref_entities) != 0:
                entity_recalls.append(len(ref_entities & generated_entities) / (0.0+len(ref_entities)))
            else:
                entity_recalls.append(1.0)

        fout.write('%.4f\t%.4f\t%.4f\n' % (
                                                 sum(entity_targets)/len(generations), # Matched Entity Score
                                                 sum(entity_sources_targets)/len(generations), #Used Entity Score
                                                 sum(entity_recalls)/len(generations), # Recall
                                                 ))



    if args.only_entity is False:
        if args.binary:
            score_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.%s.bi' % (config_id, res_suffix))
        else:
            score_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.%s' % (config_id, res_suffix))

        scores = []
        metrics = 'bleu-2,bleu-3,distinct-1,distinct-2'.split(',')
        thread_pool = Pool(args.thread)
        jobs = []
        for metric in metrics:
           job = thread_pool.apply_async(evaluation_utils.evaluate, (
                    hparams['test_tgt_file'], hparams['test_src_file'], top1_out_file_path,
                    hparams['pre_embed_file'], metric, hparams['pre_embed_dim'], None, None, hparams['beam_width']))
           jobs.append(job)

        # entropy
        metrics.append('entropy')
        job = thread_pool.apply_async(evaluation_utils.evaluate, (
            hparams['tgt_file'], hparams['src_file'], top1_out_file_path, hparams['pre_embed_file'], 'entropy',
            hparams['pre_embed_dim'], hparams['tgt_vocab_size']))
        jobs.append(job)
        thread_pool.close()
        thread_pool.join()

        # Embedding-based
        complex_score = evaluation_utils.evaluate(hparams['test_tgt_file'], hparams['test_src_file'], top1_out_file_path,
                                          hparams['pre_embed_file'], 'embed', dim=hparams['pre_embed_dim'])
        score = complex_score[0:len(complex_score)//2]
        if len(score) == 1:
            score = score[0]

        utils.print_out(('%s->%s\n') % ('embed', score))
        if type(score) is list or type(score) is tuple:
            for x in score:
                scores.append(str(x))
        else:
            scores.append(str(score))

        for job, metric in zip(jobs, metrics):
            complex_score = job.get()
            score = complex_score[0:len(complex_score) // 2]
            if len(score) == 1:
                score = score[0]
            utils.print_out(('%s->%s\n') % (metric, score))

            if type(score) is list or type(score) is tuple:
                for x in score:
                    scores.append(str(x))
            else:
                scores.append(str(score))

        with open(score_file_path, 'w+', encoding='utf-8') as fin:
            fin.write('\t'.join(scores))


if __name__ == '__main__':
    start_time = time.time()
    main(args)
    print('Evaluation Time Consuming : %d' % (time.time() - start_time))