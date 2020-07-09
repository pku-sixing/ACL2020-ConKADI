import numpy as np
import math
import codecs
import tensorflow as tf
from lib.evaluation_scripts import tokens2wordlevel
from multiprocessing import Pool
import os

def load_vocab(path,lower=True):
    vocab = set()
    with open(path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            word = line.strip('\n')
            if lower:
                word = word.lower()
            vocab.add(word)
    print('vocab size: %d' % len(vocab))
    return vocab

def load_embed_from_file(path, vocab=None, dim=300):
    """
    each line:
    word [float]*dim
    :param path:
    :return:
    """
    embeddings = dict()
    with open(path, 'r', encoding='utf-8') as fin:
        #fout = open(path+'.small', 'w', encoding='utf-8')
        line = fin.readline()
        while len(line) > 0:
            try:
                items = line.strip('\n').split()
                if vocab is None or items[0] in vocab:
                    embeddings[items[0]] = np.array([float(x) for x in items[1:dim+1]])
                    #if vocab is not None:
                       # fout.write(line)
            except:
                pass
                #print(line)
            finally:
                line = fin.readline()

    vocab_size = -1 if vocab is None else len(vocab)
    print('vocab size: %d, embedding_vocab size: %d' % (vocab_size, len(embeddings)))
    return embeddings

def unk_embed(dim):
    return (np.random.rand(dim) - 0.5)

def sentence_2_embedding(embeddings, sentence, dim=300, unk='<unk>'):
    words = sentence.strip('\r\n').split(' ')
    embedding = np.zeros([dim])
    counter = 0
    for word in words:
        if word in embeddings:
            embedding += embeddings[word]
            counter += 1
        elif unk in word:
            embedding += unk_embed(dim)
            counter += 1
        else:
            # add a new random_value
            new_embedding = unk_embed(dim)
            embeddings[word] = new_embedding
            embedding += embeddings[word]
            counter += 1
    if counter > 0:
        embedding /= counter
    return embedding, counter


def cosine_sim(a,b):
    return sum(a*b) / (np.sqrt(sum(a*a))*np.sqrt(sum(b*b)))



def _evaluate(method, inputs, refs, embedding, dim, unk):


    sims = 0
    counts = 0
    final_res = []
    avg_scores = []
    for input, ref in zip(inputs, refs):
        a, tmp = sentence_2_embedding(embedding, input, dim=dim)
        if tmp == 0:
            continue
        b, tmp = sentence_2_embedding(embedding, ref, dim=dim)
        if tmp == 0:
            continue
        avg_score = cosine_sim(a, b)
        avg_scores.append(avg_score)
        sims += avg_score
        counts += 1
    final_res.append((sims, counts, avg_scores))
    sims = 0
    counts = 0

    greedy_scores = []
    cache = dict()

    for tuple in [(inputs, refs), (refs, inputs)]:
        for input, ref in zip(tuple[0], tuple[1]):
            seq1 = input.strip('\n').split(' ')
            seq2 = ref.strip('\n').split(' ')
            local_counter = 0
            local_score = 0
            for a in seq1:  # 以Reference为主
                score = -1
                local_counter += 1
                for b in seq2:
                    key = a + '\t' + b
                    reverse_key = b + '\t' + a
                    if (key in cache) and a != unk and b != unk:
                        sim = cache[key]
                    else:
                        if a in embedding:
                            embed_a = embedding[a]
                        elif a == unk:
                            embed_a = unk_embed(dim)
                        else:
                            embed_a = unk_embed(dim)
                            embedding[a] = embed_a
                        if b in embedding:
                            embed_b = embedding[b]
                        elif a == unk:
                            embed_b = unk_embed(dim)
                        else:
                            embed_b = unk_embed(dim)
                            embedding[b] = embed_b
                        sim = cosine_sim(embed_a, embed_b)
                        cache[key] = sim
                        cache[reverse_key] = sim
                    score = max(score, sim)
                local_score += score
            local_counter = max(local_counter, 1)
            local_score /= local_counter
            greedy_scores.append(local_score)

        first_greedy = greedy_scores[0:len(greedy_scores) // 2]
        second_greedy = greedy_scores[len(greedy_scores) // 2:]
        greedy_scores = [x+y for x,y in zip(first_greedy, second_greedy)]

    final_res.append((sum(greedy_scores) , len(greedy_scores),greedy_scores))
    def create_extrema_vector(inputs, embedding):
        embeddings = []
        for word in inputs:
            if word in embedding:
                embeddings.append(embedding[word])
            elif unk in word:
                embeddings.append(unk_embed(dim))
            else:
                # add a new random_value
                new_embedding = unk_embed(dim)
                embedding[word] = new_embedding
                embeddings.append(embedding[word])
        # 取各维度的极值 [seq,embed_dim]
        embeddings = np.array(embeddings)
        abs_embeddings = np.abs(embeddings)
        second_indices = np.arange(np.shape(embeddings)[1])
        first_indices = np.argmax(abs_embeddings, 0)
        extrema_vector = embeddings[first_indices, second_indices]
        return extrema_vector

    scores = []
    for ref, input in zip(refs, inputs):
        if len(ref) == 0 or len(input) == 0:
            continue
        vector_ref = create_extrema_vector(ref, embedding)
        vector_input = create_extrema_vector(input, embedding)
        score = cosine_sim(vector_ref, vector_input)
        scores.append(score)

    final_res.append((sum(scores), len(scores),scores))

    return final_res

def evaluate_mt(input_file, ref_file, embedding, revert_func, dim=200, method='avg',unk='<unk>'):
    print('embed_[%s]_evaluating:%s ' % (method, input_file))
    with codecs.getreader("utf-8")(tf.gfile.GFile(input_file, "rb")) as fin:
        inputs = fin.readlines()
        if revert_func is not None:
            print("Before revert")
            print(inputs[0:5])
            inputs = [revert_func(x) for x in inputs]
            print("After revert")
            print(inputs[0:5])

    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fin:
        refs = fin.readlines()
        if revert_func is not None:
            print("Before revert")
            print(refs[0:5])
            refs = [revert_func(x) for x in refs]
            print("After revert")
            print(refs[0:5])

    sims = 0
    counts = 0

    pool = Pool(16)
    bulk_size = len(inputs) // 8 + 1
    jobs = []

    for i in range(0, len(inputs), bulk_size):
        job = pool.apply_async(_evaluate, (method, inputs[i:i+bulk_size], refs[i:i+bulk_size], embedding, dim, unk))
        jobs.append(job)
    pool.close()
    pool.join()

    avg_score = 0.0
    avg_count = 0.0
    grd_score = 0.0
    grd_count = 0.0
    ext_score = 0.0
    ext_count = 0.0
    avg_scores = []
    grd_scores = []
    ext_scores = []
    for job in jobs:
        avg, grd, ext = job.get()
        avg_score += avg[0]
        avg_count += avg[1]
        avg_scores += avg[2]
        grd_score += grd[0]
        grd_count += grd[1]
        grd_scores += grd[2]
        ext_score += ext[0]
        ext_count += ext[1]
        ext_scores += ext[2]

    return avg_score/avg_count, ext_score/ext_count, avg_scores,  ext_scores



def evaluate(input_file, ref_file, embedding, revert_func, dim=200, method='avg',unk='<unk>'):
    cache = dict()
    print('embed_[%s]_evaluating:%s ' % (method,input_file))
    with codecs.getreader("utf-8")(tf.gfile.GFile(input_file, "rb")) as fin:
        inputs = fin.readlines()
        if revert_func is not None:
            print("Before revert")
            print(inputs[0:5])
            inputs = [revert_func(x) for x in inputs]
            print("After revert")
            print(inputs[0:5])

    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fin:
        refs = fin.readlines()
        if revert_func is not None:
            print("Before revert")
            print(refs[0:5])
            refs = [revert_func(x) for x in refs]
            print("After revert")
            print(refs[0:5])

    sims = 0
    counts = 0
    if method == 'avg':
        for input, ref in zip(inputs, refs):
            a, tmp = sentence_2_embedding(embedding, input, dim=dim)
            if tmp == 0:
                continue
            b, tmp = sentence_2_embedding(embedding, ref, dim=dim)
            if tmp == 0:
                continue
            sims += cosine_sim(a, b)
            counts += 1
        return sims / counts
    elif method == 'greedy':
        greedy_scores = []
        for tuple in [(inputs, refs), (refs, inputs)]:
            scores = 0
            counts = 0
            for input, ref in zip(tuple[0], tuple[1]):
                seq1 = input.strip('\n').split(' ')
                seq2 = ref.strip('\n').split(' ')
                local_counter = 0
                local_score = 0
                for a in seq1:  # 以Reference为主
                    score = 0
                    local_counter += 1
                    for b in seq2:
                        key = a + '\t' + b
                        if (key in cache) and a != unk and b != unk:
                            sim = cache[key]
                        else:
                            if a in embedding:
                                embed_a = embedding[a]
                            elif a == unk:
                                embed_a = unk_embed(dim)
                            else:
                                embed_a = unk_embed(dim)
                                embedding[a] = embed_a
                            if b in embedding:
                                embed_b = embedding[b]
                            elif a == unk:
                                embed_b = unk_embed(dim)
                            else:
                                embed_b = unk_embed(dim)
                                embedding[b] = embed_b
                            sim = cosine_sim(embed_a, embed_b)
                            cache[key] = sim
                        score = max(score, sim)
                    local_score += score
                local_counter = max(local_counter, 1)
                local_score /= local_counter
                scores += local_score
                counts += 1
            greedy_scores.append(scores / counts)
        return sum(greedy_scores) / len(greedy_scores)
    elif method == 'extrema':
        def create_extrema_vector(inputs, embedding):
            embeddings = []
            for word in inputs:
                if word in embedding:
                    embeddings.append(embedding[word])
                elif unk in word:
                    embeddings.append(unk_embed(dim))
                else:
                    # add a new random_value
                    new_embedding = unk_embed(dim)
                    embedding[word] = new_embedding
                    embeddings.append(embedding[word])
            # 取各维度的极值 [seq,embed_dim]
            embeddings = np.array(embeddings)
            abs_embeddings = np.abs(embeddings)
            second_indices = np.arange(np.shape(embeddings)[1])
            first_indices = np.argmax(abs_embeddings, 0)
            extrema_vector= embeddings[first_indices, second_indices]
            return extrema_vector
        scores = []
        for ref, input in zip(refs,inputs):
            if len(ref) == 0 or len(input) == 0:
                continue
            vector_ref = create_extrema_vector(ref, embedding)
            vector_input = create_extrema_vector(input, embedding)
            score = cosine_sim(vector_ref, vector_input)
            scores.append(score)
        return sum(scores)/len(scores)




def eval(ref_file, input_file, embed_file, dim, subword=None, preload_embeddings = None):
    if preload_embeddings is not None:
        embeddings = preload_embeddings
    else:
        vocab = set()
        print([ref_file, input_file])
        for file in [ref_file, input_file]:
            with open(file, 'r', encoding='utf-8') as fin:
                for line in fin.readlines():
                    for word in tokens2wordlevel.revert_from_sentence(line.strip('\n'), subword).split():
                        vocab.add(word)
        print('#tokens in two files: %d' % len(vocab))
        embeddings = load_embed_from_file(embed_file, vocab)
    revert_func = lambda x: tokens2wordlevel.revert_from_sentence(x, subword)
    avg_sim, extrema_sim, avg_scores, extrema_scores = evaluate_mt(input_file, ref_file, embeddings, revert_func, method='extrema', dim=dim)
    return avg_sim, extrema_sim, avg_scores, extrema_scores
