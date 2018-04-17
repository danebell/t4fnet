
# coding: utf-8

#
#
# NN models to classify Twitter account users as Overweight or Not Overweight.
# 
#
CUDA_MODE = False
SEED = 947

import argparse
import configparser
import gzip
import numpy as np
np.random.seed(SEED) # for reproducibility
import pickle as pkl
import sys
import math
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from keras.preprocessing import sequence

parser = argparse.ArgumentParser(description='t4f-NN with domain adaptation.')
parser.add_argument('--dir',
                    help='directory to stores models and predictions')
parser.add_argument('--gender', action='store_true',
                    help='apply domain adapatation for gender')
parser.add_argument('--retweet', action='store_true',
                    help='apply domain adapatation for retweet')
parser.add_argument('--fold', default=None,
                    help='run for an expecific fold.')
parser.add_argument('--max_words', default=20000,
                    help='number of words in the embeddings matrix.')
parser.add_argument('--freeze', action='store_true',
                    help='freezes embeddings')
parser.add_argument('--setting', default=None,
                    help='hyperparameter setting file.')
parser.add_argument('--dev', action='store_true',
                    help='test on development set')
parser.add_argument('--tweetrel', action='store_true',
                    help='outputs tweet relevances.')
parser.add_argument('--outliers', action='store_true',
                    help='use only outlier tweet predictions')
parser.add_argument('--tweet_filter', default=None,
                    help='file with tweets used in training.')
parser.add_argument('--emb_file', default='food_vectors_clean.txt',
                    help='file with word embeddings.')
parser.add_argument('--vary_th', action='store_true',
                    help='run tests varying the threshold value')
parser.add_argument('--fixed_th', default=None, type=float,
                    help='run tests with a fixed threshold value')
parser.add_argument('--train_percent', default=None, type=float,
                    help='train with a percentage of the training data.')
parser.add_argument('--posW', default=0.5, type=float,
                    help='weight of the positive classes.')
parser.add_argument('--gru', action='store_true',
                    help='use GRU instead of CNN.')
parser.add_argument('--mlp', action='store_true',
                    help='use MLP instead of CNN.')
parser.add_argument('--wordorder', action='store_true',
                    help='use word order.')
parser.add_argument('--filebase', default='risk3dfdictugwo',
                    help='base name of the input files.')



args = parser.parse_args()
max_features = int(args.max_words)
base_dir = args.dir
run_fold = args.fold
if run_fold is not None:
    run_fold = run_fold.split(',')
model_dir = base_dir + '/models/'
domain = [False, False]
domain[0] = args.gender
domain[1] = args.retweet
freeze = args.freeze
tweetrel = args.tweetrel
dev_mode = args.dev
outliers = args.outliers
tweet_filter = args.tweet_filter
emb_file = args.emb_file
vary_th = args.vary_th
fixed_th = args.fixed_th
train_percent = args.train_percent
posW = args.posW
model_gru = args.gru
model_mlp = args.mlp
wordorder = args.wordorder
filebase = args.filebase

pred_dir = 'predictions'
if dev_mode:
    #pred_dir = pred_dir + '_dev'
    pred_dir = pred_dir + '_train'
    if tweetrel:
        #tweetrel_dir = base_dir + '/tweet_relevance_dev/'
        tweetrel_dir = base_dir + '/tweet_relevance_train/'
if outliers:
    pred_dir = pred_dir + '_outliers'
if vary_th:
    pred_dir = pred_dir + '_vary_th'
pred_dir = base_dir + '/' + pred_dir + '/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
if tweetrel:
    if not os.path.exists(tweetrel_dir):
        os.makedirs(tweetrel_dir)

torch.manual_seed(SEED)
if CUDA_MODE:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.enabled = False    
    
def pad3d(sequences, maxtweets=None, maxlen=None, dtype='int32',
          padding='pre', truncating='pre', value=0.):
    '''
        # Returns
        x: numpy array with dimensions (number_of_sequences, maxtweets, maxlen)
    '''
    nb_samples = len(sequences)
    if maxtweets is not None:
        mt = maxtweets
    else:
        mt = find_most_tweets(sequences)
    if maxlen is not None:
        ml = maxlen
    else:
        ml = find_longest(sequences)
    x = (np.ones((nb_samples, mt, ml)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue # no tweets
        x[idx, :min(mt,len(s))] = sequence.pad_sequences(s[:mt], ml, dtype, padding, truncating, value)
    return x

def find_most_tweets(x):
    currmax = 0
    for account in x:
        currlen = len(account)
        if currlen > currmax:
            currmax = currlen
    return currmax

def find_longest(x):
    currmax = 0
    for account in x:
        for tweet in account:
            currlen = len(tweet)
            if currlen > currmax:
                currmax = currlen
    return currmax

def cap_words(x, nb_words, oov=2):
    return [[[oov if w >= nb_words else w for w in z] for z in y] for y in x]

def skip_n(x, n, oov=2):
    return [[[oov if w < n else w for w in z] for z in y] for y in x]

def cap_length(x, maxlen):
    return [[z[:maxlen] for z in y] for y in x]

def push_indices(x, start, index_from):
    if start is not None:
        return [[[start] + [w + index_from for w in z] for z in y] for y in x]
    elif index_from:
        return [[[w + index_from for w in z] for z in y] for y in x]
    else:
        return x

#def load_data(path='ow3df.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3df.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3dfdict.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3dfdictu.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3dfdictu10t.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3dfdictuwo.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3dfdictug.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3dfdictugwo.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3dfdictupwo.pkl', nb_words=None, skip_top=0,
#def load_data(path='risk3dfdictug10t.pkl', nb_words=None, skip_top=0,
#def load_data(path='dow3df.pkl', nb_words=None, skip_top=0,
#def load_data(path='data_toy/ow3df.pkl', nb_words=None, skip_top=0,
def load_data(path=filebase + '.pkl', nb_words=None, skip_top=0,
              maxlen=None, seed=113, start=1, oov=2, index_from=3):
    '''
    # Arguments
        path: where the data is stored (in '.')
        nb_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occuring words
            (which may not be informative).
        maxlen: truncate sequences after this length.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov: words that were cut out because of the `nb_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `nb_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped. See preprocessText3D
    
    Adapted from keras.datasets.imdb.py by François Chollet
    '''
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    pos = pkl.load(f)
    if len(pos) == 5:
        (x_pos, i_pos, f_pos, y_pos, o_pos) = pos
    else:
        (x_pos, i_pos, f_pos, y_pos) = pos
        o_pos = []
    neg = pkl.load(f)
    if len(neg) == 5:
        (x_neg, i_neg, f_neg, y_neg, o_neg) = neg
    else:
        (x_neg, i_neg, f_neg, y_neg) = neg
        o_neg = []
    f.close()

    # randomize datum order
    np.random.seed(seed)
    np.random.shuffle(x_pos)
    np.random.seed(seed)
    np.random.shuffle(y_pos)
    np.random.seed(seed)
    np.random.shuffle(i_pos)
    np.random.seed(seed)
    np.random.shuffle(f_pos)
    np.random.seed(seed)
    np.random.shuffle(o_pos)

    np.random.seed(seed * 2)
    np.random.shuffle(x_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(y_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(i_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(f_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(o_neg)


    # keep maxlen words of each tweet
    if maxlen is not None:
        x_pos = cap_length(x_pos, maxlen)
        x_neg = cap_length(x_neg, maxlen)

    # cut off infrequent words to vocab of size nb_words
    if nb_words is not None:
        x_pos = cap_words(x_pos, nb_words, oov)
        x_neg = cap_words(x_neg, nb_words, oov)

    # cut off most frequent skip_top words
    if skip_top > 0:
        x_pos = skip_n(x_pos, skip_top, oov)
        x_neg = skip_n(x_neg, skip_top, oov)

    # prepend each sequence with start and raise indices by index_from
    x_pos = push_indices(x_pos, start, index_from)
    x_neg = push_indices(x_neg, start, index_from)

    for i in range(len(o_pos)):
        if o_pos[i] is not None:
            o = list(o_pos[i])
            o.reverse()
            o.append(0.)
            o.reverse()
            o_pos[i] = [np.array(o)]
        else:
            o_pos[i] = []
    for i in range(len(o_neg)):
        if o_neg[i] is not None:
            o = list(o_neg[i])
            o.reverse()
            o.append(0.)
            o.reverse()
            o_neg[i] = [np.array(o)]
        else:
            o_neg[i] = []

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)
    i_pos = np.array(i_pos)
    f_pos = np.array(f_pos)
    o_pos = np.array(o_pos)

    x_neg = np.array(x_neg)
    y_neg = np.array(y_neg)
    i_neg = np.array(i_neg)
    f_neg = np.array(f_neg)
    o_neg = np.array(o_neg)
   
    return (x_pos, y_pos, i_pos, f_pos, o_pos), (x_neg, y_neg, i_neg, f_neg, o_neg)


def load_embeddings(nb_words=None, emb_dim=200, index_from=3,
                    #vocab='ow3df.dict.pkl', 
                    #vocab='risk3df.dict.pkl',
                    #vocab='risk3dfdict.dict.pkl',
                    #vocab='risk3dfdictu.dict.pkl',
                    #vocab='risk3dfdictu10t.dict.pkl',
                    #vocab='risk3dfdictuwo.dict.pkl',
                    #vocab='risk3dfdictug.dict.pkl',
                    #vocab='risk3dfdictugwo.dict.pkl',
                    #vocab='risk3dfdictupwo.dict.pkl',
                    #vocab='risk3dfdictug10t.dict.pkl',
                    #vocab='dow3df.dict.pkl',  
                    #vocab='data_toy/ow3df.dict.pkl', 
                    vocab=filebase + '.dict.pkl',
                    w2v='food_vectors_clean.txt'):

    f = open(vocab, 'rb')
    word_index = pkl.load(f)
    f.close()
    
    if nb_words is not None:
        max_features = min(nb_words, len(word_index))
    else:
        max_features = len(word_index)

    embeddings_index = {}
    f = open(w2v, 'rb')
    fl = f.readline().strip().decode('UTF-8')
    i = 1
    for line in f:
        values = line.split()
        try:
            word = values[0].decode('UTF-8')
            if word in word_index:
                if word_index[word] < max_features:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            if i % 1000 == 0:
                print(".", end="")
            if i % 100000 == 0:
                print("")

            i = i + 1
        except:
            pass

    f.close()
    print("")
    print('Found %s word vectors.' % len(embeddings_index))
    
    max_features = min(max_features, len(embeddings_index))
    embedding_matrix = np.zeros((max_features+index_from, emb_dim))
    for word, i in word_index.items():
        if i < max_features:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i+index_from] = embedding_vector

    return embedding_matrix, max_features

def load_folds(file, seed=113):
    print ("Loading folds...")
    f = open(file, 'r')
    lines = f.readlines()
    last_fold = np.max([int(l.split(',')[0]) for l in lines])
    folds = list(list() for i in range(last_fold + 1))
    np.random.seed(seed)
    np.random.shuffle(lines)
    for line in lines:
        (fold, accountID, ow) = line.rstrip().split(',')
        folds[int(fold)].append((accountID, ow))
    f.close()
    return folds

def shuffle_in_unison(a, b, c, o=None):
    if o is None:
        assert len(a) == len(b) == len(c)
    else:
        assert len(a) == len(b) == len(c) == len(o)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    if o is not None:
        shuffled_o = np.empty(o.shape, dtype=o.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
        if o is not None:
            shuffled_o[new_index] = o[old_index]
    if o is None:
        return shuffled_a, shuffled_b, shuffled_c
    else:
        return shuffled_a, shuffled_b, shuffled_c, shuffled_o

def bootstrap(gold, pred, reps=100000, printit=True):
    '''
    # Arguments
        gold: list of gold (ground-truth) integer labels
        pred: list of predicted integer labels
        reps: how many repetitions to do (more=more accurate)

    Run a bootstrap significance test. Returns prediction 
    accuracy (out of 1), the accuracy of the baseline of 
    choosing the most common label in the gold labels, and 
    the p-value (the probability that you would do this much 
    better than the baseline by chance).
    '''
    accts = len(gold)
    hist = {}
    hist[-1] = 0
    for v in gold:
        if v in hist:
            hist[v] = hist[v] + 1
        else:
            hist[v] = 1
    baseline = max(hist.values()) / float(accts)
    agr = np.array(gold == pred, dtype='int32')
    better = np.zeros(reps)
    for i in range(reps):
        sample = np.random.choice(agr, accts)
        if np.mean(sample) > baseline:
            better[i] = 1
    p = (1. - np.mean(better))
    if p < 0.05:
        stars = '*'
    elif p < 0.01:
        stars = '**'
    elif p < 0.001:
        stars = '***'
    else:
        stars = ''

    acc = np.mean(agr)
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(gold)):
        if gold[i] == 1 and pred[i] == 1:
            tp = tp + 1
        elif gold[i] == 1 and pred[i] != 1:
            fn = fn + 1
        elif gold[i] != 1 and pred[i] == 1:
            fp = fp +1
        else:
            tn = tn +1

    precision = 0
    recall = 0
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

    f1s = []
    tps = []
    tns = []
    fns = []
    fps = []
    for lbl in (hist.keys()):
        if lbl == -1:
            continue
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for i in range(len(gold)):
            if gold[i] == lbl and pred[i] == lbl:
                tp = tp + 1
            elif gold[i] == lbl and pred[i] != lbl:
                fn = fn + 1
            elif gold[i] != lbl and pred[i] == lbl:
                fp = fp +1
            else:
                tn = tn +1

        tps.append(tp)
        tns.append(tn)
        fns.append(fn)
        fps.append(fp)
        if (tp + fp == 0):
            prec = 0
        else:
            prec = tp / (tp + fp)
        if (tp + fn == 0):
            rec = 0
        else:
            rec = tp / (tp + fn)
        if (prec + rec == 0):
            f1s.append(0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))

    f1s = np.array(f1s)
    tps = np.array(tps)
    tns = np.array(tns)
    fns = np.array(fns)
    fps = np.array(fps)
    prec = tps.sum() / (tps.sum() + fps.sum())
    rec = tps.sum() / (tps.sum() + fns.sum())
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    microf1 = 2 * prec * rec / (prec + rec)
    macrof1 = f1s.sum() / len(f1s)

    if printit:
        print('accuracy = %.4f' % acc)
        print('precision = %.4f' % precision)
        print('recall = %.4f' % recall)
        print('F1 = %.4f' % f1)
        print('microF1 = %.4f' % microf1)
        print('macroF1 = %.4f' % macrof1)
        print('baseline = %.4f' % baseline)
        print('p = %.6f%s' % (p, stars))
    return (acc, precision, recall, f1, microf1, macrof1, baseline, p)



def gen_iterations(pos, neg, max_features, maxtweets, maxlen, foldsfile, relevant=None):
    (x_pos, y_pos, i_pos, f_pos, o_pos), (x_neg, y_neg, i_neg, f_neg, o_neg) = pos, neg

    relevant_tweets = [[[False for i in range(2000)] for j in range(5507)] for k in range(10)]
    if relevant is not None:
        relevantfile = open(relevant,'rb')
        r_tweets = pkl.load(relevantfile)
        for fold in range(0, len(r_tweets)):
            for user in range(0, len(r_tweets[fold])):
                for tweet in r_tweets[fold][user]:
                    tweet = int(tweet)
                    relevant_tweets[fold][user][tweet] = True
        relevantfile.close()
    else:
        del (relevant_tweets)
    folds = load_folds(foldsfile, seed=SEED)
    for itern in range(0, len(folds)):
        X_train = list()
        y_train = list()
        f_train = list()
        o_train = list()
        X_test = list()
        y_test = list()
        f_test = list()
        o_test = list()
        X_dev = list()
        y_dev = list()
        f_dev = list()
        o_dev = list()
        for user in folds[itern]:
            if user[1] == "Overweight" or user[1] == "risk":
                position = np.where(i_pos == user[0])[0][0]
                X_test.append(x_pos[position])
                y_test.append(y_pos[position])
                f_test.append(f_pos[position])
                if len(o_pos) > 0:
                    o_test.append(o_pos[position])
            else:
                position = np.where(i_neg == user[0])[0][0]
                X_test.append(x_neg[position])
                y_test.append(y_neg[position])
                f_test.append(f_neg[position])
                if len(o_neg) > 0:
                    o_test.append(o_neg[position])
        nitern = itern + 1
        if nitern == len(folds):
            nitern = 0
        for user in folds[nitern]:
            if user[1] == "Overweight" or user[1] == "risk":
                position = np.where(i_pos == user[0])[0][0]
                X_dev.append(x_pos[position])
                y_dev.append(y_pos[position])
                f_dev.append(f_pos[position])
                if len(o_pos) > 0:
                    o_dev.append(o_pos[position])
            else:
                position = np.where(i_neg == user[0])[0][0]
                X_dev.append(x_neg[position])
                y_dev.append(y_neg[position])
                f_dev.append(f_neg[position])
                if len(o_neg) > 0:
                    o_dev.append(o_neg[position])

        for j in range(0, len(folds)):
            if itern != j and nitern != j:
                for user in folds[j]:
                    if user[1] == "Overweight" or user[1] == "risk":
                        position = np.where(i_pos == user[0])[0][0]
                        X_train.append(x_pos[position])
                        y_train.append(y_pos[position])
                        f_train.append(f_pos[position])
                        if len(o_pos) > 0:
                            o_train.append(o_pos[position])
                    else:
                        position = np.where(i_neg == user[0])[0][0]
                        X_train.append(x_neg[position])
                        y_train.append(y_neg[position])
                        f_train.append(f_neg[position])
                        if len(o_neg) > 0:
                            o_train.append(o_neg[position])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        f_train = np.array(f_train)
        o_train = np.array(o_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        f_test = np.array(f_test)
        o_test = np.array(o_test)
        X_dev = np.array(X_dev)
        y_dev = np.array(y_dev)
        f_dev = np.array(f_dev)
        o_dev = np.array(o_dev)

        X_train = pad3d(X_train, maxtweets=maxtweets, maxlen=maxlen)
        X_test = pad3d(X_test, maxtweets=maxtweets, maxlen=maxlen)
        X_dev = pad3d(X_dev, maxtweets=maxtweets, maxlen=maxlen)
        o_train = pad3d(o_train, maxtweets=maxtweets, maxlen=maxlen, dtype='float')
        o_test = pad3d(o_test, maxtweets=maxtweets, maxlen=maxlen, dtype='float')
        o_dev = pad3d(o_dev, maxtweets=maxtweets, maxlen=maxlen, dtype='float')
        train_shp = X_train.shape
        test_shp = X_test.shape
        dev_shp = X_dev.shape
      
        o_train_flat = None
        o_dev_flat = None
        o_test_flat = None
        o_train_shuff = None
        o_dev_shuff = None
        o_test_shuff = None
        X_train_flat = list()
        y_train_flat = list()
        f_train_flat = list()
        if relevant is not None:
            for u in range(0,len(X_train)):
                for t in range(0,len(X_train[u])):
                    if relevant_tweets[itern][u][t]:
                        X_train_flat.append(X_train[u][t])
                        y_train_flat.append(y_train[u])
                        f_train_flat.append(f_train[u])
                        if len(o_train) > 0:
                            o_train_flat.append(o_train[u][t])
        else:
            X_train_flat = X_train.reshape(train_shp[0] * train_shp[1], train_shp[2])
            y_train_flat = y_train.repeat(train_shp[1])
            f_train_flat = f_train.repeat(train_shp[1], axis=0)
            if len(o_train) > 0:
                o_train_flat = o_train.reshape(train_shp[0] * train_shp[1], train_shp[2])


        X_train_flat = np.array(X_train_flat)
        y_train_flat = np.array(y_train_flat)
        f_train_flat = np.array(f_train_flat)
        if len(o_train) > 0:
            o_train_flat = np.array(o_train_flat)
            X_train_shuff, y_train_shuff, f_train_shuff, o_train_shuff = shuffle_in_unison(X_train_flat, y_train_flat, f_train_flat, o=o_train_flat)
        else:
            X_train_shuff, y_train_shuff, f_train_shuff = shuffle_in_unison(X_train_flat, y_train_flat, f_train_flat)

        X_test_flat = X_test.reshape(test_shp[0] * test_shp[1], test_shp[2])
        y_test_flat = y_test.repeat(test_shp[1])
        f_test_flat = f_test.repeat(test_shp[1], axis=0)
        if len(o_test) > 0:
            o_test_flat = o_test.reshape(test_shp[0] * test_shp[1], test_shp[2])
            X_test_shuff, y_test_shuff, f_test_shuff, o_test_shuff = shuffle_in_unison(X_test_flat, y_test_flat, f_test_flat, o=o_test_flat)
        else:
            X_test_shuff, y_test_shuff, f_test_shuff = shuffle_in_unison(X_test_flat, y_test_flat, f_test_flat)

        X_dev_flat = X_dev.reshape(dev_shp[0] * dev_shp[1], dev_shp[2])
        y_dev_flat = y_dev.repeat(dev_shp[1])
        f_dev_flat = f_dev.repeat(dev_shp[1], axis=0)
        if len(o_dev) > 0:
            o_dev_flat = o_dev.reshape(dev_shp[0] * dev_shp[1], dev_shp[2])
            X_dev_shuff, y_dev_shuff, f_dev_shuff, o_dev_shuff = shuffle_in_unison(X_dev_flat, y_dev_flat, f_dev_flat, o=o_dev_flat)
        else:
            X_dev_shuff, y_dev_shuff, f_dev_shuff = shuffle_in_unison(X_dev_flat, y_dev_flat, f_dev_flat)

        # just clearing up space -- from now on, we use the flattened representations
        del X_train
        del X_test
        del X_dev

        iteration = list()
        iteration.append('fold' + str(itern))
        iteration.append((X_train_flat, X_train_shuff, y_train, y_train_flat, y_train_shuff, 
                          f_train, f_train_flat, f_train_shuff, o_train_flat, o_train_shuff, train_shp))
        iteration.append((X_test_flat, X_test_shuff, y_test, y_test_flat, y_test_shuff,
                          f_test, f_test_flat, f_test_shuff, o_test_flat, o_test_shuff, test_shp))
        iteration.append((X_dev_flat, X_dev_shuff, y_dev, y_dev_flat, y_dev_shuff,
                          f_dev, f_dev_flat, f_dev_shuff, o_dev_flat, o_dev_shuff, dev_shp))
        yield iteration
        

def get_threshold(gold, pred):
    maxth = 0.
    maxf1 = 0.
    vary = True
    start = 0.
    stop = 1.
    step = 0.05
    while vary:
        vary = False
        for threshold in np.arange(start, stop, step):
            pred_th = (pred >= threshold).astype(int)
            (acc, precision, recall, f1, microf1, macrof1, baseline, p) = bootstrap(gold, pred_th, printit=False)
            if f1 > maxf1:
                maxf1 = f1
                maxth = threshold
                print("threshold:", maxth, ", F1:", maxf1)
                vary = True
        start = maxth - step
        if start < 0.:
            start = 0.
        stop = maxth + step
        if stop > 1.0:
            stop = 1.0
        step = step * 0.1
        
    return maxth

def new_outs_lengths(input_lenght, kernel_size, padding=0, dilation=1, stride=1):
    return np.floor((input_lenght + 2*padding - dilation*(kernel_size-1) -1) / stride + 1)
        
    
#    
#   The Model
#


class MLP(nn.Module):
    def __init__(self, max_features, max_len, embedding_dim, hidden_size, feats=0):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.embs = nn.Embedding(max_features + 3, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_size)
        self.tanh1 = nn.Tanh()
        #self.linear2 = nn.Linear(hidden_size, hidden_size)
        #self.tanh2 = nn.Tanh()
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(hidden_size * (1 + feats * 2), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs, test_mode=False, domain=[None,None], o=None):
        embeds = self.embs(inputs)
        if o is not None:
            embeds = embeds.transpose(1,2).transpose(0,1)
            embeds =  torch.mul(embeds, o).transpose(0,1).transpose(1,2)
            nonzero = o
        else:
            nonzero = embeds
            nonzero = torch.sum(nonzero, 2)
            nonzero[nonzero!=0.]=1
        nonzero = torch.sum(nonzero, 0)
        nonzero[nonzero==0.]=1e-16
        nonzero = nonzero.repeat(embeds.size(2),1).transpose(0,1)
        out = torch.sum(embeds, 0) / nonzero
        out = self.tanh1(self.linear1(out))
        #out = self.tanh2(self.linear2(out))
        outc = out

        if domain[0] is not None or domain[1] is not None:
            if CUDA_MODE:
                zeros = Variable(torch.zeros(out.size()).cuda())
            else:            
                zeros = Variable(torch.zeros(out.size()))
        
        if domain[0] == 0:
            out = torch.cat((out,outc,zeros),1)
        elif domain[0] == 1:
            out = torch.cat((out,zeros,outc),1)
        elif domain[0] == 2:
            out = torch.cat((out,zeros,zeros),1)
      
        if domain[1] == 0:
            out = torch.cat((out,outc,zeros),1)
        elif domain[1] == 1:
            out = torch.cat((out,zeros,outc),1)
        elif domain[1] == 2:
            out = torch.cat((out,zeros,zeros),1)
                       
        if not test_mode:
            out = self.dropout(out)
        out = self.sigmoid(self.linear(out))

        return out


class GRU(nn.Module):
    def __init__(self, max_features, embedding_dim, hidden_size, feats=0):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.embs = nn.Embedding(max_features + 3, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, dropout=0.2)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(hidden_size * (1 + feats * 2), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs, test_mode=False, domain=[None,None]):
        embeds = self.embs(inputs)
        out, _ = self.gru(embeds)
        out = out[-1]
        outc = out.contiguous()
        out = outc

        if domain[0] is not None or domain[1] is not None: 
            if CUDA_MODE:
                zeros = Variable(torch.zeros(out.size()).cuda())
            else:            
                zeros = Variable(torch.zeros(out.size()))

        if domain[0] == 0:
            out = torch.cat((out,outc,zeros),1)
        elif domain[0] == 1:
            out = torch.cat((out,zeros,outc),1)
        elif domain[0] == 2:
            out = torch.cat((out,zeros,zeros),1)
      
        if domain[1] == 0:
            out = torch.cat((out,outc,zeros),1)
        elif domain[1] == 1:
            out = torch.cat((out,zeros,outc),1)
        elif domain[1] == 2:
            out = torch.cat((out,zeros,zeros),1)
        
                       
        if not test_mode:
            out = self.dropout(out)
        out = self.sigmoid(self.linear(out))
        return out.view(-1)


    
class CNN(nn.Module):
    def __init__(self, max_features, embedding_dim, seq_length, nb_filter,
                 filter_length, pool_length, hidden_size, feats=0, order=0):
        super(CNN, self).__init__()
        cnn_out_length = new_outs_lengths(seq_length, filter_length)
        pool_out_length = new_outs_lengths(cnn_out_length, pool_length, stride=pool_length)
        self.embs = nn.Embedding(max_features + 3, embedding_dim)
        self.cnn = nn.Conv1d(embedding_dim + order, nb_filter, filter_length)
        self.pool = nn.MaxPool1d(pool_length)
        self.linear1 = nn.Linear(int(pool_out_length) * nb_filter, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(hidden_size * (1 + feats * 2), 1)
        self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, inputs, intermediate=False, test_mode=False, domain=[None,None], o=None):
        embeds = self.embs(inputs)
        embeds = embeds.transpose(0, 1).transpose(1, 2)
        if o is not None:
            o = o.transpose(0, 1)
            o = o.view(o.size()[0],1,o.size()[1])
            embeds = torch.cat((embeds,o),1)
        outc = self.cnn(embeds)
        outc = self.pool(outc)
        outc = outc.view((outc.size()[0], outc.size()[1] * outc.size()[2]))
        outc = self.relu1(self.linear1(outc))
        out = outc
        
        if domain[0] is not None or domain[1] is not None: 
            if CUDA_MODE:
                zeros = Variable(torch.zeros(out.size()).cuda())
            else:            
                zeros = Variable(torch.zeros(out.size()))
        
        if domain[0] == 0:
            out = torch.cat((out,outc,zeros),1)
        elif domain[0] == 1:
            out = torch.cat((out,zeros,outc),1)
        elif domain[0] == 2:
            out = torch.cat((out,zeros,zeros),1)
      
        if domain[1] == 0:
            out = torch.cat((out,outc,zeros),1)
        elif domain[1] == 1:
            out = torch.cat((out,zeros,outc),1)
        elif domain[1] == 2:
            out = torch.cat((out,zeros,zeros),1)
                       
        if not test_mode:
            out = self.dropout1(out)
        out = self.sigmoid2(self.linear2(out))

        return out


        
def predict(net, x, f, o, batch_size, intermediate=False, domain=[False,False]):
    pred = np.empty(0)
    batches = math.ceil(x.size()[0] / batch_size)
    for b in range(batches):
        bx = x[b*batch_size:b*batch_size+batch_size]
        if o is not None:
            bo = o[b*batch_size:b*batch_size+batch_size]


        # No domain
        if domain[0] == domain[1] == False:
            bx = torch.transpose(bx, 0, 1)
            if o is not None:
                bo = torch.transpose(bo, 0, 1)
                b_pred = net(bx, test_mode=True, o=bo)
                del(bx,bo)
            else:
                b_pred = net(bx, test_mode=True)
                del(bx)
            
        # Only one domain
        elif domain[0] == False or domain[1] == False:
            bf = f[b*batch_size:b*batch_size+batch_size]
            fb = None
            mb = None
            ub = None

            if domain[1] == False:
                idx = torch.np.where(bf[:,0]==0)[0]
                if np.shape(idx)[0] == 0:
                    fb = torch.LongTensor()
                else:
                    fb = torch.LongTensor(idx)
            else:
                idx = torch.np.where(bf[:,2]==0)[0]
                if np.shape(idx)[0] == 0:
                    fb = torch.LongTensor()
                else:
                    fb = torch.LongTensor(idx)  
            if CUDA_MODE:
                fb = fb.cuda()
                f_pred = Variable(torch.LongTensor().cuda())
            else:
                f_pred = Variable(torch.LongTensor())
            if fb.dim() > 0:
                bxf = bx[fb]
                bxf = torch.transpose(bxf, 0, 1)
                if o is not None:
                    bof = bo[fb]
                    bof = torch.transpose(bof, 0, 1)
                else:
                    bof = None

                if domain[1] == False:
                    f_pred = net(bxf, test_mode=True, domain=[0,None], o=bof)
                else:
                    f_pred = net(bxf, test_mode=True, domain=[None,0])
                del(bxf, bof)

            if domain[1] == False:
                idx = torch.np.where(bf[:,0]==1)[0]
                if np.shape(idx)[0] == 0:
                    mb = torch.LongTensor()
                else:
                    mb = torch.LongTensor(idx)
            else:
                idx = torch.np.where(bf[:,2]==1)[0]
                if np.shape(idx)[0] == 0:
                    mb = torch.LongTensor()
                else:
                    mb = torch.LongTensor(idx)  
            if CUDA_MODE:
                mb = mb.cuda()
                m_pred = Variable(torch.LongTensor().cuda())
            else:
                m_pred = Variable(torch.LongTensor())
            if mb.dim() > 0:
                bxm = bx[mb]
                bxm = torch.transpose(bxm, 0, 1)
                if o is not None:
                    bom = bo[mb]
                    bom = torch.transpose(bom, 0, 1)
                else:
                    bom = None
                if domain[1] == False:
                    m_pred = net(bxm, test_mode=True, domain=[1,None], o=bom)
                else:
                    m_pred = net(bxm, test_mode=True, domain=[None,1], o=bom)
                del(bxm, bom)

            # UNK/UNK

            if domain[1] == False:
                idx = torch.np.where(bf[:,0]==None)[0]
                if np.shape(idx)[0] == 0:
                    ub = torch.LongTensor()
                else:
                    ub = torch.LongTensor(idx)
            else:
                idx = torch.np.where(bf[:,2]==None)[0]
                if np.shape(idx)[0] == 0:
                    ub = torch.LongTensor()
                else:
                    ub = torch.LongTensor(idx)  
            if CUDA_MODE:
                ub = ub.cuda()
                u_pred = Variable(torch.LongTensor().cuda())
            else:
                u_pred = Variable(torch.LongTensor())
            if ub.dim() > 0:
                bxu = bx[ub]
                bxu = torch.transpose(bxu, 0, 1)
                if o is not None:
                    bou = bo[ub]
                    bou = torch.transpose(bou, 0, 1)
                else:
                    bou = None
                if domain[1] == False:
                    u_pred = net(bxu, test_mode=True, domain=[2,None], o=bou)
                else:
                    u_pred = net(bxu, test_mode=True, domain=[None,2], o=bou)
                del(bxu, bou)
    
            del(bf)


            if fb.dim() > 0:
                cb = fb
                del(fb)
                b_pred = f_pred
                del(f_pred)
            if mb.dim() > 0:
                if 'cb' in locals():
                    cb = torch.cat((cb, mb))
                else:
                    cb = mb
                del(mb)
                if 'b_pred' in locals():
                    b_pred = torch.cat((b_pred, m_pred))
                else:
                    b_pred = m_pred
                del(m_pred)
            if ub.dim() > 0:
                if 'cb' in locals():
                    cb = torch.cat((cb, ub))
                else:
                    cb = ub
                del(ub)
                if 'b_pred' in locals():
                    b_pred = torch.cat((b_pred, u_pred))
                else:
                    b_pred = u_pred
                del(u_pred)
                    
            # if fb.dim() > 0 and mb.dim() > 0:
            #     cb = torch.cat((fb, mb))
            #     del(fb, mb)
            #     b_pred = torch.cat((f_pred, m_pred))
            #     del(f_pred, m_pred)
            # elif fb.dim() > 0:
            #     cb = fb
            #     del(fb)
            #     b_pred = f_pred
            #     del(f_pred)
            # else:
            #     cb = mb
            #     del (mb)
            #     b_pred = m_pred
            #     del(m_pred)
    
            if CUDA_MODE:
                cb = torch.LongTensor(torch.np.argsort(cb.cpu().numpy())).cuda()
            else:
                cb = torch.LongTensor(torch.np.argsort(cb.numpy()))    
    
            b_pred = b_pred[cb]
            del(cb)
            
        # Two domains
        else:
            bf = f[b*batch_size:b*batch_size+batch_size]

            fnb = torch.LongTensor()
            if(torch.np.where((bf[:,0]==0) & (bf[:,2]==0))[0].size > 0):
                fnb = torch.LongTensor(torch.np.where((bf[:,0]==0) & (bf[:,2]==0))[0])
                if CUDA_MODE:
                    fnb = fnb.cuda()
                    fn_pred = Variable(torch.LongTensor().cuda())
                else:
                    fn_pred = Variable(torch.LongTensor())
                if fnb.dim() > 0:
                    bxfn = bx[fnb]
                    bxfn = torch.transpose(bxfn, 0, 1)
                    fn_pred = net(bxfn, test_mode=True, domain=[0,0])
                    del(bxfn)

            ftb = torch.LongTensor()
            if (torch.np.where((bf[:,0]==0) & (bf[:,2]==1))[0].size > 0):
                ftb = torch.LongTensor(torch.np.where((bf[:,0]==0) & (bf[:,2]==1))[0])
                if CUDA_MODE:
                    ftb = ftb.cuda()
                    ft_pred = Variable(torch.LongTensor().cuda())
                else:
                    ft_pred = Variable(torch.LongTensor())
                if ftb.dim() > 0:
                    bxft = bx[ftb]
                    bxft = torch.transpose(bxft, 0, 1)
                    ft_pred = net(bxft, test_mode=True, domain=[0,1])
                    del(bxft)

            mnb = torch.LongTensor()
            if (torch.np.where((bf[:,0]==1) & (bf[:,2]==0))[0].size > 0):
                mnb = torch.LongTensor(torch.np.where((bf[:,0]==1) & (bf[:,2]==0))[0])
                if CUDA_MODE:
                    mnb = mnb.cuda()
                    mn_pred = Variable(torch.LongTensor().cuda())
                else:
                    mn_pred = Variable(torch.LongTensor())
                if mnb.dim() > 0:
                    bxmn = bx[mnb]
                    bxmn = torch.transpose(bxmn, 0, 1)
                    mn_pred = net(bxmn, test_mode=True, domain=[1,0])
                    del(bxmn)

            mtb = torch.LongTensor()
            if (torch.np.where((bf[:,0]==1) & (bf[:,2]==1))[0].size > 0):
                mtb = torch.LongTensor(torch.np.where((bf[:,0]==1) & (bf[:,2]==1))[0])
                if CUDA_MODE:
                    mtb =mtb.cuda()
                    mt_pred = Variable(torch.LongTensor().cuda())
                else:
                    mt_pred = Variable(torch.LongTensor())
                if mtb.dim() > 0:
                    bxmt = bx[mtb]
                    bxmt = torch.transpose(bxmt, 0, 1)
                    mt_pred = net(bxmt, test_mode=True, domain=[1,1])
                    del(bxmt)

            del(bx, bf)
            
            b_list = list()
            pred_list = list()
            if fnb.dim() > 0:
                b_list.append(fnb)
                del(fnb)
                pred_list.append(fn_pred)
                del(fn_pred)
            if ftb.dim() > 0:
                b_list.append(ftb)
                del(ftb)
                pred_list.append(ft_pred)
                del(ft_pred)
            if mnb.dim() > 0:
                b_list.append(mnb)
                del(mnb)
                pred_list.append(mn_pred)
                del(mn_pred)
            if mtb.dim() > 0:
                b_list.append(mtb)
                del(mtb)
                pred_list.append(mt_pred)
                del(mt_pred)
            
            cb = torch.cat(b_list)
            del(b_list)
            b_pred = torch.cat(pred_list)
            del(pred_list)
            
            if CUDA_MODE:
                cb = torch.LongTensor(torch.np.argsort(cb.cpu().numpy())).cuda()
            else:
                cb = torch.LongTensor(torch.np.argsort(cb.numpy()))    
    
            b_pred = b_pred[cb]
            del(cb)
            
        sys.stdout.write('\r[batch: %3d/%3d]' % (b + 1, batches))
        sys.stdout.flush()
        if CUDA_MODE:     
            pred = np.concatenate((pred, b_pred.cpu().data.numpy().flatten()))
        else:
            pred = np.concatenate((pred, b_pred.data.numpy().flatten()))
        del(b_pred)
    sys.stdout.write('\n')
    sys.stdout.flush()
    return pred

    
def train(net, x, y, f, o, nepochs, batch_size, domain=[False,False], class_weight=0.5):
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters)
    batches = math.ceil(x.size()[0] / batch_size)
    for e in range(nepochs):
        running_loss = 0.0
        for b in range(batches):
            # Get minibatch samples
            bx = x[b*batch_size:b*batch_size+batch_size]
            by = y[b*batch_size:b*batch_size+batch_size]
            if o is not None:
                bo = o[b*batch_size:b*batch_size+batch_size]

            # Clear gradients
            net.zero_grad()

            # No domain
            if domain[0] == domain[1] == False:
                bx = torch.transpose(bx, 0, 1)
                if o is not None:
                    bo = torch.transpose(bo, 0, 1)
                    y_pred = net(bx, o=bo)
                    del(bx, bo)
                else:
                    y_pred = net(bx)
                    del(bx)
                    
            # Only one domain
            elif domain[0] == False or domain[1] == False:
                bf = f[b*batch_size:b*batch_size+batch_size]
                fb = None
                mb = None
                ub = None

                # FEMALE/NO-RETWEET

                if domain[1] == False:
                    idx = torch.np.where(bf[:,0]==0)[0]
                    if np.shape(idx)[0] == 0:
                        fb = torch.LongTensor()
                    else:
                        fb = torch.LongTensor(idx)
                else:
                    idx = torch.np.where(bf[:,2]==0)[0]
                    if np.shape(idx)[0] == 0:
                        fb = torch.LongTensor()
                    else:
                        fb = torch.LongTensor(idx)                      
                if CUDA_MODE:
                    fb = fb.cuda()
                
                if fb.dim() > 0:
                    bxf = bx[fb]
                    byf = by[fb]
                    bxf = torch.transpose(bxf, 0, 1)
                    if o is not None:
                        bof = bo[fb]
                        bof = torch.transpose(bof, 0, 1)
                    else:
                        bof = None

                # MALE/RETWEET

                if domain[1] == False:
                    idx = torch.np.where(bf[:,0]==1)[0]
                    if np.shape(idx)[0] == 0:
                        mb = torch.LongTensor()
                    else:
                        mb = torch.LongTensor(idx)
                else:
                    idx = torch.np.where(bf[:,2]==1)[0]
                    if np.shape(idx)[0] == 0:
                        mb = torch.LongTensor()
                    else:
                        mb = torch.LongTensor(idx)             
                if CUDA_MODE:
                    mb = mb.cuda()

                if mb.dim() > 0:
                    bxm = bx[mb]
                    bym = by[mb]
                    bxm = torch.transpose(bxm, 0, 1)
                    if o is not None:
                        bom = bo[mb]
                        bom = torch.transpose(bom, 0, 1)
                    else:
                        bom = None

                # UNK/UNK

                if domain[1] == False:
                    idx = torch.np.where(bf[:,0]==None)[0]
                    if np.shape(idx)[0] == 0:
                        ub = torch.LongTensor()
                    else:
                        ub = torch.LongTensor(idx)
                else:
                    idx = torch.np.where(bf[:,2]==None)[0]
                    if np.shape(idx)[0] == 0:
                        ub = torch.LongTensor()
                    else:
                        ub = torch.LongTensor(idx)             
                if CUDA_MODE:
                    ub = ub.cuda()

                if ub.dim() > 0:
                    bxu = bx[ub]
                    byu = by[ub]
                    bxu = torch.transpose(bxu, 0, 1)
                    if o is not None:
                        bou = bo[ub]
                        bou = torch.transpose(bou, 0, 1)                                
                    else:
                        bou = None

                del(bx, bf)
                if o is not None:
                    del(bo)


                # Forward pass
                if domain[1] == False:
                    if fb.dim() > 0:
                        yf_pred = net(bxf, domain=[0,None], o=bof)
                        del(bxf, bof)
                    if mb.dim() > 0:
                        ym_pred = net(bxm, domain=[1,None], o=bom)
                        del(bxm, bom)
                    if ub.dim() > 0:
                        yu_pred = net(bxu, domain=[2,None], o=bou)
                        del(bxu, bou)
                else:
                    if fb.dim() > 0:
                        yf_pred = net(bxf, domain=[None,0], o=bof)
                        del(bxf, bof)
                    if mb.dim() > 0:
                        ym_pred = net(bxm, domain=[None,1], o=bom)
                        del(bxm, bom)
                    if ub.dim() > 0:
                        yu_pred = net(bxu, domain=[None,2], o=bou)
                        del(bxu, bou)

                if fb.dim() > 0:
                    cb = fb
                    del(fb)
                    y_pred = yf_pred
                    del(yf_pred)
                if mb.dim() > 0:
                    if 'cb' in locals():
                        cb = torch.cat((cb, mb))
                    else:
                        cb = mb
                    del(mb)
                    if 'y_pred' in locals():
                        y_pred = torch.cat((y_pred, ym_pred))
                    else:
                        y_pred = ym_pred
                    del(ym_pred)
                if ub.dim() > 0:
                    if 'cb' in locals():
                        cb = torch.cat((cb, ub))
                    else:
                        cb = ub
                    del(ub)
                    if 'y_pred' in locals():
                        y_pred = torch.cat((y_pred, yu_pred))
                    else:
                        y_pred = yu_pred
                    del(yu_pred)



                # if fb.dim() > 0 and mb.dim() > 0:
                # #    cb = torch.cat((fb, mb))
                #     del(fb, mb)
                #     y_pred = torch.cat((yf_pred, ym_pred))
                #     del(yf_pred, ym_pred)
                # elif fb.dim() > 0:
                # #    cb = fb
                #     del(fb)
                #     y_pred = yf_pred
                #     del(yf_pred)
                # else:
                # #    cb = mb
                #     del (mb)
                #     y_pred = ym_pred
                #     del(ym_pred)


                if CUDA_MODE:
                   cb = torch.LongTensor(torch.np.argsort(cb.cpu().numpy())).cuda()
                else:
                   cb = torch.LongTensor(torch.np.argsort(cb.numpy()))    
    
                y_pred = y_pred[cb]
                del(cb)


                #by = torch.cat((byf, bym))
                #del(byf, bym)
                #y_pred = torch.cat((yf_pred, ym_pred))
                #del(yf_pred, ym_pred)

            # Two domains                
            else:
                bf = f[b*batch_size:b*batch_size+batch_size]

                fnb = torch.LongTensor(torch.np.where((bf[:,0]==0) & (bf[:,2]==0))[0])
                if CUDA_MODE:
                    fnb = fnb.cuda()
                if fnb.dim() > 0:
                    bxfn = bx[fnb]
                    byfn = by[fnb]
                    bxfn = torch.transpose(bxfn, 0, 1)
                    yfn_pred = net(bxfn, domain=[0,0])
                    del(bxfn)
                
                ftb = torch.LongTensor(torch.np.where((bf[:,0]==0) & (bf[:,2]==1))[0])
                if CUDA_MODE:
                    ftb = ftb.cuda()
                if ftb.dim() > 0:
                    bxft = bx[ftb]
                    byft = by[ftb]
                    bxft = torch.transpose(bxft, 0, 1)
                    yft_pred = net(bxft, domain=[0,1])
                    del(bxft)
                                
                mnb = torch.LongTensor(torch.np.where((bf[:,0]==1) & (bf[:,2]==0))[0])
                if CUDA_MODE:
                    mnb = mnb.cuda()
                if mnb.dim() > 0:
                    bxmn = bx[mnb]
                    bymn = by[mnb]
                    bxmn = torch.transpose(bxmn, 0, 1)
                    ymn_pred = net(bxmn, domain=[1,0])
                    del(bxmn)
        
                mtb = torch.LongTensor(torch.np.where((bf[:,0]==1) & (bf[:,2]==1))[0])
                if CUDA_MODE:
                    mtb = mtb.cuda()
                if mtb.dim() > 0:
                    bxmt = bx[mtb]
                    bymt = by[mtb]
                    bxmt = torch.transpose(bxmt, 0, 1)
                    ymt_pred = net(bxmt, domain=[1,1])
                    del(bxmt)

                del(bx, bf)
                
                by_list = list()
                ypred_list = list()
                if fnb.dim() > 0:
                    by_list.append(byfn)
                    del(byfn)
                    ypred_list.append(yfn_pred)
                    del(yfn_pred)
                    del(fnb)
                if ftb.dim() > 0:
                    by_list.append(byft)
                    del(byft)
                    ypred_list.append(yft_pred)
                    del(yft_pred)
                    del(ftb)
                if mnb.dim() > 0:
                    by_list.append(bymn)
                    del(bymn)
                    ypred_list.append(ymn_pred)
                    del(ymn_pred)
                    del(mnb)
                if mtb.dim() > 0:
                    by_list.append(bymt)
                    del(bymt)
                    ypred_list.append(ymt_pred)
                    del(ymt_pred)
                    del(mtb)
                
                by = torch.cat(by_list)                
                del(by_list)
                y_pred = torch.cat(ypred_list)
                del(ypred_list)
                                
            # Compute loss
            byW = [class_weight if cy == 1. else 1. - class_weight for cy in by.data]
            byW = Variable(torch.FloatTensor([byW]))
            criterion = nn.BCELoss(weight=byW)
            loss = criterion(y_pred, by)
            del(by)
            
            # Print loss
            running_loss += loss.data[0]
            sys.stdout.write('\r[epoch: %3d, batch: %3d/%3d] loss: %.3f' % (e + 1, b + 1, batches, running_loss / (b+1)))
            sys.stdout.flush()

            # Backward propagation and update the weights.
            loss.backward()
            net.embs.weight.grad.data[0:3,:].fill_(0)
            optimizer.step()
            del(y_pred)
        sys.stdout.write('\n')
        sys.stdout.flush()

#max_features = 20000
#max_features = 50000
#max_features = 100000
maxtweets = 2000
maxlen = 50  # cut texts to this number of words (among top max_features most common words)
emb_dim = 200
hidden_dim = 128
nb_filter = 64 # how many convolutional filters
filter_length = 5 # how many tokens a convolution covers
pool_length = 4 # how many cells of convolution to pool across when maxing
nb_epoch = 1 # how many training epochs
batch_size = 256 # how many tweets to train at a time
predict_batch_size = 612

if args.setting is not None:
    config = configparser.ConfigParser()
    config.read(args.setting)
    maxtweets = int(config['SETTING']['maxtweets'])
    maxlen = int(config['SETTING']['maxlen'])
    emb_dim = int(config['SETTING']['emb_dim'])
    hidden_dim = int(config['SETTING']['hidden_dim'])
    nb_filter = int(config['SETTING']['nb_filter'])
    filter_length = int(config['SETTING']['filter_length'])
    pool_length = int(config['SETTING']['pool_length'])
    nb_epoch = int(config['SETTING']['nb_epoch'])
    batch_size = int(config['SETTING']['batch_size'])
    predict_batch_size = int(config['SETTING']['predict_batch_size'])

embeddings, max_features = load_embeddings(nb_words=max_features, emb_dim=emb_dim, w2v=emb_file)

if run_fold is not None:
    run_fold = ['fold' + str(rf) for rf in run_fold]
pos, neg = load_data(nb_words=max_features, maxlen=maxlen, seed=SEED)

predictions = dict()
if not vary_th:
    predictions["cnnv"] = list()
    predictions["cnnw"] = list()
else:
    predictions["cnnv"] = dict()
    predictions["cnnw"] = dict()
gold_test = list()
iterations = list()
#foldsfile = "folds.csv"
#foldsfile = "foldsrisk.csv"
foldsfile = "foldsriskdict.csv"
#foldsfile = "foldsdow.csv"
#foldsfile = "data_toy/folds.csv"
for iteration in gen_iterations(pos, neg, max_features, maxtweets, maxlen, foldsfile, relevant=tweet_filter):
    iterid = iteration[0]
    if run_fold is not None and iterid not in run_fold:
        continue
    iterations.append(iterid)
    print('')
    print('Iteration: %s' % iterid)
    (_, X_train_shuff, _, _, y_train_shuff,
     _, _, f_train_shuff, _, o_train_shuff,train_shp) = iteration[1]
    (X_test_flat, _, y_test, _, _,
     _, f_test_flat, _, o_test_flat, _, test_shp) = iteration[2]
    (X_dev_flat, _, y_dev, _, _,
     _, f_dev_flat, _, o_dev_flat, _, dev_shp) = iteration[3]
   
    if not wordorder:
        o_train_shuff = None
        o_test_flat = None
        o_dev_flat = None

    print('X_train shape:', train_shp)
    print('X_test shape:', test_shp)
    print('X_dev shape:', dev_shp)
 
    order = 0
    if o_train_shuff is not None:
        order = 1
     
    gold_dev = y_dev.flatten()
    if dev_mode:
        gold_test.extend(y_dev.flatten())
    else:
        gold_test.extend(y_test.flatten())

    if not (os.path.isfile(pred_dir + 'cnnv_' + iterid + '.pkl') and 
        os.path.isfile(pred_dir + 'cnnw_' + iterid + '.pkl')):
        #
        # Pre-train tweet-level vectors
        #
    
        print('Build first model (tweet-level)...')
        num_feats = int(np.sum(np.array(domain)==True))
        if model_gru:
            net = GRU(max_features, emb_dim, hidden_dim, feats=num_feats)
        if model_mlp:
            net = MLP(max_features, maxlen, emb_dim, hidden_dim, feats=num_feats)
        else:
            net = CNN(max_features, emb_dim, maxlen, nb_filter, filter_length, pool_length, hidden_dim, feats=num_feats, order=order)
        print(net)
        if freeze:
            net.embs.weight.requires_grad = False

        # Train or load the model
        if (os.path.isfile(model_dir + 'tweet_classifier_' + iterid + '.pkl')):
            print('Loading model weights...')
            net.load_state_dict(torch.load(model_dir + 'tweet_classifier_' + iterid + '.pkl'))
            if CUDA_MODE:
                net = net.cuda()
        else:
            net.embs.weight.data.copy_(torch.from_numpy(np.array(embeddings)))

            if train_percent is not None:
                if train_percent == 1.0:
                    ceil = len(X_train_shuff)
                    floor = 0
                else:
                    ceil = int(len(X_train_shuff) * train_percent)
                    floor = np.random.choice(np.arange(0,len(X_train_shuff)-ceil))
                X_train_shuff = X_train_shuff[floor:floor+ceil]
                y_train_shuff = y_train_shuff[floor:floor+ceil]
                f_train_shuff = f_train_shuff[floor:floor+ceil]
                if o_train_shuff is not None:
                    o_train_shuff = o_train_shuff[floor:floor+ceil]


            data_o = None
            if CUDA_MODE:
                net = net.cuda()
                data_x = Variable(torch.from_numpy(X_train_shuff).long().cuda())
                data_y = Variable(torch.from_numpy(y_train_shuff).float().cuda())
                if o_train_shuff is not None:
                    data_o = Variable(torch.from_numpy(o_train_shuff).float().cuda())
            else:
                data_x = Variable(torch.from_numpy(X_train_shuff).long())
                data_y = Variable(torch.from_numpy(y_train_shuff).float())
                if o_train_shuff is not None:
                    data_o = Variable(torch.from_numpy(o_train_shuff).float())
            data_f = f_train_shuff
            
            print('Train...')
            train(net, data_x, data_y, data_f, data_o, nb_epoch, batch_size, domain=domain, class_weight=posW)
            del(data_x, data_y, data_f, data_o)
            torch.save(net.state_dict(), model_dir + 'tweet_classifier_' + iterid + '.pkl')
            
    
        #
        #  CNN+V/CNN+W
        #

        if fixed_th is not None:
            print('Threshold fixed to: %f' % fixed_th)
            thldmn = fixed_th
            thldwm = fixed_th

        elif not vary_th:
            # Prediction for DEV set
            print('Dev...')
            data_o = None
            if CUDA_MODE:
                data_x = Variable(torch.from_numpy(X_dev_flat).long().cuda())
                if o_dev_flat is not None:
                    data_o = Variable(torch.from_numpy(o_dev_flat).float().cuda())
            else:
                data_x = Variable(torch.from_numpy(X_dev_flat).long())
                if o_dev_flat is not None:
                    data_o = Variable(torch.from_numpy(o_dev_flat).float())
            data_f = f_dev_flat
        
            predDev = predict(net, data_x, data_f, data_o, predict_batch_size, domain=domain)
            del(data_x, data_f, data_o)
            predDev = predDev.reshape((dev_shp[0], dev_shp[1]))

            wts = np.linspace(1., 0.01, maxtweets)
            if outliers:
                min_out = np.mean(predDev) - np.std(predDev)
                max_out = np.mean(predDev) + np.std(predDev)
                predDevmn = list()
                predDevwm = list()
                for account in predDev:
                    tweets = list()
                    weights = list()
                    for i in range(0,len(account)):
                        if account[i] > max_out or account[i] < min_out:
                            tweets.append(account[i])
                            weights.append(wts[i])
                    if len(tweets) > 0:
                        predDevmn.append(np.mean(tweets))
                        predDevwm.append(np.average(tweets, weights=weights))
                    else:
                        predDevmn.append(0.0)
                        predDevwm.append(0.0)
                predDevmn = np.array(predDevmn)
                predDevwm = np.array(predDevwm)
            else:
                predDevmn = np.mean(predDev, axis=1)
                predDevwm = np.average(predDev, axis=1, weights=wts)

            print('Search CNN+V threshold')
            thldmn = get_threshold(gold_dev, predDevmn)        
            print('Search CNN+W threshold')
            thldwm = get_threshold(gold_dev, predDevwm)

        if dev_mode:
            if tweetrel:
                tweetrelfile = open(tweetrel_dir + 'cnnv_' + iterid + '.pkl', 'wb')
                pkl.dump(predDev, tweetrelfile)
                pkl.dump(gold_dev, tweetrelfile)
                tweetrelfile.close()

            predDevmn = (predDevmn >= thldmn).astype(int)
            predfile = open(pred_dir + 'cnnv_' + iterid + '.pkl', 'wb')
            pkl.dump(predDevmn, predfile)
            predfile.close()
            del(predDevmn)

            predDevwm = (predDevwm >= thldwm).astype(int)
            predfile = open(pred_dir + 'cnnw_' + iterid + '.pkl', 'wb')
            pkl.dump(predDevwm, predfile)
            predfile.close()
            del(predDevwm)
            del(predDev, gold_dev)
            
        else:
            #Prediction for TEST set
            try:
                del(predDevmn)
                del(predDevwm)
                del(predDev, gold_dev)
            except NameError:
                pass

            print('Test...')
            data_o = None
            if CUDA_MODE:
                data_x = Variable(torch.from_numpy(X_test_flat).long().cuda())
                if o_test_flat is not None:
                    data_o = Variable(torch.from_numpy(o_test_flat).float().cuda())
            else:
                data_x = Variable(torch.from_numpy(X_test_flat).long())
                if o_test_flat is not None:
                    data_o = Variable(torch.from_numpy(o_test_flat).float())
            data_f = f_test_flat
            predTest = predict(net, data_x, data_f, data_o, predict_batch_size, domain=domain)
            del(data_x, data_f, data_o)
            predTest = predTest.reshape((test_shp[0], test_shp[1]))

            wts = np.linspace(1., 0.01, maxtweets)
            if outliers:
                min_out = np.mean(predTest) - np.std(predTest)
                max_out = np.mean(predTest) + np.std(predTest)
                predTestmn = list()
                predTestwm = list()
                for account in predTest:
                    tweets = list()
                    weights = list()
                    for i in range(0,len(account)):
                        if account[i] > max_out or account[i] < min_out:
                            tweets.append(account[i])
                            weights.append(wts[i])
                    if len(tweets) > 0:
                        predTestmn.append(np.mean(tweets))
                        predTestwm.append(np.average(tweets, weights=weights))
                    else:
                        predTestmn.append(0.0)
                        predTestwm.append(0.0)
                predTestmn = np.array(predTestmn)
                predTestwm = np.array(predTestwm)
            else:
                predTestmn = np.mean(predTest, axis=1)
                predTestwm = np.average(predTest, axis=1, weights=wts)

            if not vary_th:
                print('CNN+V with threshold = ', thldmn)
                predTestmn = (predTestmn >= thldmn).astype(int)
                predfile = open(pred_dir + 'cnnv_' + iterid + '.pkl', 'wb')
                pkl.dump(predTestmn, predfile)
                predfile.close()
                del(predTestmn)

                print('CNN+W with threshold = ', thldwm)
                predTestwm = (predTestwm >= thldwm).astype(int)
                predfile = open(pred_dir + 'cnnw_' + iterid + '.pkl', 'wb')
                pkl.dump(predTestwm, predfile)
                predfile.close()
                del(predTestwm)
            else:
                predTestmnthld = list()
                predTestwmthld = list()
                start = 0.
                stop = 1.
                step = 0.005
                for thld in np.arange(start, stop, step):
                    print('CNN+V with threshold = ', thld)
                    predTestmnthld.append((thld, (predTestmn >= thld).astype(int)))

                    print('CNN+W with threshold = ', thld)
                    predTestwmthld.append((thld, (predTestwm >= thld).astype(int)))
                    
                predTestmn = predTestmnthld
                predTestwm = predTestwmthld

                predfile = open(pred_dir + 'cnnv_' + iterid + '.pkl', 'wb')
                pkl.dump(predTestmn, predfile)
                predfile.close()
                del(predTestmn)

                predfile = open(pred_dir + 'cnnw_' + iterid + '.pkl', 'wb')
                pkl.dump(predTestwm, predfile)
                predfile.close()
                del(predTestwm)

if run_fold is None:
    if not vary_th:
        for iterid in iterations:
            print(iterid + ': Loading cnn prediction files...')
            predfile = open(pred_dir + 'cnnv_' + iterid + '.pkl', 'rb')
            predTestmn = pkl.load(predfile)
            predictions["cnnv"].extend(predTestmn)
            predfile.close()

            predfile = open(pred_dir + 'cnnw_' + iterid + '.pkl', 'rb')
            predTestwm = pkl.load(predfile)
            predictions["cnnw"].extend(predTestwm)
            predfile.close()

        gold_test = np.array(gold_test)
        print("\nResults")
        print("\nCNN+V")
        bootstrap(gold_test, np.array(predictions["cnnv"]))
        predfile = open(pred_dir + 'cnnv.pkl', 'wb')
        pkl.dump(predictions["cnnv"], predfile)
        predfile.close()
        print("\nCNN+W")
        bootstrap(gold_test, np.array(predictions["cnnw"]))
        predfile = open(pred_dir + 'cnnw.pkl', 'wb')
        pkl.dump(predictions["cnnw"], predfile)
        predfile.close()

    else:
        for iterid in iterations:
            print(iterid + ': Loading cnn prediction files...')
            predfile = open(pred_dir + 'cnnv_' + iterid + '.pkl', 'rb')
            predTestmn = pkl.load(predfile)
            for (thld,predTest) in predTestmn:
                #thld = str(thld)
                if thld not in predictions["cnnv"]:
                    predictions["cnnv"][thld] = list()
                predictions["cnnv"][thld].extend(predTest)
            predfile.close()

            predfile = open(pred_dir + 'cnnw_' + iterid + '.pkl', 'rb')
            predTestwm = pkl.load(predfile)
            for (thld,predTest) in predTestwm:
                #thld = str(thld)
                if thld not in predictions["cnnw"]:
                    predictions["cnnw"][thld] = list()
                predictions["cnnw"][thld].extend(predTest)
            predfile.close()


        gold_test = np.array(gold_test)
        print("\nResults")
        print("\nCNN+V")
        for thld in sorted(predictions["cnnv"]):
            print("\nThreshold:",thld)
            bootstrap(gold_test, np.array(predictions["cnnv"][thld]))
        predfile = open(pred_dir + 'cnnv.pkl', 'wb')
        pkl.dump(predictions["cnnv"], predfile)
        predfile.close()

        print("\nCNN+W")
        for thld in sorted(predictions["cnnw"]):
            print("\nThreshold:",thld)
            bootstrap(gold_test, np.array(predictions["cnnw"][thld]))
        predfile = open(pred_dir + 'cnnw.pkl', 'wb')
        pkl.dump(predictions["cnnw"], predfile)
        predfile.close()
