#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:03:00 2017

@author: egoitz
"""
import sys
import pickle as pkl
import numpy as np
np.random.seed(947)

from keras.preprocessing import sequence


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
    #print('maximum # tweets: %i' % mt)
    
    if maxlen is not None:
        ml = maxlen
    else:
        ml = find_longest(sequences)
    #print('maximum tweet length: %i' % ml)
        
    x = (np.ones((nb_samples, mt, ml)) * value).astype(dtype)
    #print('x shape: ', x.shape)
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


def load_data(path='ow3d.pkl', nb_words=None, skip_top=0,
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
    
    print ('Loading data...')
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    ((x_pos, i_pos), y_pos) = pkl.load(f)
    ((x_neg, i_neg), y_neg) = pkl.load(f)
    
    f.close()

    # randomize datum order
    np.random.seed(seed)
    np.random.shuffle(x_pos)
    np.random.seed(seed)
    np.random.shuffle(y_pos)
    np.random.seed(seed)
    np.random.shuffle(i_pos)

    np.random.seed(seed * 2)
    np.random.shuffle(x_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(y_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(i_neg)

#    # keep maxlen words of each tweet
#    if maxlen is not None:
#        x_pos = cap_length(x_pos, maxlen)
#        x_neg = cap_length(x_neg, maxlen)
#
#    # cut off infrequent words to vocab of size nb_words
#    if nb_words is not None:
#        x_pos = cap_words(x_pos, nb_words, oov)
#        x_neg = cap_words(x_neg, nb_words, oov)
#
#    # cut off most frequent skip_top words
#    if skip_top > 0:
#        x_pos = skip_n(x_pos, skip_top, oov)
#        x_neg = skip_n(x_neg, skip_top, oov)

    # prepend each sequence with start and raise indices by index_from
    x_pos = push_indices(x_pos, start, index_from)
    x_neg = push_indices(x_neg, start, index_from)
    
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)
    i_pos = np.array(i_pos)

    x_neg = np.array(x_neg)
    y_neg = np.array(y_neg)
    i_neg = np.array(i_neg)
    
    return (x_pos, y_pos, i_pos), (x_neg, y_neg, i_neg)


def load_embeddings(nb_words=None, emb_dim=200, index_from=3,
                    vocab='ow3d.dict.pkl', 
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

    f.close()
    print("")
    print('Found %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.zeros((max_features+index_from, emb_dim))
    for word, i in word_index.items():
        if i < max_features:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i+index_from] = embedding_vector

    return embedding_matrix

def load_folds(file, seed=113):
    print ("Loading folds...")
    folds = list(list() for i in range(10))
    f = open(file, 'r')
    lines = f.readlines()
    np.random.seed(seed)
    np.random.shuffle(lines)
    for line in lines:
        (fold, accountID, ow) = line.rstrip().split(',')
        folds[int(fold)].append((accountID, ow))
    f.close()
    return folds

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def bootstrap(gold, pred, reps=100000):
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
    microf1 = 2 * prec * rec / (prec + rec)
    macrof1 = f1s.sum() / len(f1s)
    
    print('accuracy = %.4f' % acc)
    print('precision = %.4f' % precision)
    print('recall = %.4f' % recall)
    print('microF1 = %.4f' % microf1)
    print('macroF1 = %.4f' % macrof1)
    print('baseline = %.4f' % baseline)
    print('p = %.6f%s' % (p, stars))
    return (acc, precision, recall, microf1, macrof1, baseline, p)



def gen_iterations(pos, neg, max_features, maxtweets, maxlen, optcv, itern):
    (x_pos, y_pos, i_pos), (x_neg, y_neg, i_neg) = pos, neg

    if optcv != 'nocv':
        itern = int(itern)
        folds = load_folds(optcv)
        X_train = list()
        y_train = list()
        X_test = list()
        y_test = list()
        X_dev = list()
        y_dev = list()
        for user in folds[itern]:
            if user[1] == "Overweight":
                position = np.where(i_pos == user[0])[0][0]
                X_test.append(x_pos[position])
                y_test.append(y_pos[position])
            else:
                position = np.where(i_neg == user[0])[0][0]
                X_test.append(x_neg[position])
                y_test.append(y_neg[position])
        nitern = itern + 1
        if nitern == 10:
            nitern = 0
        for user in folds[nitern]:
            if user[1] == "Overweight":
                position = np.where(i_pos == user[0])[0][0]
                X_dev.append(x_pos[position])
                y_dev.append(y_pos[position])
            else:
                position = np.where(i_neg == user[0])[0][0]
                X_dev.append(x_neg[position])
                y_dev.append(y_neg[position])
        for j in range(0, len(folds)):
            if itern != j and nitern != j:
                for user in folds[j]:
                    if user[1] == "Overweight":
                        position = np.where(i_pos == user[0])[0][0]
                        X_train.append(x_pos[position])
                        y_train.append(y_pos[position])
                    else:
                        position = np.where(i_neg == user[0])[0][0]
                        X_train.append(x_neg[position])
                        y_train.append(y_neg[position])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_dev = np.array(X_dev)
        y_dev = np.array(y_dev)

#        X_train = X_train[:10]
#        y_train = y_train[:10]
#        X_test = X_test[:10]
#        y_test = y_test[:10]
#        X_dev = X_dev[:10]
#        y_dev = y_dev[:10]
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print(len(X_dev), 'dev sequences')

        # In[4]:

        X_train = pad3d(X_train, maxtweets=maxtweets, maxlen=maxlen)
        X_test = pad3d(X_test, maxtweets=maxtweets, maxlen=maxlen)
        X_dev = pad3d(X_dev, maxtweets=maxtweets, maxlen=maxlen)
        train_shp = X_train.shape
        test_shp = X_test.shape
        dev_shp = X_dev.shape
        print('X_train shape:', train_shp)
        print('X_test shape:', test_shp)
        print('X_dev shape:', dev_shp)

        # In[7]:

        X_train_flat = X_train.reshape(train_shp[0] * train_shp[1], train_shp[2])
        y_train_flat = y_train.repeat(train_shp[1])
        X_train_shuff, y_train_shuff = shuffle_in_unison(X_train_flat, y_train_flat)

        X_test_flat = X_test.reshape(test_shp[0] * test_shp[1], test_shp[2])
        y_test_flat = y_test.repeat(test_shp[1])

        X_dev_flat = X_dev.reshape(dev_shp[0] * dev_shp[1], dev_shp[2])
        y_dev_flat = y_dev.repeat(dev_shp[1])
        
        # We shuffle the flattened reps. for better training
        # (but keep the original order for our by-account classification)
        X_test_shuff, y_test_shuff = shuffle_in_unison(X_test_flat, y_test_flat)
        X_dev_shuff, y_dev_shuff = shuffle_in_unison(X_dev_flat, y_dev_flat)
        
        # In[8]:

        # just clearing up space -- from now on, we use the flattened representations
        del X_train
        del X_test
        del X_dev
        
        iteration = list()
        iteration.append('fold' + str(itern+1))
        iteration.append((X_train_flat, X_train_shuff, y_train, y_train_flat, y_train_shuff, train_shp))
        iteration.append((X_test_flat, X_test_shuff, y_test, y_test_flat, y_test_shuff, test_shp))
        iteration.append((X_dev_flat, X_dev_shuff, y_dev, y_dev_flat, y_dev_shuff, dev_shp))
        return iteration

    else:
        # length of the test partition
        pos_len = int(len(y_pos)/10.0)
        neg_len = int(len(y_neg)/10.0)

        # This convoluted way of making partitions assures equal pos and neg labels per partition
        pos_test_ids = list(range(pos_len))
        neg_test_ids = list(range(neg_len))

        pos_train_ids = list(range(pos_len, len(y_pos)))
        neg_train_ids = list(range(neg_len, len(y_neg)))

        X_train = np.append(x_pos[pos_train_ids], x_neg[neg_train_ids])
        y_train = np.append(y_pos[pos_train_ids], y_neg[neg_train_ids])

        X_test = np.append(x_pos[pos_test_ids], x_neg[neg_test_ids])
        y_test = np.append(y_pos[pos_test_ids], y_neg[neg_test_ids])

        X_train, y_train = shuffle_in_unison(X_train, y_train)
        X_test, y_test = shuffle_in_unison(X_test, y_test)

        # X_train = X_train[:10]
        # y_train = y_train[:10]
        # X_test = X_test[:10]
        # y_test = y_test[:10]
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')


        # In[4]:

        X_train = pad3d(X_train, maxtweets=maxtweets, maxlen=maxlen)
        X_test = pad3d(X_test, maxtweets=maxtweets, maxlen=maxlen)
        train_shp = X_train.shape
        test_shp = X_test.shape
        print('X_train shape:', train_shp)
        print('X_test shape:', test_shp)

        # In[7]:


        X_train_flat = X_train.reshape(train_shp[0] * train_shp[1], train_shp[2])
        y_train_flat = y_train.repeat(train_shp[1])
        X_train_shuff, y_train_shuff = shuffle_in_unison(X_train_flat, y_train_flat)

        X_test_flat = X_test.reshape(test_shp[0] * test_shp[1], test_shp[2])
        y_test_flat = y_test.repeat(test_shp[1])

        # We shuffle the flattened reps. for better training
        # (but keep the original order for our by-account classification)
        X_test_shuff, y_test_shuff = shuffle_in_unison(X_test_flat, y_test_flat)


        # In[8]:

        # just clearing up space -- from now on, we use the flattened representations
        del X_train
        del X_test

        iteration = list()
        iteration.append('split')
        iteration.append((X_train_flat, X_train_shuff, y_train, y_train_flat, y_train_shuff, train_shp))
        iteration.append((X_test_flat, X_test_shuff, y_test, y_test_flat, y_test_shuff, test_shp))
        return iteration


max_features = 20000
maxtweets = 2000
maxlen = 50  # cut texts to this number of words (among top max_features most common words)
emb_dim = 200
#embeddings = load_embeddings(nb_words=max_features, emb_dim=emb_dim)
nb_filter = 64 # how many convolutional filters
filter_length = 5 # how many tokens a convolution covers
pool_length = 4 # how many cells of convolution to pool across when maxing
nb_epoch = 1 # how many training epochs
batch_size = 256 # how many tweets to train at a time

pos, neg = load_data(nb_words=max_features, maxlen=maxlen)
print ("a")
iteration = gen_iterations(pos, neg, max_features, maxtweets, maxlen, sys.argv[3], sys.argv[4])
iterid = iteration[0]
print ('')
print ('Iteration: %s' % iterid)
(X_train_flat, X_train_shuff, y_train, y_train_flat, y_train_shuff, train_shp) = iteration[1]
(X_test_flat, X_test_shuff, y_test, y_test_flat, y_test_shuff, test_shp) = iteration[2]
evalset = "test"
if sys.argv[2] == "dev":
    evalset = "dev"
    (X_test_flat, X_test_shuff, y_test, y_test_flat, y_test_shuff, test_shp) = iteration[3]

predfile = open(sys.argv[1], 'rb')
pred = pkl.load(predfile)
predfile.close()
print (pred)

# In[ ]:

threshold = sys.argv[5]
    
# account classification with each tweet's classification getting an equal vote
predmn = np.mean(pred, axis=1)
predmn = (predmn >= threshold).astype(int)

# weight by recency (most recent tweets first)
wts = np.linspace(1., 0.01, 2000)
predwm = np.average(pred, axis=1, weights=wts)
predwm = (predwm >= threshold).astype(int)

y = y_test.flatten()

print ('File:',sys.argv[1])
print ('Fold:',sys.argv[4])
print ('Set:',sys.argv[2])
print ('Treshold:', threshold)
print ('')
print('\nUnweighted mean')
(acc, precision, recall, microf1, macrof1, baseline, p) = bootstrap(y, predmn)

print('\nWeighted mean')
(acc, precision, recall, microf1, macrof1, baseline, p) = bootstrap(y, predwm)