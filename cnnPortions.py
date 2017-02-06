import gzip
import numpy as np
np.random.seed(947) # for reproducibility
import pickle as pkl

testOnDev = False

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.core import K

def pad3d(sequences, maxtweets=None, maxlen=None, dtype='int32',
          padding='pre', truncating='pre', value=0., start=0.0,
          imaxtweets=None):
    '''
        # Returns
        x: numpy array with dimensions (number_of_sequences, maxtweets, maxlen)
    '''
    nb_samples = len(sequences)
    
    if imaxtweets is not None:
        width = imaxtweets
    elif maxtweets is not None:
        width = int(float(max([len(s) for s in sequences])) * maxtweets)
    else:
        width = max([len(s) for s in sequences])

    if maxlen is not None:
        ml = maxlen
    else:
        ml = find_longest(sequences)
        
    x = (np.zeros((nb_samples, width, ml)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue # no tweets
        sstart = int(len(s) * start)
        if maxtweets is not None:
            mt = int(len(s) * maxtweets)
        elif imaxtweets is not None:
            mt = width
        else:
            mt = len(s)
        if imaxTweets is not None:
            x[idx, :min(imaxTweets,len(s)-sstart)] = sequence.pad_sequences(s[sstart:(mt+sstart)], ml, dtype, padding, truncating, value)
        else:
            x[idx, :min(mt,len(s)-sstart)] = sequence.pad_sequences(s[sstart:(mt+sstart)], ml, dtype, padding, truncating, value)
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
    '''
    Substitute Out-Of-Vocabulary index for words outside the max size
    '''
    return [[[oov if w >= nb_words else w for w in z] for z in y] for y in x]

def skip_n(x, n, oov=2):
    '''
    Substitute Out-Of-Vocabulary index for the n most common words
    '''
    return [[[oov if w < n else w for w in z] for z in y] for y in x]

def cap_length(x, maxlen):
    '''
    Ignore words after maxlen tokens
    '''
    return [[z[:maxlen] for z in y] for y in x]

def push_indices(x, start, index_from):
    '''
    Increase all indices by index_from and insert start index
    '''
    if start is not None:
        return [[[start] + [w + index_from for w in z] for z in y] for y in x]
    elif index_from:
        return [[[w + index_from for w in z] for z in y] for y in x]
    else:
        return x

def load_data(path='ow3d.pkl', nb_words=None, skip_top=0,
              maxlen=None, seed=113, start=1, oov=2, index_from=3,
              testOnDev=True):
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

    (train_X, train_y) = pkl.load(f)
    (dev_X, dev_y) = pkl.load(f)
    (test_X, test_y) = pkl.load(f)
    if (testOnDev):
        (test_X, test_y) = (dev_X, dev_y)
    else:
        train_X = train_X + dev_X
        train_y = train_y + dev_y

    f.close()

    # randomize datum order
    np.random.seed(seed)
    np.random.shuffle(train_X)
    np.random.seed(seed)
    np.random.shuffle(train_y)

    np.random.seed(seed * 2)
    np.random.shuffle(test_X)
    np.random.seed(seed * 2)
    np.random.shuffle(test_y)
    
    # keep maxlen words of each tweet
    if maxlen is not None:
        train_X = cap_length(train_X, maxlen)
        test_X = cap_length(test_X, maxlen)

    # cut off infrequent words to vocab of size nb_words
    if nb_words is not None:
        train_X = cap_words(train_X, nb_words, oov)
        test_X = cap_words(test_X, nb_words, oov)

    # cut off most frequent skip_top words
    if skip_top > 0:
        train_X = skip_n(train_X, skip_top, oov)
        test_X = skip_n(test_X, skip_top, oov)

    # prepend each sequence with start and raise indices by index_from
    train_X = push_indices(train_X, start, index_from)
    test_X = push_indices(test_X, start, index_from)
    
    train_X = np.array(train_X)
    train_y = np.array(train_y)

    test_X = np.array(test_X)
    test_y = np.array(test_y)
    
    return (train_X, train_y), (test_X, test_y)


def load_embeddings(nb_words=None, emb_dim=200, index_from=3,
                    vocab='ow3d.dict.pkl', 
                    w2v='/data/nlp/corpora/twitter4food/food_vectors_clean.txt'):
    '''
    Load pre-made embeddings from word2vec or similar. See preprocessText3D
    '''

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
        if i % 10000 == 0:
            print(".", end="")
        if i % 500000 == 0:
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

def shuffle_in_unison(a, b):
    '''
    Given two numpy arrays, shuffle them so their indices match.
    '''
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
    
    return [baseline, acc, prec, rec, microf1, macrof1, p]


max_features = 20000
maxtweets = 3200
maxlen = 50  # cut texts to this number of words (among top max_features most common words)

(X_traink, y_train), (X_testk, y_test) = load_data(nb_words=max_features, maxlen=maxlen, testOnDev=testOnDev)
print(len(X_traink), 'train sequences')
print(len(X_testk), 'test sequences')

emb_dim = 200
embeddings = load_embeddings(nb_words=max_features, emb_dim=emb_dim)

res = []

for start in range(0, 95, 5):
    floatstart = float(start) / 100.0
    maxtweets = 0.10
    print(str(floatstart) + ' to ' + str(floatstart + maxtweets))
    X_train = pad3d(X_traink, maxtweets=maxtweets, maxlen=maxlen, start=floatstart, imaxtweets=320)
    width = X_train.shape[1]
    X_test = pad3d(X_testk, maxtweets=None, maxlen=maxlen, start=floatstart, imaxtweets=width)
    train_shp = X_train.shape
    test_shp = X_test.shape
    print('X_train shape:', train_shp)
    print('X_test shape:', test_shp)

    X_train_flat = X_train.reshape(train_shp[0] * train_shp[1], train_shp[2])
    y_train_flat = y_train.repeat(train_shp[1])
    X_train_shuff, y_train_shuff = shuffle_in_unison(X_train_flat, y_train_flat)
    
    X_test_flat = X_test.reshape(test_shp[0] * test_shp[1], test_shp[2])
    y_test_flat = y_test.repeat(test_shp[1])
    
    # We shuffle the flattened reps. for better training
    # (but keep the original order for our by-account classification)
    X_test_shuff, y_test_shuff = shuffle_in_unison(X_test_flat, y_test_flat)

    nb_filter = 64 # how many convolutional filters
    filter_length = 5 # how many tokens a convolution covers
    pool_length = 4 # how many cells of convolution to pool across when maxing
    nb_epoch = 1 # how many training epochs
    batch_size = 256 # how many tweets to train at a time

    print('Build first model (tweet-level)...')
    model1 = Sequential()
    model1.add(Embedding(max_features + 3, 
                         emb_dim, 
                         input_length=maxlen,
                         weights=[embeddings]
                        ))#, 
                         #mask_zero=True))
    model1.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model1.add(MaxPooling1D(pool_length=pool_length))
    model1.add(Flatten())
    model1.add(Dense(128))
    model1.add(Activation('relu'))
    model1.add(Dropout(0.4))
    model1.add(Dense(1))
    model1.add(Activation('sigmoid'))
    model1.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

    #model1.summary()

    print('Train...')
    model1.fit(X_train_shuff, y_train_shuff, batch_size=batch_size, nb_epoch=nb_epoch,
               validation_data=(X_test_shuff, y_test_shuff))

    pred = model1.predict(X_test_flat)
    pred = pred.reshape((test_shp[0], test_shp[1]))

    # account classification with each tweet's classification getting an equal vote
    predmn = np.mean(pred, axis=1)
    predmn = (predmn >= 0.5).astype(int)
    
    # # weight by recency (most recent tweets first)
    # wts = np.linspace(1., 0.01, train_shp[1])
    # predwm = np.average(pred, axis=1, weights=wts)
    # predwm = (predwm >= 0.5).astype(int)

    y = y_test.flatten()
    
    bs = bootstrap(y, predmn)
    bs = bs[1:]
    bs.append(floatstart)
    res.append(bs)

print('acc\tprec\trec\tmicrof1\tmacrof1\tp\tstart')
for record in res:
    print("\t".join(format(x, ".4f") for x in record))
