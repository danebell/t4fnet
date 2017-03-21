
# coding: utf-8

# # Text classifier
# 
# This LSTM classifier operates over tweets to classify Twitter account users as Overweight or Not Overweight.
# 
# Each tweet is its own entry that the LSTM operates over, so we need a 3D data structure.

# In[1]:

import gzip
import numpy as np
np.random.seed(947) # for reproducibility
import pickle as pkl
import sys

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed, Reshape, Input, merge, RepeatVector
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Convolution1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from keras.layers.core import K


# In[21]:

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
              maxlen=None, seed=113,
              start=1, oov=2, index_from=3):
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
    have simply been skipped.
    '''
    
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    (train_X, train_y) = pkl.load(f)
    (test_X, test_y) = pkl.load(f)

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

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    np.random.seed(149)
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
    print('accuracy = %.4f' % acc)
    print('baseline = %.4f' % baseline)
    print('p = %.6f%s' % (p, stars))
    return (acc, baseline, p)


# ## Load and reshape data
# 
# The data must be loaded, padded, and reshaped to train a tweet-level classifier.
# Each tweet gets the label of the account (even though most tweets are irrelevant to the classification).

# In[3]:

max_features = 20000
maxtweets = 2000
maxlen = 50  # cut texts to this number of words (among top max_features most common words)

(X_train, y_train), (X_test, y_test) = load_data(nb_words=max_features, maxlen=maxlen)

# X_train = X_train[:10]
# y_train = y_train[:10]
# X_test = X_test[:5]
# y_test = y_test[:5]
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


# In[9]:

emb_dim = 200
embeddings = load_embeddings(nb_words=max_features, emb_dim=emb_dim)


# In[10]:

nb_filter = 64 # how many convolutional filters
filter_length = 5 # how many tokens a convolution covers
pool_length = 4 # how many cells of convolution to pool across when maxing
nb_epoch = 1 # how many training epochs
batch_size = 256 # how many tweets to train at a time


#
# Pretrain cnn
#

model1 = Sequential()
model1.add(Embedding(max_features + 3,
                     emb_dim,
                     input_length=maxlen,
                     weights=[embeddings],
                    name="emb"))#,
                     #mask_zero=True))
model1.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1,
                         name="conv1d"))
model1.add(MaxPooling1D(pool_length=pool_length))
model1.add(Flatten())
model1.add(Dense(128, name="dense1"))
model1.add(Activation('relu'))
model1.add(Dropout(0.4, name="dense2"))
model1.add(Dense(1, name="dense3"))
model1.add(Activation('sigmoid'))
model1.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])


# In[14]:

model1.load_weights('models/tweet_classifier.h5')

model1.summary()

cembsf = open('cnn_embeddings.txt','w')
labelf = open('labels.txt','w')


cnnembs = model1.predict(X_train_shuff)

for i in range(0, len(cnnembs)):
    cembsf.write('\t'.join(str(n) for n in cnnembs[i]) + '\n')
    labelf.write(str(y_train_shuff[i]) + '\n')

cnnembs = model1.predict(X_test_flat)

for i in range(0, len(cnnembs)):
    cembsf.write('\t'.join(str(n) for n in cnnembs[i]) + '\n')
    labelf.write(str(y_test_flat) + '\n')


cembsf.close()
labelf.close()

