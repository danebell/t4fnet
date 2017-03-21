
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
from keras.layers import BatchNormalization, Lambda
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed, Reshape, Input, merge, RepeatVector
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from keras.layers.core import K



def GlobalSumPooling1D(x):
    return K.sum(x, axis=1)

def GlobalMaskedAveragePooling1D(x):
    not_masked = K.cast(K.any(x, axis=2), dtype="float32")
    not_masked = K.sum(not_masked, axis=1)
    not_masked = K.reshape(not_masked, (K.shape(not_masked)[0], 1))
    return K.sum(x, axis=1) / not_masked

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

    (x_pos, y_pos) = pkl.load(f)
    (x_neg, y_neg) = pkl.load(f)

    f.close()

    # randomize datum order
    np.random.seed(seed)
    np.random.shuffle(x_pos)
    np.random.seed(seed)
    np.random.shuffle(y_pos)

    np.random.seed(seed * 2)
    np.random.shuffle(x_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(y_neg)
    
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
    
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    x_neg = np.array(x_neg)
    y_neg = np.array(y_neg)
    
    return (x_pos, y_pos), (x_neg, y_neg)


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
    return (acc, baseline, p)


# ## Load and reshape data
# 
# The data must be loaded, padded, and reshaped to train a tweet-level classifier.
# Each tweet gets the label of the account (even though most tweets are irrelevant to the classification).

# In[3]:

max_features = 20000
maxtweets = 2000
maxlen = 50  # cut texts to this number of words (among top max_features most common words)

# These come out shuffled
(x_pos, y_pos), (x_neg, y_neg) = load_data(nb_words=max_features, maxlen=maxlen)

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
# X_test = X_test[:1]
# y_test = y_test[:1]
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


print('Build first model (tweet-level)...')
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

model1.summary()


# In[15]:

if (sys.argv[2] == "pre"):
    print('Train...')
    model1.fit(X_train_shuff, y_train_shuff, batch_size=batch_size, nb_epoch=nb_epoch,
               validation_data=(X_test_shuff, y_test_shuff))
    model1.save_weights('models/tweet_classifier.h5')

else:
    print('Load model...')
    model1.load_weights('models/tweet_classifier.h5')


if (sys.argv[1] == "cnn"):
    # In[ ]:
    
    score, acc = model1.evaluate(X_test_flat, y_test_flat, batch_size=batch_size)
    
    
    # In[ ]:
    
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    
    # In[ ]:
    
    pred = model1.predict(X_test_flat)
    pred = pred.reshape((test_shp[0], test_shp[1]))
    
    
    # In[ ]:
    
    # account classification with each tweet's classification getting an equal vote
    predmn = np.mean(pred, axis=1)
    predmn = (predmn >= 0.5).astype(int)
    
    # weight by recency (most recent tweets first)
    wts = np.linspace(1., 0.01, 2000)
    predwm = np.average(pred, axis=1, weights=wts)
    predwm = (predwm >= 0.5).astype(int)
    
    
    # In[ ]:
    
    y = y_test.flatten()
    
    print('Unweighted mean')
    bootstrap(y, predmn)
    
    print('\nWeighted mean')
    bootstrap(y, predwm)    
    

else:
    # ## Intermediate data structure
    # 
    # Having trained a tweet-level classifier with `model1`, we now create an identical (trained) net except that we cut off final, classifying layer. This allows us to pass forward a 128-length vector for each tweet. The tweets will then be grouped by account (there being a fixed number of tweets per account). The resulting 2-D structure can be passed to a GRU or LSTM for classification.
    
    # In[25]:
    
    intermediate = Sequential()
    intermediate.add(Embedding(max_features + 3, 
                         emb_dim, 
                         input_length=maxlen,
                         weights=[embeddings]
                        ))#, 
                         #mask_zero=True))
    intermediate.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    intermediate.add(MaxPooling1D(pool_length=pool_length))
    intermediate.add(Flatten())
    intermediate.add(Dense(128))
    intermediate.add(Activation('relu'))
    
    for l in range(len(intermediate.layers)):
        intermediate.layers[l].set_weights(model1.layers[l].get_weights())
        intermediate.layers[l]
    
    intermediate.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
    
    
    # In[26]:
    
    intermediate.summary()
    
    
    # In[28]:
    
    X_test_mid = K.eval(intermediate(K.variable(X_test_flat)))
    X_test_mid = X_test_mid.reshape((test_shp[0], test_shp[1], 128))
    X_test_mid = np.fliplr(X_test_mid)
    y_test_mid = y_test_flat.reshape((test_shp[0], test_shp[1], 1))
    y_test_mid = np.fliplr(y_test_mid)
    
    
    # This is the second part of the net, which takes in the 2D account representation and returns an account classification.
    
    if (sys.argv[1] == "rnn"):
        #
        #  Base RNN model
        #
    
        # In[29]:
    
        batch_size = 32
    
        modelRNN = Sequential()
        modelRNN.add(GRU(128,
                       dropout_W=0.2,
                       dropout_U=0.2,
                       input_shape=(X_test_mid.shape[1], X_test_mid.shape[2])))
        modelRNN.add(Dense(1))
        modelRNN.add(Activation('sigmoid'))
    
        # try using different optimizers and different optimizer configs
        modelRNN.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
    
        # In[30]:

        modelRNN.summary()
    
    
        # In[31]:
    
        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets : (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk

        if (sys.argv[3] == "train"):
            modelRNN.fit(X_train_mid,
                          y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test_mid, y_test))
            modelRNN.save_weights('models/rnn.h5')
        else:
            print('Load model...')
            modelRNN.load_weights('models/rnn.h5')


        # In[33]:
    
        score, acc = modelRNN.evaluate(X_test_mid, y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        pred = modelRNN.predict(X_test_mid)
        pred = pred.flatten()
        pred = (pred >= 0.5).astype(int)
        y = y_test.flatten()
        bootstrap(y, pred)

    if (sys.argv[1] == "birnn"):
        #
        #  Base RNN model
        #

        # In[29]:

        batch_size = 32

        modelRNN = Sequential()
        modelRNN.add(Bidirectional(GRU(128,
                         dropout_W=0.2,
                         dropout_U=0.2),
                         input_shape=(X_test_mid.shape[1], X_test_mid.shape[2]),
                        merge_mode='ave'))
        modelRNN.add(Dense(1))
        modelRNN.add(Activation('sigmoid'))

        # try using different optimizers and different optimizer configs
        modelRNN.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

        # In[30]:

        modelRNN.summary()

        # In[31]:

        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets: (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk

        if (sys.argv[3] == "train"):
            modelRNN.fit(X_train_mid,
                         y_train,
                         batch_size=batch_size,
                         nb_epoch=nb_epoch,
                         validation_data=(X_test_mid, y_test))
            modelRNN.save_weights('models/birnn.h5')
        else:
            print('Load model...')
            modelRNN.load_weights('models/birnn.h5')

        # In[33]:

        score, acc = modelRNN.evaluate(X_test_mid, y_test,
                                       batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        pred = modelRNN.predict(X_test_mid)
        pred = pred.flatten()
        pred = (pred >= 0.5).astype(int)
        y = y_test.flatten()
        bootstrap(y, pred)

    elif (sys.argv[1] == "stacked"):
        #
        #  Stacked RNN model
        # The first RNN layer returns sequences to feed the stacked RNN layer
        #
    
        # In[29]:
    
        batch_size = 32
    
        modelStack = Sequential()
        modelStack.add(GRU(128,
                       dropout_W=0.2,
                       dropout_U=0.2,
                       input_shape=(X_test_mid.shape[1], X_test_mid.shape[2]),
                       return_sequences=True))
        modelStack.add(GRU(128,
                       dropout_W=0.2,
                       dropout_U=0.2))
        modelStack.add(Dense(1))
        modelStack.add(Activation('sigmoid'))
    
        # try using different optimizers and different optimizer configs
        modelStack.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
    
        # In[30]:
    
        modelStack.summary()
    
    
        # In[31]:
    
        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets : (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk
    
    
        # In[32]:
        if (sys.argv[3] == "train"):
            modelStack.fit(X_train_mid,
                          y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test_mid, y_test))
            modelStack.save_weights('models/stacked-rnn.h5')
        else:
            print('Load model...')
            modelStack.load_weights('models/stacked-rnn.h5')


        # In[33]:
    
        score, acc = modelStack.evaluate(X_test_mid, y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
    
        pred = modelStack.predict(X_test_mid)
        pred = pred.flatten()
        pred = (pred >= 0.5).astype(int)
        y = y_test.flatten()
        bootstrap(y, pred)


    elif (sys.argv[1] == "mlp"):
        #
        #  RNN+MLP model
        #
    
        # In[29]:
    
        batch_size = 32
    
        modelMLP = Sequential()
        modelMLP.add(GRU(128,
                       dropout_W=0.2,
                       dropout_U=0.2,
                       input_shape=(X_test_mid.shape[1], X_test_mid.shape[2]),
                       return_sequences=True))
        modelMLP.add(TimeDistributed(Dense(64,activation='relu')))
        modelMLP.add(Flatten())
        modelMLP.add(Dense(200,activation='relu'))
        modelMLP.add(Dense(1))
        modelMLP.add(Activation('sigmoid'))
    
        # try using different optimizers and different optimizer configs
        modelMLP.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
    
        # In[30]:
    
        modelMLP.summary()
    
    
        # In[31]:
    
        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets : (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk
    
    
        # In[32]:
        if (sys.argv[3] == "train"):
            modelMLP.fit(X_train_mid,
                          y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test_mid, y_test))
            modelMLP.save_weights('models/mlp.h5')
        else:
            print('Load model...')
            modelMLP.load_weights('models/mlp.h5')
    
    
        # In[33]:
    
        score, acc = modelMLP.evaluate(X_test_mid, y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        pred = modelMLP.predict(X_test_mid)
        pred = pred.flatten()
        pred = (pred >= 0.5).astype(int)
        y = y_test.flatten()
        bootstrap(y, pred)
    
    
    elif (sys.argv[1] == "weighting"):
        #
        #  RNN recency weighting model
        # In[29]:
    
        batch_size = 32
    
        modelRWeight = Sequential()
        modelRWeight.add(GRU(128,
                       dropout_W=0.2,
                       dropout_U=0.2,
                       input_shape=(X_test_mid.shape[1], X_test_mid.shape[2]),
                       return_sequences=True))
        modelRWeight.add(TimeDistributed(Dense(1, activation='sigmoid')))
    
        # try using different optimizers and different optimizer configs
        modelRWeight.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
    
        # In[30]:
    
        modelRWeight.summary()
    
    
        # In[31]:
    
        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        y_train_mid = np.zeros((train_shp[0], train_shp[1], 1))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets : (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk
            y_train_chunk = y_train_flat[i * maxtweets : (i + last_idx) * maxtweets]
            y_train_chunk = y_train_chunk.reshape((last_idx, maxtweets, 1))
            y_train_chunk = np.fliplr(y_train_chunk)
            y_train_mid[i:(i + last_idx)] = y_train_chunk
    
    
        # In[32]:
        if (sys.argv[3] == "train"):
            modelRWeight.fit(X_train_mid,
                          y_train_mid,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test_mid, y_test_mid))
            modelRWeight.save_weights('models/rnn-rweights.h5')
        else:
            print('Load model...')
            modelRWeight.load_weights('models/rnn-rweights.h5')
    
    
        # In[33]:
    
        score, acc = modelRWeight.evaluate(X_test_mid, y_test_mid,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
    
    
        # In[20]:
    
        pred = modelRWeight.predict(X_test_mid)
        pred = pred.reshape((test_shp[0], test_shp[1]))
    
        # In[23]:
    
        # account classification with each tweet's classification getting an equal vote
        predmn = np.mean(pred, axis=1)
        predmn = (predmn >= 0.5).astype(int)
    
        # weight by recency (most recent tweets first)
        wts = np.linspace(0.01, 1., 2000)
        predwm = np.average(pred, axis=1, weights=wts)
        predwm = (predwm >= 0.5).astype(int)
    
    
        # In[24]:
    
        y = y_test.flatten()
    
        print('Unweighted mean')
        bootstrap(y, predmn)
    
        print('\nWeighted mean')
        bootstrap(y, predwm)
    
    
    elif (sys.argv[1] == "pool"):
        #
        #  CNN with pooling
        # In[29]:
    
        batch_size = 32
    
        cnnInput = Input(shape=(train_shp[1], 128), dtype='float32', name='cnn_input')
        averagePooling = GlobalAveragePooling1D()(cnnInput)
        dropout = Dropout(0.4)(averagePooling)
        top = Dense(1, activation='sigmoid')(dropout)
        modelPool = Model(input=[cnnInput], output=[top])
        modelPool.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
    
        # In[30]:
    
        modelPool.summary()
    
        # In[31]:
    
        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets: (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk
    
        # In[32]:
        if (sys.argv[3] == "train"):
            modelPool.fit(X_train_mid,
                          y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test_mid, y_test))
            modelPool.save_weights('models/cnn-pooling.h5')
        else:
            print('Load model...')
            modelPool.load_weights('models/cnn-pooling.h5')
    
        
        # In[33]:
    
        score, acc = modelPool.evaluate(X_test_mid, y_test,
                                        batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        pred = modelPool.predict(X_test_mid)
        pred = pred.flatten()
        pred = (pred >= 0.5).astype(int)
        y = y_test.flatten()
        bootstrap(y, pred)
    
    
    elif (sys.argv[1] == "relu"):
        #
        #  CNN recency weighting with relu model
        # In[29]:
    
        batch_size = 32

        recentInput = Input(shape=(train_shp[1], 1), dtype='float32', name='recent_input')
        recentNorm = TimeDistributed(Dense(1,activation='tanh'),name='tanh_norm')(recentInput)
        recentSelect = TimeDistributed(Dense(1, activation='relu'), name='relu_select')(recentNorm)
        recentRepeat = TimeDistributed(RepeatVector(128), name='repeat_vector')(recentSelect)
        recentReshape = Reshape((train_shp[1], 128), name='reshape')(recentRepeat)
        cnnInput = Input(shape=(train_shp[1], 128), dtype='float32', name='cnn_input')
        mergedInputs = merge([cnnInput, recentReshape], mode='mul')
        sumPooling = Lambda(GlobalSumPooling1D, output_shape=(128,))(mergedInputs)
        dropout = Dropout(0.4, name='dropout')(sumPooling)
        top = Dense(1, activation='sigmoid', name='top_sigmoid')(dropout)
        modelRelu = Model(input=[recentInput, cnnInput], output=[top])
        modelRelu.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
        # In[30]:
    
        modelRelu.summary()
    
        wts = np.linspace(0.1, 1, train_shp[1])
        wtsTrain = np.tile(wts,(train_shp[0],1))
        wtsTrain = np.reshape(wtsTrain, (train_shp[0], train_shp[1], 1))
    
        wtsTest = np.tile(wts, (test_shp[0], 1))
        wtsTest = np.reshape(wtsTest, (test_shp[0], train_shp[1], 1))

        # In[31]:
    
        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets : (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk

                        
        # In[32]:
        if (sys.argv[3] == "train"):
            modelRelu.fit([wtsTrain, X_train_mid],
                          y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=([wtsTest, X_test_mid], y_test))
            modelRelu.save_weights('models/cnn-relu.h5')
        else:
            print('Load model...')
            modelRelu.load_weights('models/cnn-relu.h5')



        # In[33]:

        score, acc = modelRelu.evaluate([wtsTest, X_test_mid], y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        pred = modelRelu.predict([wtsTest, X_test_mid])
        pred = pred.flatten()
        pred = (pred >= 0.5).astype(int)
        y = y_test.flatten()
        bootstrap(y, pred)


    elif (sys.argv[1] == "rnnrelu"):
        #
        #  CNN recency weighting with relu model
        # In[29]:

        batch_size = 32

        recentInput = Input(shape=(train_shp[1], 1), dtype='float32', name='recent_input')
        recentNorm = TimeDistributed(Dense(1, activation='tanh'), name='tanh_norm')(recentInput)
        recentSelect = TimeDistributed(Dense(1, activation='relu'), name='relu_select')(recentNorm)
        recentRepeat = TimeDistributed(RepeatVector(128), name='repeat_vector')(recentSelect)
        recentReshape = Reshape((train_shp[1], 128), name='reshape')(recentRepeat)
        cnnInput = Input(shape=(train_shp[1], 128), dtype='float32', name='cnn_input')
        mergedInputs = merge([cnnInput, recentReshape], mode='mul')
        gru = GRU(128, dropout_W=0.2, dropout_U=0.2)(mergedInputs)
        top = Dense(1, activation='sigmoid', name='top_sigmoid')(gru)
        modelRelu = Model(input=[recentInput, cnnInput], output=[top])
        modelRelu.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])


        # In[30]:

        modelRelu.summary()

        wts = np.linspace(0.1, 1, train_shp[1])
        wtsTrain = np.tile(wts, (train_shp[0], 1))
        wtsTrain = np.reshape(wtsTrain, (train_shp[0], train_shp[1], 1))

        wtsTest = np.tile(wts, (test_shp[0], 1))
        wtsTest = np.reshape(wtsTest, (test_shp[0], train_shp[1], 1))

        # In[31]:

        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets: (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk

        # In[32]:
        if (sys.argv[3] == "train"):
            modelRelu.fit([wtsTrain, X_train_mid],
                          y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=([wtsTest, X_test_mid], y_test))
            modelRelu.save_weights('models/cnn-relu.h5')
        else:
            print('Load model...')
            modelRelu.load_weights('models/cnn-relu.h5')

        # In[33]:

        score, acc = modelRelu.evaluate([wtsTest, X_test_mid], y_test,
                                        batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        pred = modelRelu.predict([wtsTest, X_test_mid])
        pred = pred.flatten()
        pred = (pred >= 0.5).astype(int)
        y = y_test.flatten()
        bootstrap(y, pred)


    elif (sys.argv[1] == "attention"):
        #
        #  CNN recency weighting with attention mechanism
        # In[29]:
            
        batch_size = 32
    
        cnnInput = Input(shape=(train_shp[1], 128), dtype='float32', name='cnn_input')
        cnnTanh = TimeDistributed(Dense(128, activation='tanh'))(cnnInput)
        cnnLinear = TimeDistributed(Dense(1, activation='linear', bias=False))(cnnTanh)
        cnnFlat = Flatten()(cnnLinear)
        attentionSoftmax = Activation('softmax')(cnnFlat)
        attentionToSeq = Reshape((train_shp[1], 1))(attentionSoftmax)
        attentionRepeat = TimeDistributed(RepeatVector(128))(attentionToSeq)
        attentionReshape = Reshape((train_shp[1], 128))(attentionRepeat)
        mergedInputs = merge([cnnInput, attentionReshape], mode='mul')
        #averagePooling = GlobalAveragePooling1D()(mergedInputs)
        averagePooling = Lambda(GlobalSumPooling1D, output_shape=(128,))(mergedInputs)
        dropout = Dropout(0.4)(averagePooling)
        top = Dense(1, activation='sigmoid')(dropout)
        modelAttention = Model(input=[cnnInput], output=[top])
        modelAttention.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # In[30]:

        modelAttention.summary()
    
        # In[31]:
    
        chunk = 256
        X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
        for i in range(0, train_shp[0], chunk):
            last_idx = min(chunk, train_shp[0] - i)
            print('accounts ' + str(i) + ' through ' + str(i + last_idx))
            X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets : (i + last_idx) * maxtweets])))
            X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
            X_train_chunk = np.fliplr(X_train_chunk)
            X_train_mid[i:(i + last_idx)] = X_train_chunk
    
        # In[32]:
        if (sys.argv[3] == "train"):
            modelAttention.fit(X_train_mid,
                          y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test_mid, y_test))
            modelAttention.save_weights('models/cnn-attention.h5')
        else:
            print('Load model...')
            modelAttention.load_weights('models/cnn-attention.h5')
    
    
        # In[33]:
    
        score, acc = modelAttention.evaluate(X_test_mid, y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        pred = modelAttention.predict(X_test_mid)
        pred = pred.flatten()
        pred = (pred >= 0.5).astype(int)
        y = y_test.flatten()
        bootstrap(y, pred)

