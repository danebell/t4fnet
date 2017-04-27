
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

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.core import K


# In[2]:

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
                    w2v='/data/nlp/corpora/twitter4food/food_vectors_clean.txt'):

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

# In[4]:

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

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')


# In[5]:

X_train = pad3d(X_train, maxtweets=maxtweets, maxlen=maxlen)
X_test = pad3d(X_test, maxtweets=maxtweets, maxlen=maxlen)
train_shp = X_train.shape
test_shp = X_test.shape
print('X_train shape:', train_shp)
print('X_test shape:', test_shp)


# In[6]:

X_train_flat = X_train.reshape(train_shp[0] * train_shp[1], train_shp[2])
y_train_flat = y_train.repeat(train_shp[1])
X_train_shuff, y_train_shuff = shuffle_in_unison(X_train_flat, y_train_flat)

X_test_flat = X_test.reshape(test_shp[0] * test_shp[1], test_shp[2])
y_test_flat = y_test.repeat(test_shp[1])

# We shuffle the flattened reps. for better training
# (but keep the original order for our by-account classification)
X_test_shuff, y_test_shuff = shuffle_in_unison(X_test_flat, y_test_flat)


# In[ ]:

# just clearing up space -- from now on, we use the flattened representations
del X_train
del X_test


# In[7]:

emb_dim = 200
embeddings = load_embeddings(nb_words=max_features, emb_dim=emb_dim)


# In[8]:

nb_filter = 64 # how many convolutional filters
filter_length = 5 # how many tokens a convolution covers
pool_length = 4 # how many cells of convolution to pool across when maxing
nb_epoch = 1 # how many training epochs
batch_size = 256 # how many tweets to train at a time


# In[9]:

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


# In[10]:

model1.summary()


# In[ ]:

print('Train...')
model1.fit(X_train_shuff, y_train_shuff, batch_size=batch_size, nb_epoch=nb_epoch,
           validation_data=(X_test_shuff, y_test_shuff))


# In[ ]:

model1.save('tweet_classifier.h5')


# In[ ]:

model1 = load_model('tweet_classifier.h5')


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


# ## Intermediate data structure
# 
# Having trained a tweet-level classifier with `model1`, we now create an identical (trained) net except that we cut off final, classifying layer. This allows us to pass forward a 128-length vector for each tweet. The tweets will then be grouped by account (there being a fixed number of tweets per account). The resulting 2-D structure can be passed to a GRU or LSTM for classification.

# In[ ]:

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


# In[ ]:

intermediate.summary()


# In[ ]:

X_test_mid = K.eval(intermediate(K.variable(X_test_flat)))
X_test_mid = X_test_mid.reshape((test_shp[0], test_shp[1], 128))
X_test_mid = np.fliplr(X_test_mid)


# ## GRU classification
# 
# This is the second part of the net, which takes in the 2D account representation and returns an account classification.

# In[ ]:

batch_size = 32

model2 = Sequential()
model2.add(GRU(128, 
               dropout_W=0.2, 
               dropout_U=0.2,
               input_shape=(X_test_mid.shape[1], X_test_mid.shape[2])))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:

model2.summary()


# In[ ]:

chunk = 256
X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
for i in range(0, train_shp[0], chunk):
    last_idx = min(chunk, train_shp[0] - i)
    print('accounts ' + str(i) + ' through ' + str(i + last_idx))
    X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets : (i + last_idx) * maxtweets])))
    X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
    X_train_chunk = np.fliplr(X_train_chunk)
    X_train_mid[i:(i + last_idx)] = X_train_chunk


# In[ ]:

model2.fit(X_train_mid,  
           y_train, 
           batch_size=batch_size,
           nb_epoch=1,
           validation_data=(X_test_mid, y_test))


# In[ ]:

score, acc = model2.evaluate(X_test_mid, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# # 3 convolutions
# 
# It is possible in some instances to get more or better information from multiple convolution sizes.

# In[ ]:

from keras.models import Model
from keras.layers import Input, MaxPooling1D, Flatten, merge

# fewer filters because we have 3x the convolution stacks
nb_filters = 64
batch_size = 256

tweet = Input(shape=(50,), dtype='int32')
embed = Embedding(max_features + 3, 
                            emb_dim, 
                            weights=[embeddings]
                           )(tweet) #,
                     #mask_zero=True))

conv3 = Convolution1D(nb_filter=nb_filters,
                      filter_length=3,
                      border_mode="same",
                      activation='relu')(embed)
conv3 = MaxPooling1D(pool_length= maxlen - 3 + 1)(conv3)
conv3 = Flatten()(conv3)

conv4 = Convolution1D(nb_filter=nb_filters,
                      filter_length=4,
                      border_mode="same",
                      activation='relu')(embed)
conv4 = MaxPooling1D(pool_length= maxlen - 4 + 1)(conv4)
conv4 = Flatten()(conv4)

conv5 = Convolution1D(nb_filter=nb_filters,
                      filter_length=5,
                      border_mode="same",
                      activation='relu')(embed)
conv5 = MaxPooling1D(pool_length= maxlen - 5 + 1)(conv5)
conv5 = Flatten()(conv5)

# wider dense layer because more convolutions
dense_wide = Dense(256)(merge([conv3, conv4, conv5], mode='concat', concat_axis=1))
relu = Activation('relu')(dense_wide)
dropout = Dropout(0.4)(relu)
dense_narrow = Dense(1)(dropout)
output = Activation('sigmoid')(dense_narrow)

towers = Model(input=tweet, output=output)
towers.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])


# In[ ]:

towers.summary()


# In[ ]:

print('Train...')
towers.fit(X_train_shuff, y_train_shuff, batch_size=batch_size, nb_epoch=nb_epoch,
           validation_data=(X_test_shuff, y_test_shuff))


# In[ ]:

score, acc = towers.evaluate(X_test_shuff, y_test_shuff,
                            batch_size=batch_size)


# In[ ]:

print('Test score:', score)
print('Test accuracy:', acc)

