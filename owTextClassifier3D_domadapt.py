
# coding: utf-8

#
#
# NN models to classify Twitter account users as Overweight or Not Overweight.
# 
#
CUDA_MODE = True

import gzip
import numpy as np
np.random.seed(947) # for reproducibility
import pickle as pkl
import sys
import math

from keras.preprocessing import sequence
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed
#from keras.layers import GRU
#from keras.layers import Convolution1D, MaxPooling1D, Flatten
#from keras.layers.core import K

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim


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

def load_data(path='ow3df.pkl', nb_words=None, skip_top=0,
#def load_data(path='data_toy/ow3df.pkl', nb_words=None, skip_top=0,
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

    (x_pos, i_pos, f_pos, y_pos) = pkl.load(f)
    (x_neg, i_neg, f_neg, y_neg) = pkl.load(f)
    
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

    np.random.seed(seed * 2)
    np.random.shuffle(x_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(y_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(i_neg)
    np.random.seed(seed * 2)
    np.random.shuffle(f_neg)

    
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
    i_pos = np.array(i_pos)
    f_pos = np.array(f_pos)

    x_neg = np.array(x_neg)
    y_neg = np.array(y_neg)
    i_neg = np.array(i_neg)
    f_neg = np.array(f_neg)
    
    return (x_pos, y_pos, i_pos, f_pos), (x_neg, y_neg, i_neg, f_neg)


def load_embeddings(nb_words=None, emb_dim=200, index_from=3,
                    vocab='ow3df.dict.pkl', 
                    #vocab='data_toy/ow3df.dict.pkl', 
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

def shuffle_in_unison(a, b, c):
    assert len(a) == len(b) == len(c)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
    return shuffled_a, shuffled_b, shuffled_c

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



def gen_iterations(pos, neg, max_features, maxtweets, maxlen, foldsfile):
    (x_pos, y_pos, i_pos, f_pos), (x_neg, y_neg, i_neg, f_neg) = pos, neg

    folds = load_folds(foldsfile)
    for itern in range(0, len(folds)):
        X_train = list()
        y_train = list()
        f_train = list()
        X_test = list()
        y_test = list()
        f_test = list()
        X_dev = list()
        y_dev = list()
        f_dev = list()
        for user in folds[itern]:
            if user[1] == "Overweight":
                position = np.where(i_pos == user[0])[0][0]
                X_test.append(x_pos[position])
                y_test.append(y_pos[position])
                f_test.append(f_pos[position])
            else:
                position = np.where(i_neg == user[0])[0][0]
                X_test.append(x_neg[position])
                y_test.append(y_neg[position])
                f_test.append(f_neg[position])
        nitern = itern + 1
        if nitern == len(folds):
            nitern = 0
        for user in folds[nitern]:
            if user[1] == "Overweight":
                position = np.where(i_pos == user[0])[0][0]
                X_dev.append(x_pos[position])
                y_dev.append(y_pos[position])
                f_dev.append(f_pos[position])
            else:
                position = np.where(i_neg == user[0])[0][0]
                X_dev.append(x_neg[position])
                y_dev.append(y_neg[position])
                f_dev.append(f_neg[position])
        for j in range(0, len(folds)):
            if itern != j and nitern != j:
                for user in folds[j]:
                    if user[1] == "Overweight":
                        position = np.where(i_pos == user[0])[0][0]
                        X_train.append(x_pos[position])
                        y_train.append(y_pos[position])
                        f_train.append(f_pos[position])
                    else:
                        position = np.where(i_neg == user[0])[0][0]
                        X_train.append(x_neg[position])
                        y_train.append(y_neg[position])
                        f_train.append(f_neg[position])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        f_train = np.array(f_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        f_test = np.array(f_test)
        X_dev = np.array(X_dev)
        y_dev = np.array(y_dev)
        f_dev = np.array(f_dev)

#        X_train = X_train[:10]
#        y_train = y_train[:10]
#        X_test = X_test[:10]
#        y_test = y_test[:10]
#        X_dev = X_dev[:10]
#        y_dev = y_dev[:10]
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print(len(X_dev), 'dev sequences')

        X_train = pad3d(X_train, maxtweets=maxtweets, maxlen=maxlen)
        X_test = pad3d(X_test, maxtweets=maxtweets, maxlen=maxlen)
        X_dev = pad3d(X_dev, maxtweets=maxtweets, maxlen=maxlen)
        train_shp = X_train.shape
        test_shp = X_test.shape
        dev_shp = X_dev.shape
        print('X_train shape:', train_shp)
        print('X_test shape:', test_shp)
        print('X_dev shape:', dev_shp)

        X_train_flat = X_train.reshape(train_shp[0] * train_shp[1], train_shp[2])
        y_train_flat = y_train.repeat(train_shp[1])
        f_train_flat = f_train.repeat(train_shp[1], axis=0)
        X_train_shuff, y_train_shuff, f_train_shuff = shuffle_in_unison(X_train_flat, y_train_flat, f_train_flat)

        X_test_flat = X_test.reshape(test_shp[0] * test_shp[1], test_shp[2])
        y_test_flat = y_test.repeat(test_shp[1])
        f_test_flat = f_test.repeat(test_shp[1], axis=0)

        X_dev_flat = X_dev.reshape(dev_shp[0] * dev_shp[1], dev_shp[2])
        y_dev_flat = y_dev.repeat(dev_shp[1])
        f_dev_flat = f_dev.repeat(dev_shp[1], axis=0)
        
        # We shuffle the flattened reps. for better training
        # (but keep the original order for our by-account classification)
        X_test_shuff, y_test_shuff, f_test_shuff = shuffle_in_unison(X_test_flat, y_test_flat, f_test_flat)
        X_dev_shuff, y_dev_shuff, f_dev_shuff = shuffle_in_unison(X_dev_flat, y_dev_flat, f_dev_flat)

        # just clearing up space -- from now on, we use the flattened representations
        del X_train
        del X_test
        del X_dev
        
        iteration = list()
        iteration.append('fold' + str(itern))
        iteration.append((X_train_flat, X_train_shuff, y_train, y_train_flat, y_train_shuff, 
                          f_train, f_train_flat, f_train_shuff, train_shp))
        iteration.append((X_test_flat, X_test_shuff, y_test, y_test_flat, y_test_shuff,
                          f_test, f_test_flat, f_test_shuff, test_shp))
        iteration.append((X_dev_flat, X_dev_shuff, y_dev, y_dev_flat, y_dev_shuff,
                          f_dev, f_dev_flat, f_dev_shuff, dev_shp))
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
        earlystop = 0
        for threshold in np.arange(start, stop, step):
            pred_th = (pred >= threshold).astype(int)
            (acc, precision, recall, f1, microf1, macrof1, baseline, p) = bootstrap(gold, pred_th, printit=False)
            if f1 > maxf1:
                maxf1 = f1
                maxth = threshold
                print("threshold:", maxth, ", F1:", maxf1)
                vary = True
                earlystop = 0
            else:
                earlystop += 1
                if earlystop == 2:
                    start = maxth - step
                    if start < 0.:
                        start = 0.
                    stop = threshold - step
                    if stop > 1.0:
                        stop = 1.0
                    step = step * 0.1
                    break
    return maxth

def new_outs_lengths(input_lenght, kernel_size, padding=0, dilation=1, stride=1):
    return np.floor((input_lenght + 2*padding - dilation*(kernel_size-1) -1) / stride + 1)
        
    
class Pre(nn.Module):
    def __init__(self, max_features, embedding_dim, seq_length, nb_filter, filter_length, pool_length, hidden_size):
        super(Pre, self).__init__()
        cnn_out_length = new_outs_lengths(seq_length, filter_length)
        pool_out_length = new_outs_lengths(cnn_out_length, pool_length, stride=pool_length)
        self.embs = nn.Embedding(max_features + 3, embedding_dim)
        # Common 
        self.c_cnn = nn.Conv1d(embedding_dim, nb_filter, filter_length)
        self.c_pool = nn.MaxPool1d(pool_length)        
        self.c_linear1 = nn.Linear(int(pool_out_length) * nb_filter, hidden_size)
        self.c_relu1 = nn.ReLU()
        # Female
        self.f_cnn = nn.Conv1d(embedding_dim, nb_filter, filter_length)
        self.f_pool = nn.MaxPool1d(pool_length)        
        self.f_linear1 = nn.Linear(int(pool_out_length) * nb_filter, hidden_size)
        self.f_relu1 = nn.ReLU()
        # Male
        self.m_cnn = nn.Conv1d(embedding_dim, nb_filter, filter_length)
        self.m_pool = nn.MaxPool1d(pool_length)        
        self.m_linear1 = nn.Linear(int(pool_out_length) * nb_filter, hidden_size)
        self.m_relu1 = nn.ReLU()
        # Top
        self.dropout1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2_domadapt = nn.Linear(hidden_size*2, 1)
        self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, inputs, intermediate=False, domain=-1):
        embeds = self.embs(inputs)
        embeds = embeds.transpose(0, 1).transpose(1, 2)
        out = self.c_cnn(embeds)
        out = self.c_pool(out)
        out = out.view((out.size()[0],out.size()[1] * out.size()[2]))
        out = self.c_relu1(self.c_linear1(out))
        if domain == 0:
            outf = self.f_cnn(embeds)
            outf = self.f_pool(outf)
            outf = outf.view((outf.size()[0],outf.size()[1] * outf.size()[2]))
            outf = self.f_relu1(self.f_linear1(outf))
            out = torch.cat((out,outf),1)
        if domain == 1:
            outm = self.m_cnn(embeds)
            outm = self.m_pool(outm)
            outm = outm.view((outm.size()[0],outm.size()[1] * outm.size()[2]))
            outm = self.m_relu1(self.m_linear1(outm))
            out = torch.cat((out,outm),1)
        if domain == -1:
            if not intermediate:
                out = self.dropout1(out)
                out = self.sigmoid2(self.linear2(out))
        else:
            if not intermediate:
                out = self.dropout1(out)
                out = self.sigmoid2(self.linear2_domadapt(out))
        return out

        
def predict(net, x, f, intermediate=False):
    fb = torch.LongTensor(torch.np.where(f[:,0]==0)[0])
    if CUDA_MODE:
        fb = fb.cuda()
    xf = x[fb]
    xf = torch.transpose(xf, 0, 1)
    mb = torch.LongTensor(torch.np.where(f[:,0]==1)[0])
    if CUDA_MODE:
        mb = mb.cuda()
    xm = x[mb]
    xm = torch.transpose(xm, 0, 1)
    f_pred = net(xf, domain=0)
    m_pred = net(xm, domain=1)
    fb = fb + 5
    b = torch.cat((fb, mb))
    if CUDA_MODE:
        b = torch.LongTensor(torch.np.argsort(b.cpu().numpy())).cuda()
    else:
        b = torch.LongTensor(torch.np.argsort(b.numpy()))    
    pred = torch.cat((f_pred, m_pred))
    pred = pred[b]
    return pred
    
def train(net, x, y, f, nepochs, batch_size):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    batches = math.ceil(x.size()[0] / batch_size)
    for e in range(nepochs):
        running_loss = 0.0
        for b in range(batches):
            # Get minibatch samples
            bx = x[b*batch_size:b*batch_size+batch_size]
            by = y[b*batch_size:b*batch_size+batch_size]
            bf = f[b*batch_size:b*batch_size+batch_size]
            
            fb = torch.LongTensor(torch.np.where(bf[:,0]==0)[0])
            if CUDA_MODE:
                fb = fb.cuda()
            bxf = bx[fb]
            byf = by[fb]
            bxf = torch.transpose(bxf, 0, 1)

            mb = torch.LongTensor(torch.np.where(bf[:,0]==1)[0])
            if CUDA_MODE:
                mb = mb.cuda()
            bxm = bx[mb]
            bym = by[mb]
            bxm = torch.transpose(bxm, 0, 1)
            
            # Clear gradients
            net.zero_grad()
            
            # Forward pass
            yf_pred = net(bxf, domain=0)
            ym_pred = net(bxm, domain=1)
            
            # Compute loss
            y_pred = torch.cat((yf_pred, ym_pred))
            by = torch.cat((byf, bym))
            loss = criterion(y_pred, by)

            # Print loss
            running_loss += loss.data[0]
            sys.stdout.write('\r[epoch: %3d, batch: %3d/%3d] loss: %.3f' % (e + 1, b + 1, batches, running_loss / (b+1)))
            sys.stdout.flush()

            # Backward propagation and update the weights.
            loss.backward()
            optimizer.step()
        sys.stdout.write('\n')
        sys.stdout.flush()


max_features = 20000
maxtweets = 2000
maxlen = 50  # cut texts to this number of words (among top max_features most common words)
emb_dim = 200
embeddings = load_embeddings(nb_words=max_features, emb_dim=emb_dim)
nb_filter = 64 # how many convolutional filters
filter_length = 5 # how many tokens a convolution covers
pool_length = 4 # how many cells of convolution to pool across when maxing
nb_epoch = 1 # how many training epochs
batch_size = 256 # how many tweets to train at a time

pos, neg = load_data(nb_words=max_features, maxlen=maxlen)
predictions = dict()
predictions["cnnv"] = list()
predictions["cnnw"] = list()
predictions["gruv"] = list()
predictions["gruw"] = list()
gold_test = list()
foldsfile = "folds.csv"
#foldsfile = "data_toy/folds.csv"
for iteration in gen_iterations(pos, neg, max_features, maxtweets, maxlen, foldsfile):
    iterid = iteration[0]
    print('')
    print('Iteration: %s' % iterid)
    (X_train_flat, X_train_shuff, y_train, y_train_flat, y_train_shuff,
     f_train, f_train_flat, f_train_shuff, train_shp) = iteration[1]
    (X_test_flat, X_test_shuff, y_test, y_test_flat, y_test_shuff,
     f_test, f_test_flat, f_test_shuff, test_shp) = iteration[2]
    (X_dev_flat, X_dev_shuff, y_dev, y_dev_flat, y_dev_shuff,
     f_dev, f_dev_flat, f_dev_shuff, dev_shp) = iteration[3]
    
    gold_dev = y_dev.flatten()
    gold_test.extend(y_test.flatten())

    #
    # Pre-train tweet-level vectors
    #

    print('Build first model (tweet-level)...')
    net = Pre(max_features, emb_dim, maxlen, nb_filter, filter_length, pool_length, 128)
    net.embs.weight.data.copy_(torch.from_numpy(np.array(embeddings)))
    if CUDA_MODE:
        net = net.cuda()
        data_x = Variable(torch.from_numpy(X_train_shuff).long().cuda())
        data_y = Variable(torch.from_numpy(y_train_shuff).float().cuda())
    else:
        data_x = Variable(torch.from_numpy(X_train_shuff).long())        
        data_y = Variable(torch.from_numpy(y_train_shuff).float())
    data_f = f_train_shuff
    
    print('Train...')
    train(net, data_x, data_y, data_f, nb_epoch, batch_size)
    torch.save(net.state_dict(), 'domadapt/models/tweet_classifier_' + iterid + '.pkl')

    
#    modelPre.fit(X_train_shuff, y_train_shuff, batch_size=batch_size, nb_epoch=nb_epoch,
#               validation_data=(X_test_shuff, y_test_shuff))
#    modelPre.save_weights('models/tweet_classifier_' + iterid + '.h5')

    #
    #  CNN+V/CNN+W
    #

    # Prediction for DEV set
#    score, acc = modelPre.evaluate(X_dev_flat, y_dev_flat, batch_size=batch_size)
#    print('Dev score:', score)
#    print('Dev accuracy:', acc)
#    predDev = modelPre.predict(X_dev_flat)
#    predDev = predDev.reshape((dev_shp[0], dev_shp[1]))

    if CUDA_MODE:
        data_x = Variable(torch.from_numpy(X_dev_flat).long().cuda())
    else:
        data_x = Variable(torch.from_numpy(X_dev_flat).long())
    data_f = f_dev_flat
    
    predDev = predict(net, data_x, data_f)
    if CUDA_MODE:
        predDev = predDev.cpu().data.numpy().reshape((dev_shp[0], dev_shp[1]))
    else:    
        predDev = predDev.data.numpy().reshape((dev_shp[0], dev_shp[1]))
        
    predDevmn = np.mean(predDev, axis=1)
    print('Search CNN+V threshold')
    thldmn = get_threshold(gold_dev, predDevmn)

    wts = np.linspace(1., 0.01, 2000)
    predDevwm = np.average(predDev, axis=1, weights=wts)
    print('Search CNN+W threshold')
    thldwm = get_threshold(gold_dev, predDevwm)

    # Prediction for TEST set
#    score, acc = modelPre.evaluate(X_test_flat, y_test_flat, batch_size=batch_size)
#    print('Test score:', score)
#    print('Test accuracy:', acc)
#    predTest = modelPre.predict(X_test_flat)
#    predTest = predTest.reshape((test_shp[0], test_shp[1]))

    if CUDA_MODE:
        data_x = Variable(torch.from_numpy(X_test_flat).long().cuda())
    else:
        data_x = Variable(torch.from_numpy(X_test_flat).long())
    data_f = f_test_flat
    predTest = predict(net, data_x, data_f)
    if CUDA_MODE:
        predTest = predTest.cpu().data.numpy().reshape((test_shp[0], test_shp[1]))
    else:
        predTest = predTest.data.numpy().reshape((test_shp[0], test_shp[1]))
    predTestmn = np.mean(predTest, axis=1)
    predTestmn = (predTestmn >= thldmn).astype(int)
    predictions["cnnv"].extend(predTestmn)

    wts = np.linspace(1., 0.01, 2000)
    predTestwm = np.average(predTest, axis=1, weights=wts)
    predTestwm = (predTestwm >= thldwm).astype(int)
    predictions["cnnw"].extend(predTestwm)


#    #
#    # Intermediate data structure to get the input for GRU+V/GRU+W
#    #
#    #
#
#    data_x = Variable(torch.from_numpy(X_test_flat).long())
#    X_test_mid = predict(net, data_x, intermediate=True)
#    X_test_mid = X_test_mid.data.numpy().reshape((test_shp[0], test_shp[1], 128))
#    X_test_mid = np.fliplr(X_test_mid)
#    y_test_mid = y_test_flat.reshape((test_shp[0], test_shp[1], 1))
#    y_test_mid = np.fliplr(y_test_mid)
#
#    data_x = Variable(torch.from_numpy(X_dev_flat).long())
#    X_dev_mid = predict(net, data_x, intermediate=True)
#    X_dev_mid = X_dev_mid.data.numpy().reshape((dev_shp[0], dev_shp[1], 128))
#    X_dev_mid = np.fliplr(X_dev_mid)
#    y_dev_mid = y_dev_flat.reshape((dev_shp[0], dev_shp[1], 1))
#    y_dev_mid = np.fliplr(y_dev_mid)
#
#
#    #
#    #  GRU+V/GRU+W
#    #
#
#    batch_size = 32
#
#    modelGRU = Sequential()
#    modelGRU.add(GRU(128,
#                   dropout_W=0.2,
#                   dropout_U=0.2,
#                   input_shape=(X_test_mid.shape[1], X_test_mid.shape[2]),
#                   return_sequences=True))
#    modelGRU.add(TimeDistributed(Dense(1, activation='sigmoid')))
#
#    # Compile
#    modelGRU.compile(loss='binary_crossentropy',
#                  optimizer='adam',
#                  metrics=['accuracy'])
#    modelGRU.summary()
#
#    chunk = 256
#    X_train_mid = np.zeros((train_shp[0], train_shp[1], 128))
#    y_train_mid = np.zeros((train_shp[0], train_shp[1], 1))
#    for i in range(0, train_shp[0], chunk):
#        last_idx = min(chunk, train_shp[0] - i)
#        print('accounts ' + str(i) + ' through ' + str(i + last_idx))
#        X_train_chunk = K.eval(intermediate(K.variable(X_train_flat[i * maxtweets : (i + last_idx) * maxtweets])))
#        X_train_chunk = X_train_chunk.reshape((last_idx, maxtweets, 128))
#        X_train_chunk = np.fliplr(X_train_chunk)
#        X_train_mid[i:(i + last_idx)] = X_train_chunk
#        y_train_chunk = y_train_flat[i * maxtweets : (i + last_idx) * maxtweets]
#        y_train_chunk = y_train_chunk.reshape((last_idx, maxtweets, 1))
#        y_train_chunk = np.fliplr(y_train_chunk)
#        y_train_mid[i:(i + last_idx)] = y_train_chunk
#
#
#    # Train the model
#    modelGRU.fit(X_train_mid,
#                  y_train_mid,
#                  batch_size=batch_size,
#                  nb_epoch=nb_epoch,
#                  validation_data=(X_dev_mid, y_dev_mid))
#    modelGRU.save_weights('models/gru_' + iterid + '.h5')
#
#    # Prediction for DEV set
#    score, acc = modelGRU.evaluate(X_dev_mid, y_dev_mid, batch_size=batch_size)
#    print('Dev score:', score)
#    print('Dev accuracy:', acc)
#    predDev = modelGRU.predict(X_dev_mid)
#    predDev = predDev.reshape((dev_shp[0], dev_shp[1]))
#
#    predDevmn = np.mean(predDev, axis=1)
#    print('Search GRU+V threshold')
#    thldmn = get_threshold(gold_dev, predDevmn)
#
#    wts = np.linspace(1., 0.01, 2000)
#    predDevwm = np.average(predDev, axis=1, weights=wts)
#    print('Search GRU+W threshold')
#    thldwm = get_threshold(gold_dev, predDevwm)
#
#    # Prediction for TEST set
#    score, acc = modelGRU.evaluate(X_test_mid, y_test_mid, batch_size=batch_size)
#    print('Test score:', score)
#    print('Test accuracy:', acc)
#    predTest = modelGRU.predict(X_test_mid)
#    predTest = predTest.reshape((test_shp[0], test_shp[1]))
#
#    predTestmn = np.mean(predTest, axis=1)
#    predTestmn = (predTestmn >= thldmn).astype(int)
#    predictions["gruv"].extend(predTestmn)
#
#    wts = np.linspace(1., 0.01, 2000)
#    predTestwm = np.average(predTest, axis=1, weights=wts)
#    predTestwm = (predTestwm >= thldwm).astype(int)
#    predictions["gruw"].extend(predTestwm)


gold_test = np.array(gold_test)
print("\nResults")
print("\nCNN+V")
bootstrap(gold_test, np.array(predictions["cnnv"]))
predfile = open('predictions/cnnv.pkl', 'wb')
pkl.dump(predictions["cnnv"], predfile)
predfile.close()
print("\nCNN+W")
bootstrap(gold_test, np.array(predictions["cnnw"]))
predfile = open('predictions/cnnw.pkl', 'wb')
pkl.dump(predictions["cnnw"], predfile)
predfile.close()
#print("\nGRU+V")
#bootstrap(gold_test, np.array(predictions["gruv"]))
#predfile = open('predictions/gruv.pkl', 'wb')
#pkl.dump(predictions["gruv"], predfile)
#predfile.close()
#print("\nGRU+W")
#bootstrap(gold_test, np.array(predictions["gruw"]))
#predfile = open('predictions/gruw.pkl', 'wb')
#pkl.dump(predictions["gruw"], predfile)
#predfile.close()
