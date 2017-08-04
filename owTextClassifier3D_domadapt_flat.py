
# coding: utf-8

#
#
# NN models to classify Twitter account users as Overweight or Not Overweight.
# 
#
CUDA_MODE = False
SEED = 947

import argparse
import gzip
import numpy as np
np.random.seed(SEED) # for reproducibility
import pickle as pkl
import sys
import math
import os

from keras.preprocessing import sequence

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim


parser = argparse.ArgumentParser(description='t4f-NN with domain adaptation.')
parser.add_argument('--dir',
                    help='directory to stores models and predictions')
parser.add_argument('--gender', action='store_true',
                    help='apply domain adapatation for gender')
parser.add_argument('--retweet', action='store_true',
                    help='apply domain adapatation for retweet')
parser.add_argument('--fold', default=None,
                    help='run for an expecific fold.')

args = parser.parse_args()

base_dir = args.dir
run_fold = args.fold
model_dir = base_dir + '/models/'
pred_dir = base_dir + '/predictions/'
domain = [False, False]
domain[0] = args.gender
domain[1] = args.retweet

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)


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

    folds = load_folds(foldsfile, seed=SEED)
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

        X_train = pad3d(X_train, maxtweets=maxtweets, maxlen=maxlen)
        X_test = pad3d(X_test, maxtweets=maxtweets, maxlen=maxlen)
        X_dev = pad3d(X_dev, maxtweets=maxtweets, maxlen=maxlen)
        train_shp = X_train.shape
        test_shp = X_test.shape
        dev_shp = X_dev.shape
        
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
    
class CNN(nn.Module):
    def __init__(self, max_features, embedding_dim, seq_length, nb_filter,
                 filter_length, pool_length, hidden_size, feats=0):
        super(CNN, self).__init__()
        cnn_out_length = new_outs_lengths(seq_length, filter_length)
        pool_out_length = new_outs_lengths(cnn_out_length, pool_length, stride=pool_length)
        self.embs = nn.Embedding(max_features + 3, embedding_dim)
        self.cnn = nn.Conv1d(embedding_dim, nb_filter, filter_length)
        self.pool = nn.MaxPool1d(pool_length)        
        self.linear1 = nn.Linear(int(pool_out_length) * nb_filter, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(hidden_size * (1 + feats * 2), 1)
        self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, inputs, intermediate=False, test_mode=False, domain=[None,None]):
        embeds = self.embs(inputs)
        embeds = embeds.transpose(0, 1).transpose(1, 2)
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
      
        if domain[1] == 0:
            out = torch.cat((out,outc,zeros),1)
        elif domain[1] == 1:
            out = torch.cat((out,zeros,outc),1)
                       
        if not test_mode:
            out = self.dropout1(out)
        out = self.sigmoid2(self.linear2(out))

        return out


        
def predict(net, x, f, batch_size, intermediate=False, domain=[False,False]):
    pred = np.empty(0)
    batches = math.ceil(x.size()[0] / batch_size)
    for b in range(batches):
        bx = x[b*batch_size:b*batch_size+batch_size]

        # No domain
        if domain[0] == domain[1] == False:
            bx = torch.transpose(bx, 0, 1)
            b_pred = net(bx, test_mode=True)
            del(bx)
            
        # Only one domain
        elif domain[0] == False or domain[1] == False:
            bf = f[b*batch_size:b*batch_size+batch_size]
            
            if domain[1] == False:
                fb = torch.LongTensor(torch.np.where(bf[:,0]==0)[0])
            else:
                fb = torch.LongTensor(torch.np.where(bf[:,2]==0)[0])
            if CUDA_MODE:
                fb = fb.cuda()
                f_pred = Variable(torch.LongTensor().cuda())
            else:
                f_pred = Variable(torch.LongTensor())
            if fb.dim() > 0:
                bxf = bx[fb]
                bxf = torch.transpose(bxf, 0, 1)
                if domain[1] == False:
                    f_pred = net(bxf, test_mode=True, domain=[0,None])
                else:
                    f_pred = net(bxf, test_mode=True, domain=[None,0])
                del(bxf)

            if domain[1] == False:
                mb = torch.LongTensor(torch.np.where(bf[:,0]==1)[0])
            else:
                mb = torch.LongTensor(torch.np.where(bf[:,2]==1)[0])
            if CUDA_MODE:
                mb = mb.cuda()
                m_pred = Variable(torch.LongTensor().cuda())
            else:
                m_pred = Variable(torch.LongTensor())
            if mb.dim() > 0:
                bxm = bx[mb]
                bxm = torch.transpose(bxm, 0, 1)
                if domain[1] == False:
                    m_pred = net(bxm, test_mode=True, domain=[1,None])
                else:
                    m_pred = net(bxm, test_mode=True, domain=[None,1])
                del(bxm)
    
            del(bf)
                    
            if fb.dim() > 0 and mb.dim() > 0:
                cb = torch.cat((fb, mb))
                del(fb, mb)
                b_pred = torch.cat((f_pred, m_pred))
                del(f_pred, m_pred)
            elif fb.dim() > 0:
                cb = fb
                del(fb)
                b_pred = f_pred
                del(f_pred)
            else:
                cb = mb
                del (mb)
                b_pred = m_pred
                del(m_pred)
    
            if CUDA_MODE:
                cb = torch.LongTensor(torch.np.argsort(cb.cpu().numpy())).cuda()
            else:
                cb = torch.LongTensor(torch.np.argsort(cb.numpy()))    
    
            b_pred = b_pred[cb]
            del(cb)
            
        # Two domains
        else:
            bf = f[b*batch_size:b*batch_size+batch_size]

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

    
def train(net, x, y, f, nepochs, batch_size, domain=[False,False]):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    batches = math.ceil(x.size()[0] / batch_size)
    for e in range(nepochs):
        running_loss = 0.0
        for b in range(batches):
            # Get minibatch samples
            bx = x[b*batch_size:b*batch_size+batch_size]
            by = y[b*batch_size:b*batch_size+batch_size]
            
            # Clear gradients
            net.zero_grad()

            # No domain
            if domain[0] == domain[1] == False:
                bx = torch.transpose(bx, 0, 1)
                y_pred = net(bx)
                del(bx)
            
            # Only one domain
            elif domain[0] == False or domain[1] == False:
                bf = f[b*batch_size:b*batch_size+batch_size]

                if domain[1] == False:
                    fb = torch.LongTensor(torch.np.where(bf[:,0]==0)[0])
                else:
                    fb = torch.LongTensor(torch.np.where(bf[:,2]==0)[0])                    
                if CUDA_MODE:
                    fb = fb.cuda()
                bxf = bx[fb]
                byf = by[fb]
                del(fb)
                bxf = torch.transpose(bxf, 0, 1)
    
                if domain[1] == False:
                    mb = torch.LongTensor(torch.np.where(bf[:,0]==1)[0])
                else:
                    mb = torch.LongTensor(torch.np.where(bf[:,2]==1)[0])
                    
                if CUDA_MODE:
                    mb = mb.cuda()
                bxm = bx[mb]
                bym = by[mb]
                del(mb)
                bxm = torch.transpose(bxm, 0, 1)
                                
                del(bx, bf)

                # Forward pass
                if domain[1] == False:
                    yf_pred = net(bxf, domain=[0,None])
                    del(bxf)
                    ym_pred = net(bxm, domain=[1,None])
                    del(bxm)
                else:
                    yf_pred = net(bxf, domain=[None,0])
                    del(bxf)
                    ym_pred = net(bxm, domain=[None,1])
                    del(bxm)
                    
                by = torch.cat((byf, bym))
                del(byf, bym)
                y_pred = torch.cat((yf_pred, ym_pred))
                del(yf_pred, ym_pred)

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
            loss = criterion(y_pred, by)
            del(by)
            
            # Print loss
            running_loss += loss.data[0]
            sys.stdout.write('\r[epoch: %3d, batch: %3d/%3d] loss: %.3f' % (e + 1, b + 1, batches, running_loss / (b+1)))
            sys.stdout.flush()

            # Backward propagation and update the weights.
            loss.backward()
            optimizer.step()
            del(y_pred)
        sys.stdout.write('\n')
        sys.stdout.flush()


max_features = 50000
maxtweets = 2000
maxlen = 50  # cut texts to this number of words (among top max_features most common words)
emb_dim = 200
embeddings = load_embeddings(nb_words=max_features, emb_dim=emb_dim)
nb_filter = 64 # how many convolutional filters
filter_length = 5 # how many tokens a convolution covers
pool_length = 4 # how many cells of convolution to pool across when maxing
nb_epoch = 1 # how many training epochs
batch_size = 256 # how many tweets to train at a time
predict_batch_size = 612


if run_fold is not None:
    run_fold = 'fold' + str(run_fold)
pos, neg = load_data(nb_words=max_features, maxlen=maxlen, seed=SEED)
predictions = dict()
predictions["cnnv"] = list()
predictions["cnnw"] = list()
predictions["gruv"] = list()
predictions["gruw"] = list()
gold_test = list()
iterations = list()
foldsfile = "folds.csv"
#foldsfile = "data_toy/folds.csv"
for iteration in gen_iterations(pos, neg, max_features, maxtweets, maxlen, foldsfile):
    iterid = iteration[0]
    if run_fold is not None and iterid != run_fold:
        continue
    iterations.append(iterid)
    print('')
    print('Iteration: %s' % iterid)
    (_, X_train_shuff, _, _, y_train_shuff,
     _, _, f_train_shuff, train_shp) = iteration[1]
    (X_test_flat, _, y_test, _, _,
     _, f_test_flat, _, test_shp) = iteration[2]
    (X_dev_flat, _, y_dev, _, _,
     _, f_dev_flat, _, dev_shp) = iteration[3]
    
    print('X_train shape:', train_shp)
    print('X_test shape:', test_shp)
    print('X_dev shape:', dev_shp)
 
     
    gold_dev = y_dev.flatten()
    gold_test.extend(y_test.flatten())

    if not (os.path.isfile(pred_dir + 'cnnv_' + iterid + '.pkl') and 
        os.path.isfile(pred_dir + 'cnnw_' + iterid + '.pkl')):
        #
        # Pre-train tweet-level vectors
        #
    
        print('Build first model (tweet-level)...')
        num_feats = int(np.sum(np.array(domain)==True))
        net = CNN(max_features, emb_dim, maxlen, nb_filter, filter_length, pool_length, 128, feats=num_feats)

        # Train or load the model
        if (os.path.isfile(model_dir + 'tweet_classifier_' + iterid + '.pkl')):
            print('Loading model weights...')
            net.load_state_dict(torch.load(model_dir + 'tweet_classifier_' + iterid + '.pkl'))
            if CUDA_MODE:
                net = net.cuda()
        else:
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
            train(net, data_x, data_y, data_f, nb_epoch, batch_size, domain=domain)
            del(data_x, data_y, data_f)
            torch.save(net.state_dict(), model_dir + 'tweet_classifier_' + iterid + '.pkl')
            
    
        #
        #  CNN+V/CNN+W
        #
    
        # Prediction for DEV set
        print('Dev...')
        if CUDA_MODE:
            data_x = Variable(torch.from_numpy(X_dev_flat).long().cuda())
        else:
            data_x = Variable(torch.from_numpy(X_dev_flat).long())
        data_f = f_dev_flat
        
        predDev = predict(net, data_x, data_f, predict_batch_size, domain=domain)
        del(data_x, data_f)
        predDev = predDev.reshape((dev_shp[0], dev_shp[1]))

        predDevmn = np.mean(predDev, axis=1)
        print('Search CNN+V threshold')
        thldmn = get_threshold(gold_dev, predDevmn)
        del(predDevmn)
        
        wts = np.linspace(1., 0.01, 2000)
        predDevwm = np.average(predDev, axis=1, weights=wts)
        print('Search CNN+W threshold')
        thldwm = get_threshold(gold_dev, predDevwm)
        del(predDevwm)
        del(predDev, gold_dev)
        
        #Prediction for TEST set
        print('Test...')
        if CUDA_MODE:
            data_x = Variable(torch.from_numpy(X_test_flat).long().cuda())
        else:
            data_x = Variable(torch.from_numpy(X_test_flat).long())
        data_f = f_test_flat
        predTest = predict(net, data_x, data_f, predict_batch_size, domain=domain)
        del(data_x, data_f)
        predTest = predTest.reshape((test_shp[0], test_shp[1]))
    
        print('CNN+V with threshold = ', thldmn)
        predTestmn = np.mean(predTest, axis=1)
        predTestmn = (predTestmn >= thldmn).astype(int)
        predfile = open(pred_dir + 'cnnv_' + iterid + '.pkl', 'wb')
        pkl.dump(predTestmn, predfile)
        predfile.close()
        del(predTestmn)

        
        print('CNN+W with threshold = ', thldwm)
        wts = np.linspace(0.01, 1., 2000)
        predTestwm = np.average(predTest, axis=1, weights=wts)
        predTestwm = (predTestwm >= thldwm).astype(int)
        predfile = open(pred_dir + 'cnnw_' + iterid + '.pkl', 'wb')
        pkl.dump(predTestwm, predfile)
        predfile.close()
        del(predTestwm)


if run_fold is None:  
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
