
# coding: utf-8

# # Text preprocessing
# 
# This script pickles the dataset for use in LSTMs etc.
# 
# It is adapted from the preprocessing script at http://deeplearning.net/tutorial/lstm.html
# 
# Reserved indices:
# * 0 = padding
# * 1 = start
# * 2 = OOV (out of vocabulary)

# In[1]:

#dataset_path=['/data/nlp/corpora/twitter4food/diabetes/rawTokens/risk_da',
#              '/data/nlp/corpora/twitter4food/diabetes/rawTokens/not_da']


#dataset_path=['/data/nlp/corpora/twitter4food/ow2/rawTokens/Overweight_da',
#              '/data/nlp/corpora/twitter4food/ow2/rawTokens/Notoverweight_da']

# dataset_path=['/data/nlp/corpora/twitter4food/diabetes/rawTokens/risk_da_dictOnly',
#               '/data/nlp/corpora/twitter4food/diabetes/rawTokens/not_da_dictOnly']

# dataset_path=['/data/nlp/corpora/twitter4food/diabetes/rawTokens/risk_gold_dictOnly',
#               '/data/nlp/corpora/twitter4food/diabetes/rawTokens/not_gold_dictOnly']

# dataset_path=['/data/nlp/corpora/twitter4food/diabetes/rawTokens/risk_pred_dictOnly',
#              '/data/nlp/corpora/twitter4food/diabetes/rawTokens/not_pred_dictOnly']

dataset_path=['/code/netsing/toy/risk_pred_dictOnly',
             '/code/netsing/toy/not_pred_dictOnly']

# In[2]:

import numpy as np
import pickle as pkl
from collections import OrderedDict
import glob
import os
import sys
# In[3]:

def build_dict(dirs):
    features = [None] * 3
    features[0] = set()
    features[1] = set()
    features[2] = set()
    
    sentences = []
    currdir = os.getcwd()
    #dirs = [x[0] for x in os.walk(path)]
    print('Reading from %i directories' % len(dirs))
    for dir in dirs:
        least = len(dir)
        #if (dir[-least:] != path[-least:]):
        os.chdir(dir)
        print('Reading %i files in %s' % (len(glob.glob('*.txt')), dir))
        for ff in glob.glob("*.txt"):
            with open(ff, 'r') as f:
                for line in f.readlines():
                    feats, tweet = line.split('\t')
                    feats = feats[1:len(feats)-1].split(', ')
                    features[0].add(feats[0])
                    features[1].add(feats[2])
                    features[2].add("UNK")
                    #if feats[4] != "":
                    #    features[2].add(feats[4])
                    sentences.append(tweet.strip())
    os.chdir(currdir)

    features[0] = list(features[0])
    features[1] = list(features[1])
    features[2] = list(features[2])
        
    print('Found %i sentences' % len(sentences))

    print('Building dictionary')
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w in wordcount:
                wordcount[w] = wordcount[w] + 1
            else:
                wordcount[w] = 1

    counts = list(wordcount.values())
    keys = list(wordcount.keys())
    
    sorted_idx = np.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx

    print(np.sum(counts), ' total words ', len(keys), ' unique words')

    return (worddict, features)


def grab_data(path, dictionary, features):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        account_id = ff.split('.')[0]
        account = []
        with open(ff, 'r') as f:
            feats = None
            accounttweets = list()
            tweetorder = list()
            tweetcounter = 0
            for line in f.readlines():
                feats, tweet = line.split('\t')
                feats = feats[1:len(feats)-1].split(', ')
                feats.append("UNK")
                feats[0] = features[0].index(feats[0])
                if feats[1] == "UNK":
                    feats[1] = -1.
                else:
                    feats[1] = float(feats[1])
                feats[2] = features[1].index(feats[2])
                if feats[4] not in features[2]:
                    feats[4] = -1
                else:
                    feats[4] = features[2].index(feats[4])
                accounttweets.append(tweet.strip())
                tweetcounter += 1
                tweetorder.extend([tweetcounter]*len(tweet.strip().split(' ')))
                if feats is not None and tweetcounter == -1: # set tweetcounter == -1 to get a unique tweet per account
                # if feats is not None and tweetcounter == 10: # set tweetcounter == -1 to get a unique tweet per account
                    account.append((" ".join(accounttweets),feats))
                    feats = None
                    accounttweets = list()
                    tweetcounter = 0
            if feats is not None:
                feats.append(tweetorder)
                account.append((" ".join(accounttweets),feats))
        sentences.append((account_id, account))
    os.chdir(currdir)

    seqs = [None] * len(sentences)
    acids =  [None] * len(sentences)
    feats = [None] * len(sentences)
    order = [None] * len(sentences)
    for i, (acid, account) in enumerate(sentences):
        acids[i] = acid
        seqs[i] = [None] * len(account)
        feats[i] = [None] * (2 + len(features[2]))
        for j, (ss, ft) in enumerate(account):
            feats[i][0] = ft[0]
            feats[i][1] = ft[2]
            feat_list = [0] * len(features[2])
            if ft[4] > -1:
                feat_list[ft[4]] = 1
            feats[i][2:] = feat_list
            seqs[i][j] = []
            words = ss.strip().lower().split()
            for w in words:
                if w in dictionary:
                    seqs[i][j].append(dictionary[w]) # skip OOV here
            order[i] = ft[-1]/np.amax(ft[-1])
            if np.any(np.isnan(order[i])):
                order[i] = np.nan_to_num(order[i]) + 1. # if nan found it means only one tweet in the account

    return (seqs, acids, feats, order)


# In[6]:

dirs = dataset_path
(dictionary, features) = build_dict(dirs)

(x_pos, i_pos, f_pos, o_pos) = grab_data(dirs[0], dictionary, features)
(x_neg, i_neg, f_neg, o_neg) = grab_data(dirs[1], dictionary, features)
y_pos = [1] * len(x_pos)
y_neg = [0] * len(x_neg)
print('%i accounts' % (len(y_pos) + len(y_neg)))

# f = open('risk3df2.pkl', 'wb')
# f = open('dow3df.pkl', 'wb')
# f = open('risk3dfdictu.pkl', 'wb')
# f = open('risk3dfdictu10t.pkl', 'wb')
# f = open('risk3dfdictuwo.pkl', 'wb')
# f = open('risk3dfdictugwo.pkl', 'wb')
f = open('toy.pkl', 'wb')

pkl.dump((x_pos, i_pos, f_pos, y_pos, o_pos), f, -1)
pkl.dump((x_neg, i_neg, f_neg, y_neg, o_neg), f, -1)
f.close()

# f = open('risk3df.dict.pkl', 'wb')
# f = open('dow3df.dict.pkl', 'wb')
# f = open('risk3dfdictu.dict.pkl', 'wb')
# f = open('risk3dfdictu10t.dict.pkl', 'wb')
# f = open('risk3dfdictuwo.dict.pkl', 'wb')
# f = open('risk3dfdictugwo.dict.pkl', 'wb')
f = open('toy.dict.pkl', 'wb')
pkl.dump(dictionary, f, -1)
pkl.dump(features, f, -1)
f.close()

