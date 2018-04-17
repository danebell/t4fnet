
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

# dataset_path='/home/egoitz/Data/Datasets/SocialMedia/twitter4food/overweightData_tokenized/features/'
dataset_path='/home/egoitz/Code/python/git/t4fnet/data_toy/features/'

# In[2]:

import numpy as np
import pickle as pkl
from collections import OrderedDict
import glob
import os

# In[3]:

def build_dict(path):
    features = [None] * 3
    features[0] = set()
    features[1] = set()
    features[2] = set()
    
    sentences = []
    currdir = os.getcwd()
    dirs = [x[0] for x in os.walk(path)]
    print('Reading from %i directories in %s' % (len(dirs), path))
    for dir in dirs:
        least = min(len(dir), len(path))
        if (dir[-least:] != path[-least:]):
            os.chdir(dir)
            print('Reading %i files in %s' % (len(glob.glob('*.txt')), dir))
            for ff in glob.glob("*.txt"):
                with open(ff, 'r') as f:
                    for line in f.readlines():
                        feats, tweet = line.split('\t')
                        feats = feats[1:len(feats)-1].split(', ')
                        features[0].add(feats[0])
                        features[1].add(feats[2])
                        if feats[4] != "":
                            features[2].add(feats[4])
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
            for line in f.readlines():
                feats, tweet = line.split('\t')
                feats = feats[1:len(feats)-1].split(', ')
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
                account.append((tweet.strip(),feats))
        sentences.append((account_id, account))
    os.chdir(currdir)

    seqs = [None] * len(sentences)
    acids =  [None] * len(sentences)
    feats = [None] * len(sentences)
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
        
    return (seqs, acids, feats)


# In[6]:

path = dataset_path
(dictionary, features) = build_dict(path)

(x_pos, i_pos, f_pos) = grab_data(path+'Overweight_da', dictionary, features)
(x_neg, i_neg, f_neg) = grab_data(path+'Notoverweight_da', dictionary, features)
y_pos = [1] * len(x_pos)
y_neg = [0] * len(x_neg)
print('%i accounts' % (len(y_pos) + len(y_neg)))

# f = open('ow3df.pkl', 'wb')
# f = open('data_toy/ow3df.pkl', 'wb')
f = open('data_toy/ow3df2.pkl', 'wb')
pkl.dump((x_pos, i_pos, f_pos, y_pos), f, -1)
pkl.dump((x_neg, i_neg, f_neg, y_neg), f, -1)
f.close()

# f = open('ow3df.dict.pkl', 'wb')
# f = open('data_toy/ow3df.dict.pkl', 'wb')
f = open('data_toy/ow3df2.dict.pkl', 'wb')
pkl.dump(dictionary, f, -1)
pkl.dump(features, f, -1)
f.close()

