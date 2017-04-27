
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

dataset_path='/Users/laparra/Data/Datasets/SocialMedia/twitter4food/overweightData_tokenized/rawTokens/'


# In[2]:

import numpy as np
import pickle as pkl
from collections import OrderedDict
import glob
import os


# In[3]:

def build_dict(path):
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
                        sentences.append(line.strip())
    os.chdir(currdir)
    
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

    return worddict


def grab_data(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        account = []
        with open(ff, 'r') as f:
            for line in f.readlines():
                account.append(line.strip())
        sentences.append(account)
    os.chdir(currdir)

    seqs = [None] * len(sentences)
    for i, account in enumerate(sentences):
        seqs[i] = [None] * len(account)
        for j, ss in enumerate(account):
            seqs[i][j] = []
            words = ss.strip().lower().split()
            for w in words:
                if w in dictionary:
                    seqs[i][j].append(dictionary[w]) # skip OOV here
        
    return seqs


# In[4]:

path = dataset_path
dictionary = build_dict(os.path.join(path, 'train'))

train_x_pos = grab_data(path+'train/Overweight', dictionary)
train_x_neg = grab_data(path+'train/Notoverweight', dictionary)
train_x = train_x_pos + train_x_neg
train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)
print('%i training accounts' % len(train_y))

test_x_pos = grab_data(path+'test/Overweight', dictionary)
test_x_neg = grab_data(path+'test/Notoverweight', dictionary)
test_x = test_x_pos + test_x_neg
test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)
print('%i testing accounts' % len(test_y))

f = open('ow3d.pkl', 'wb')
pkl.dump((train_x, train_y), f, -1)
pkl.dump((test_x, test_y), f, -1)
f.close()

f = open('ow3d.dict.pkl', 'wb')
pkl.dump(dictionary, f, -1)
f.close()

