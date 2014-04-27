__author__ = 'Qiji'

import nltk
from nltk.collocations import *

# ------------------Task 2: withpit punctuation and stoplist ---------------------

# load corpus
print "Task 1: without punctuation and stoplist"
print "Start................"
file_corpus = open('corpus.txt')
raw = file_corpus.read()
file_corpus.close()
# load punctuations and stoplist

f1 = open('punctuation.txt')
punc_line = f1.readlines()
punctuation = []
for line in punc_line:
    punctuation.append(line.strip('\n'))
f1.close()

f2 = open('stoplist.txt')
stop_line = f2.readlines()
stoplist = []
for line in stop_line:
    stoplist.append(line.strip('\n'))
f2.close()

# Create bigrams using nltk

tokens = nltk.word_tokenize(raw)
bi_gram = nltk.bigrams(tokens)

# Store bi_gram in hash-map(dict)
# and compute fequencies
fdist_raw = nltk.FreqDist(bi_gram)
fdist = {}
# remove the punctuations

for e in fdist_raw:
    if e[0] in punctuation or e[1] in punctuation or e[0] in stoplist or e[1] in stoplist:
        pass
    else:
        fdist[e] = fdist_raw[e]

# Re sorted the dict with velue in descreasing order
fdist = nltk.FreqDist(fdist)

# get key of the dict
collocation_key = fdist.keys()


top20_coll = collocation_key[:20]
print "The 20 most frequent collocations and their corresponding frequencies\n"
for i in top20_coll:
    print i, fdist[i]
print '\n'
# Outputs the 20 highest scoring collocations according to point-wise mutual information

print "The 20 highest scoring collocations according to point-wise mutual information\n"
bigram_mesures = nltk.collocations.BigramAssocMeasures
finder = BigramCollocationFinder.from_documents(fdist)
top20_pmi = finder.nbest(bigram_mesures.pmi, 20)
for i in top20_pmi:
    print i
print '\n'

# Outputs the 20 highest scoring collocations according to Pearson's (Chi-squared) test
print "The 20 highest scoring collocations according to Pearson's (Chi-squared) test\n"
top20_chi_sq = finder.nbest(bigram_mesures.chi_sq, 20)
for i in top20_chi_sq:
    print i
print '\n'

# Outputs the 20 highest scoring collocations according to Dunning's log-likelihood test.
print "The 20 highest scoring collocations according to Dunning's log-likelihood test.\n"
top20_likelihood = finder.nbest(bigram_mesures.likelihood_ratio, 20)
for i in top20_likelihood:
    print i
print '\n'

print "----------------------------------------------------------------------------------"
print "end............"
