__author__ = 'Qiji'

import nltk
from nltk.collocations import *

# ------------------Task 1: with punctuation and stoplist ---------------------

# load corpus
print "Task 1: with punctuation and stoplist"
print "Start................"
file_corpus = open('corpus.txt')
raw = file_corpus.read()
file_corpus.close()

# Create bigrams using nltk

tokens = nltk.word_tokenize(raw)
bi_gram = nltk.bigrams(tokens)

# Store bi_gram in hash-map(dict)
# and compute fequencies
fdist = nltk.FreqDist(bi_gram)
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
