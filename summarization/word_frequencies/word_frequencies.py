#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re
from string import punctuation

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances


def detectLanguage(text):
    """
    based on source: 
    http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/
    """
    languages_scores = {}
    
    words = [word.lower() for word in word_tokenize(text)]    
    
    # Compute per language included in nltk number of unique stopwords
    # appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        languages_scores[language] = len(common_elements) # language "score"

    return max(languages_scores, key=languages_scores.get)

#from gensim
PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)

def tokenize(text, lowercase=True, errors="strict"):
    """
    Iteratively yield tokens as unicode strings 
    and optionally lowercasing the unidoce string by assigning True
    to one of the parameters, lowercase, to_lower, or lower.
    Input text may be either unicode or utf8-encoded byte string.
    The tokens on output are maximal contiguous sequences of alphabetic
    characters (no digits!).
    
    """
    
    text = decode_string(text)
    if lowercase:
        text = text.lower()
    
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def simple_preprocess(doc, min_len=2, max_len=20):
    """
    Convert a document into a list of tokens.
    This lowercases, tokenizes, de-accents (optional). -- the output are final
    tokens = unicode strings, that won't be processed any further.
    """
    lang = detectLanguage(doc)
    #stop_words = stopwords.words(lang) + [i for i in punctuation]
    stemmer = SnowballStemmer(lang)
    
    tokens = [
        stemmer.stem(token) for token in tokenize(doc, lowercase=True, errors='ignore')
        #token for token in tokenize(doc, lowercase=True, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_') #and token not in stop_words
    ]
    
    return tokens
    
#from pattern    
def decode_string(v, encoding="utf-8"):
    """ Returns the given value as a Unicode string (if possible).
    """
    if isinstance(encoding, basestring):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(v, str):
        for e in encoding:
            try: return v.decode(*e)
            except:
                pass
        return v
    return unicode(v)

def encode_string(v, encoding="utf-8"):
    """ Returns the given value as a Python byte string (if possible).
    """
    if isinstance(encoding, basestring):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(v, unicode):
        for e in encoding:
            try: return v.encode(*e)
            except:
                pass
        return v
    return str(v)
    
def TFIDF(text):
    lang = detectLanguage(text)
    stop_words = stopwords.words(lang) + [i for i in punctuation]    
    vec = TfidfVectorizer(stop_words=stop_words,tokenizer=simple_preprocess)
    sents = sent_tokenize(text)
    matrix = vec.fit_transform(sents)
    
    feats = vec.get_feature_names()
    idf = vec.idf_
    return FreqDist(dict(zip(feats, idf)))
    
    
def maxTF(text, normalize=True):
    
    lang = detectLanguage(text)
    stop_words = stopwords.words(lang) + [i for i in punctuation]    

    words = simple_preprocess(text)
    clean_words = filter(lambda word: not word in stop_words, words)
    fdist = FreqDist(clean_words)

    # Maximum tf normalization source: 
    # http://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html
    if normalize:
        norm = float(max(fdist.values()))
        a = 0.5
        for word in fdist.keys():
            fdist[word] = a + (1-a) * (fdist[word] / norm)
            # remove too frequent and too rare words
            if fdist[word] >= 0.9 or fdist[word] <= 0.1:
                del fdist[word]
    return fdist

#==============================================================================
# KEYWORDS DETECTION IN RUSSIAN TEXTS

with open ("text1.txt", "r") as f:
    text1=f.read().replace('\n', ' ')
    
with open ("text2.txt", "r") as f:
    text2=f.read().replace('\n', ' ')

text1=decode_string(text1)
text2=decode_string(text2)

for w in TFIDF(text1).most_common(10):
    print w[0]
    
for w in maxTF(text1, normalize=True).most_common(10):
    print w[0]


#==============================================================================
    

#==============================================================================
# COMPUTE DOCS SIMILARITIES
# need to merge texts to single corpus and stemming perform

vec = TfidfVectorizer(stop_words='english',tokenizer=simple_preprocess)
matrix = vec.fit_transform([text1,text2])

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.scatter(matrix.toarray().T[:,0],matrix.toarray().T[:,1], c=['b','r'])
plt.legend()

plt.hist(matrix.toarray().T[:,0], alpha=0.5, histtype='bar', bins=20, log=True, label='text1')
plt.hist(matrix.toarray().T[:,1], alpha=0.5, histtype='bar', bins=20, log=True, label='text2')
plt.title('Tokens Frequency Distribution')
plt.legend()

cos = pairwise_distances(matrix, metric='cosine')
print cos[0,1]

print pairwise_distances(
    X=matrix.toarray()[0,:].reshape(1,-1), 
    Y=matrix.toarray()[1,:].reshape(1,-1), 
    metric='cosine'
)

#==============================================================================

#==============================================================================
# TOKENS INTERSECTION/OVERLAP

stop_words = stopwords.words('russian') + [i for i in punctuation]    

tokens1 = [word.lower() for word in word_tokenize(text1) if word not in stop_words] 
tokens2 = [word.lower() for word in word_tokenize(text2) if word not in stop_words]

# before stemming
intersection = set(tokens1) & set(tokens2)
overlap = float(len(intersection)) / (len(tokens1) + len (tokens2) - len(intersection))

# after stemming
stemmer = SnowballStemmer('russian')

tokens1 = [stemmer.stem(word.lower()) for word in word_tokenize(text1) if word not in stop_words] 
tokens2 = [stemmer.stem(word.lower()) for word in word_tokenize(text2) if word not in stop_words]

intersection = set(tokens1) & set(tokens2)
overlap = float(len(intersection)) / (len(tokens1) + len (tokens2) - len(intersection))
#==============================================================================
