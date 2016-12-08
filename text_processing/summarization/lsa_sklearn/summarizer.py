#!/usr/bin/env python
# -*- coding: utf-8 -*-


#import os, sys; sys.path.insert(0, os.path.join(os.path.dirname('__file__'), ".."))

#os.path.dirname(os.path.realpath('__file__'))

#os.getcwd()

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from string import punctuation

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import utils, sys

def word_ranking(text):
    """
    extract most relevant sentences from text according to LSA algorithm
    steps:    
    1. tokenize text by sentences
    2. compute tfidf matrix
    3. applying SVD of tfidf matrix 
    4. ranking sentences according to cross-method (source: http://www.aclweb.org/anthology/C10-1098.pdf)
        
    - text: string consisting of a few sentences
    - n: number of sentences to extract
    
    """     
    
    # input text language detection
    lang = utils.detectLanguage(text)

    # stop words and punctuation    
    bad_items = stopwords.words(lang) + [i for i in punctuation]

    # build tfidf vectorizer
    vec = TfidfVectorizer(stop_words=bad_items,tokenizer=utils.simple_preprocess)
    
    # apply vectorizer to tokenized text by sentences
    sents = sent_tokenize(text)
    matrix = vec.fit_transform(sents)    
    
    U, S, V = np.linalg.svd(matrix.toarray(), full_matrices=False)
    
    # average sentence score for each concept/topic by the rows of the Vt matrix
    avg_score = np.mean(V,axis=1).reshape((-1,1))
    
    # cell values which are less than or equal to the average score are set to zero
    V[V <= avg_score] = 0.0
    
    # total length of each sentence vector      
    length = np.sum(V*S.reshape((-1,1)),axis=0)
    
    # ranking words by length score    
    ranking = Counter(dict(zip(vec.get_feature_names(),length)))#.most_common(n)
        
    return ranking
    
def summarize(text, n=2):
    """
    determine most informative sentences by summarizing words ranks
    which occure in the corresponding  sentences
    """        
    # tokenize text to sentences list
    sentences = sent_tokenize(text)
    
    # tokenize sentence list by words
    words_sent = [sent.lower().split() for sent in sentences]
    
    # words ranking
    w_ranking = word_ranking(text)
    
    # sents ranking = sum of words score
    s_ranking = defaultdict(int)
    
    for i, sent in enumerate(words_sent):
        for word in sent:
            if word in w_ranking:
                s_ranking[i] += w_ranking[word]
    
    # placed sents ranking into high-performance container    
    s_ranking = Counter(s_ranking)
    
    # get top n sents indexes with scores    
    sents_idx = s_ranking.most_common(n)
    
    output = [sentences[j[0]] for j in sents_idx]
            
    # reordering 
    output.sort(lambda s1, s2: text.find(s1) - text.find(s2))
    
    return ' '.join(output)
    
def dispersion(text, keywords):
    """
    Dispersion of occurence of given keywords among given text
    - text: string 
    - keywords: list of keywords
    """
    
    # tokenize by words    
    tokens = [word.lower() for word in word_tokenize(text)]
        
    n_tokens = len(tokens)
    n_words = len(keywords)
    disp = []
    
    for x in range(n_tokens):
        for y in range(n_words):
            if tokens[x] == keywords[y]:
                disp.append((x,y))
                
    x, y = list(zip(*disp))
    return x, y
    
def draw(words, scores, x, y):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
    fig = plt.figure(figsize=(14,6))
    fig.subplots_adjust(wspace = 0.3)
        
    ax1 = fig.add_subplot(121)
    plt.yticks(range(len(words)), words)
    ax1.barh(range(len(scores)), scores, align='center', alpha=0.4)
    ax1.set_title('Key Words Frequency')    
    ax1.set_xlabel('Normalized frequency')    
    
    ax2 = fig.add_subplot(122, sharey=ax1)
    plt.yticks(range(len(words)), words)
    ax2.plot(x, y, "b|", scalex=.1, markersize=10)
    ax2.set_title('Lexical Dispersion Plot')
    ax2.set_xlabel("Word Offset")
    
    plt.savefig('key_words_fig.pdf', format='pdf')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = utils.get_text(sys.argv[1])
        if len(sys.argv) > 2:
            print summarize(text, int(sys.argv[2]))
        else:
            print summarize(text)
            
        while True:
            is_drawing = raw_input("Want to draw key words dispersion plot?[y/n]:")
            if is_drawing == "y" or is_drawing == "Y":
                keywords_ranking = word_ranking(text).most_common(15)
                words, scores =  list(zip(*keywords_ranking))
                x, y = dispersion(text, words)
                words = [utils.encode_string(w) for w in words]                
                draw(words, scores, x, y)
                break
            elif is_drawing == "n" or is_drawing == "N":
                break
            else:
                print("Incorrect command.")            
            
        sys.exit(0)
    else:
        print('There is no text to summarize')
        sys.exit(1)
