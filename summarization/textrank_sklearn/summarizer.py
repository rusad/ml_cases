#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from string import punctuation
from itertools import combinations
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import utils, sys

def summarize(text, n=1):
    """
    extract most relevant sentences from text according to TextRank algorithm
    steps:    
    1. tokenize text by sentences
    2. compute tfidf matrix
    3. compute cosine distance of tfidf matrix
    4. create graph based on cosine distance matrix
    5. compute pagerank
    
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
    
    # compute pairwise cosine similarity of each sentence
    cos = pairwise.pairwise_distances(matrix, metric='cosine')
    
    # build graph based on similarity matrix
    G = nx.from_numpy_matrix(cos)
    
    # compute pagerank of graph nodes
    pr = nx.pagerank(G, alpha=0.85)
    
    # indexes of most relevant sentences
    top_sents_idx, _ = list(zip(*Counter(pr).most_common(n)))

    # reordering
    output = [sents[i] for i in sorted(top_sents_idx)]
    
    return ' '.join(output)
    
def keywords(text, n=15):
    """
    extract most relevant keywords from given text
    steps:    
    1. tokenize text by words
    2. applying synctatic filter
    3. compute pairwise levenshtein distance
    4. create graph based on cosine distance matrix
    5. compute pagerank
    
    - text: string consisting of a few sentences
    - n: number of keywords to extract
    """    
    import nltk    
    synctatic_filter = ['NN', 'JJ']
    
    # tokenizung by words
    words = word_tokenize(text)
    
    # pos-tagging
    tagged = nltk.pos_tag(words)
    
    #applying synctatic filter
    filtered = [i[0].lower() for i in tagged if i[1] in synctatic_filter]

    # pairwise combinations
    pairs = list(combinations(filtered, 2))
    
    # compute distance between every pair and set it as weight of graph edge
    weighted_edges = []
    
    for i in range(len(pairs)):
        # distance define as weight of edge
        weight = utils.levenshtein(pairs[i][0], pairs[i][1])
        weighted_edges.append((pairs[i][0], pairs[i][1], weight))
        
    # create graph       
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    
    # calculate pagerank
    pr = nx.pagerank(G, alpha=0.85)
    
    # dict of TextRank ranking of levenshtein distance matrix    
    ranking = Counter(pr)
    
    # top n keywords
    keywords, scores = list(zip(*ranking.most_common(n)))
    
    return keywords, scores
    
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
                words, scores = keywords(text, n=15)
                x, y = dispersion(text, words)
                words = [w for w in words]                
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
