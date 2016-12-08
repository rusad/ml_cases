#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pattern.vector import Document, Model, TFIDF
from pattern.en import tokenize, tag
from pattern.text import encode_string
import utils, sys

def summarize(text, n=1):
    """
    extract most relevant sentences from text according to TextRank algorithm
    - text: string consisting of a few sentences
    - n: number of sentences to extract
    """    
    # tokenize text to sentences list        
    sentences = tokenize(text)
    
    # create documents list
    # stop words and punctuation erase by default
    docs = [Document(sentences[i], name=i) for i in range(len(sentences))] 

    # model initialize    
    m = Model(docs, weight=TFIDF)

    # dict of TextRank ranking of cosine similarity matrix
    ranking = utils.textrank(m.documents, m.distance)

    # indexes of top n sentences    
    top_sents_idx, _ = list(zip(*ranking.most_common(n)))
    
    # reordering
    output = [sentences[i] for i in sorted(top_sents_idx)]
    
    return ''.join(output)

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
    # tokenize text to sentences list        
    sentences = tokenize(text)    
    
    #synctatic filter    
    words = []
    for sent in sentences:
        for word, pos in tag(sent):
            if pos == "JJ" or pos == 'NN': # Retrieve all adjectives and nouns.
                words.append(word.lower())
                 
    # dict of TextRank ranking of levenshtein distance matrix
    ranking = utils.textrank(words, utils.levenshtein)
    
    # top n keywords
    keywords, scores = list(zip(*ranking.most_common(n)))
    return keywords, scores
    
def dispersion(text, keywords):
    """
    Dispersion of occurence of given keywords among given text
    - text: string 
    - keywords: list of keywords
    """
    # tokenize text to sentences list        
    sentences = tokenize(text)
    
    # tokenize by words    
    tokens = []
    for sent in sentences:
        for w in sent.lower().split():
            tokens.append(w)
    
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
                words = [encode_string(w) for w in words]                
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

