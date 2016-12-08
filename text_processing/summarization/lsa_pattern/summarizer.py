#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bs4, urllib2, sys
import numpy as np
from pattern.text import decode_string, encode_string
from pattern.vector import Document, Model, TFIDF
from pattern.en import tokenize, tag
from collections import Counter, defaultdict

def get_text(src):
    """
    Check if given src is url, so retrieve text from html, if not then 
    return inputted src. There is assumed that text is largest sequence of
    p-tags
    """    
    # check is url    
    try:
        html = decode_string(urllib2.urlopen(src).read())
        raw = bs4.BeautifulSoup(html, 'html.parser')
        # find text content: get largest sequence of p tag, 
        # using BeautifulSoup
        # cleaning body from script and style tags        
        if len(raw.select('script style')) > 0:
           for script in raw.body(["script", "style"]):
                script.extract()
        # largest sequence of p tags    
        return max(raw.find_all(), 
                   key=lambda t: len(t.find_all('p', recursive=False))).get_text()
        
    except ValueError:
        return src
        

def word_ranking(text, n='L2'):
    """
    extract most relevant sentences from text according to LSA algorithm
    steps:    
    1. tokenize text by sentences
    2. compute tfidf matrix
    3. applying SVD of tfidf matrix (reduce to n-dimensions) 
    4. ranking sentences according to cross-method (source: http://www.aclweb.org/anthology/C10-1098.pdf)
        
    - text: string consisting of a few sentences
    - n: number of sentences to extract
    
    """ 
    # tokenize text to sentences list
    sentences = tokenize(text)
    
#==============================================================================
#     #synctatic filter    
#     exclude_list = []
#     for sent in sentences:
#         for word, pos in tag(sent):
#             if pos != "JJ" or pos != 'NN': # Retrieve all adjectives and nouns.
#                 exclude_list.append(word.lower())
#==============================================================================
    
    # create documents list
    # stop words and punctuation erase by default
    docs = [Document(sentences[i], name=i) for i in range(len(sentences))] 

    # model initialize    
    m = Model(docs, weight=TFIDF)
    
    # dimensions number equal to euclidean norm of singular values
    # U, S, Vt = np.linalg.svd(m.vectors, full_matrices=False)
    # dimensions=int(round(np.linalg.norm(S, 2)))
    m.reduce(dimensions=n)
    
    # sentences selection according to cross-method
    # source: http://www.ceng.metu.edu.tr/~e1395383/papers/TextSummarizationUsingLSA(Journal).pdf        
    # topic(rows) x tokens(cols) matrix(tfidf)
    V = np.array(m.lsa.vt)

    # average sentence score for each concept/topic by the rows of the Vt matrix
    avg_score = np.mean(V,axis=1).reshape((-1,1))
    
    # cell values which are less than or equal to the average score are set to zero
    V[V <= avg_score] = 0.0

    # sigma natrix after svd performing
    S = np.array(m.lsa.sigma).reshape((-1,1))
    
    # total length of each sentence vector      
    length = np.sum(V*S,axis=0)
    
    # ranking words by length score    
    ranking = Counter(dict(zip(m.lsa.terms,length)))#.most_common(n)
    
    #words, score =  list(zip(*ranking))   
    
    return ranking
    
def summarize(text, n=2):
    """
    determine most informative sentences by summarizing words ranks
    which occure in the corresponding  sentences
    """        
    # tokenize text to sentences list
    sentences = tokenize(text)
    
    # tokenize sentence list by words
    words_sent = [sent.lower().split() for sent in sentences]
    
    # words ranking
    w_ranking = word_ranking(text, n)
    
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
        text = get_text(sys.argv[1])
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
