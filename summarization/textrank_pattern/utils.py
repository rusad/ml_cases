#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import combinations
import networkx as nx
from collections import Counter
from pattern.text import decode_string
import bs4, urllib2

def levenshtein(s1, s2):
    """
    compute levenshtein distance 
    source: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python    
    """    
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def textrank(items, distance_func):
    """
    compute textrank, using ntworkx pagerank function
    steps:    
    1. tokenize text by sentences
    2. compute tfidf matrix
    3. compute cosine distance of tfidf matrix
    4. create graph based on cosine distance matrix
    5. compute pagerank
    - items: list of sentences or words
    - distance_func: function to compute distance matrix
    """
    # pairwise combinations    
    pairs = list(combinations(items, 2))

    # compute distance between every pair and set it as weight of graph edge
    weighted_edges = []
    
    for i in range(len(pairs)):
        # distance define as weight of edge
        weight = distance_func(pairs[i][0], pairs[i][1])
        
        # check whether item is instance of Document class or just string        
        if isinstance(pairs[i][0], (str,unicode)):
            weighted_edges.append((pairs[i][0], pairs[i][1], weight))
        else:
            weighted_edges.append((pairs[i][0].name, pairs[i][1].name, weight))
    
    # create graph    
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    
    # calculate pagerank
    pr = nx.pagerank(G, alpha=0.85)
    return Counter(pr)

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
