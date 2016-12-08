#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import bs4, urllib2, re
from nltk.corpus import stopwords

def detectLanguage(text):
    """
    based on source: 
    http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/
    """
    languages_scores = {}
    words = simple_preprocess(text)
    # Compute per language included in nltk number of unique stopwords
    # appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        languages_scores[language] = len(common_elements) # language "score"

    return max(languages_scores, key=languages_scores.get)

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
    tokens = [
        token for token in tokenize(doc, lowercase=True, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
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
