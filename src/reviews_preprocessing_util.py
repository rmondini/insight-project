import pandas as pd
import numpy as np
import csv
import string
from nltk import pos_tag,word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords,wordnet
import nltk.data

# remove new-line character and replace it with full stop
def remove_new_line(review_item):
    if pd.isnull(review_item):
        return np.nan
    else:
        return review_item.replace('\n', '. ').replace('\r', '. ')

# define sentence tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# split review into sentences    
def split_reviews(review_item):
    if pd.isnull(review_item):
        return np.nan
    else:
        return tokenizer.tokenize(review_item)

# convert part-of-speech tags for lemmatization
def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:                    
        return None

# define lemmatizer
lemmatizer = WordNetLemmatizer()

# lemmatize tokenized review sentence
def lemmatize_sentence(sentence):
    
    # create pos tags
    nltk_tagged = pos_tag(sentence)    
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    
    # lemmatize words based on their pos tag
    res_words = []
    for word,tag in wn_tagged:
        if tag is None:                        
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word,tag))
    
    return res_words

# load english stopwords
STOPWORDS = stopwords.words('english')

# create list of stopwords without punctuation (doesn't -> doesnt)
STOPWORDS_NO_PUNC = []
for stopword in STOPWORDS:
    stopword_no_punc = ''.join([char for char in stopword if char not in string.punctuation])
    if stopword_no_punc not in STOPWORDS:
        STOPWORDS_NO_PUNC.append(stopword_no_punc)

# master list of stopwords
STOPWORDS += STOPWORDS_NO_PUNC

# pre-process review sentence
def clean_sentence(review_sentence):
    
    # remove punctuation
    review_sentence_str = str(review_sentence)
    no_punc_sentence = [char if char not in string.punctuation else ' ' for char in review_sentence_str]
    no_punc_sentence = ''.join(no_punc_sentence)
    
    # lowercase and tokenize into words
    tokenized_text = word_tokenize(no_punc_sentence.lower())

    # remove stopwords
    cleaned_text = [token for token in tokenized_text if token not in STOPWORDS]
    
    # lemmatize tokenized text
    lemmatized_text = lemmatize_sentence(cleaned_text)
    
    return lemmatized_text
