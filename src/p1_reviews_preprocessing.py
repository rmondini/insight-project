import pandas as pd
import numpy as np
import csv
import string
from nltk import pos_tag,word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords,wordnet
import nltk.data

# import auxiliary functions
from reviews_preprocessing_util import *

# define main function
def main():

    # import complete dataset of all reviews
    df_complete  = pd.read_csv('./input/sample0_nys_hotel_reviews_list.csv')

    # replace '\n' and '\r' in review sentences and headers
    df_complete['review_pos'] = df_complete['review_pos'].apply(remove_new_line)
    df_complete['review_neg'] = df_complete['review_neg'].apply(remove_new_line)
    df_complete['review_item_header_content'] = df_complete['review_item_header_content'].apply(remove_new_line)

    # split reviews into sentences
    df_complete['review_pos_split']=df_complete['review_pos'].apply(split_reviews)
    df_complete['review_neg_split']=df_complete['review_neg'].apply(split_reviews)

    # create a new data point for each sentence
    df_complete_pos_col = df_complete.apply(lambda x: pd.Series(x['review_pos_split'], dtype='object'),axis=1).stack().reset_index(level=1, drop=True)
    df_complete_pos_col.name = 'review_pos_sentence'
    df_complete_pos = df_complete.copy()
    df_complete_pos.drop('review_pos_split', axis=1, inplace=True)
    df_complete_pos = df_complete_pos.join(df_complete_pos_col)

    df_complete_neg_col = df_complete.apply(lambda x: pd.Series(x['review_neg_split'], dtype='object'),axis=1).stack().reset_index(level=1, drop=True)
    df_complete_neg_col.name = 'review_neg_sentence'
    df_complete_neg = df_complete.copy()
    df_complete_neg.drop('review_neg_split', axis=1, inplace=True)
    df_complete_neg = df_complete_neg.join(df_complete_neg_col)

    # clean up dataframes by removing unneeded columns
    df_complete_pos.drop(['review_pos','review_neg','review_neg_split'], axis=1, inplace=True)
    df_complete_neg.drop(['review_pos','review_neg','review_pos_split'], axis=1, inplace=True)

    # remove reviews without text
    df_complete_pos = df_complete_pos[~pd.isnull(df_complete_pos['review_pos_sentence'])]
    df_complete_neg = df_complete_neg[~pd.isnull(df_complete_neg['review_neg_sentence'])]

    # write out positive and negative review sentences to separate csv files
    df_complete_pos.reset_index(inplace=True,drop=True)
    df_complete_pos.to_csv('./output/df_positive_sentences.csv')

    df_complete_neg.reset_index(inplace=True,drop=True)
    df_complete_neg.to_csv('./output/df_negative_sentences.csv')

    return

if __name__ == "__main__":
    main()