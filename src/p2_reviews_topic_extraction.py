import pandas as pd
import numpy as np
import string
import gensim.downloader as api
from nltk import pos_tag,word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from gensim import models, corpora
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from sklearn.metrics import classification_report

# import auxiliary functions
from reviews_preprocessing_util import *

# define main function
def main():

    # load pre-trained word-embedding model
    print('Loading pre-trained FastText model. Might take a few minutes...')
    fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    print('done')

    # import negative sentences dataset
    df_negative_sentences = pd.read_csv('./output/df_negative_sentences.csv')
    df_negative_sentences.rename({'Unnamed: 0':'unique_id'},axis=1,inplace=True)

    # pre-process negative sentences
    print('Pre-processing review sentences. Might take a few minutes...')
    df_negative_sentences['review_sentence_cleaned'] = df_negative_sentences['review_neg_sentence'].apply(clean_sentence)
    print('done')

    # remove empty reviews after pre-processing
    df_negative_sentences = df_negative_sentences[df_negative_sentences['review_sentence_cleaned'].apply(lambda x:len(x))>0]

    # create 'review_topic' column and set it to NaN
    df_negative_sentences['review_topic'] = np.nan

    # identify set of topics and topic keywords
    room_comfort_topic = 'Noise noisy loud slam voice music thin wall hear' + ' Smell smelly odor stink' + ' Ac heat hot cold thermostat air vent ventilation fan adjust heater temperature'
    staff_topic = 'Staff rude unfriendly friendly polite front desk manager maid reception valet clerk housekeep housekeeping waiter waitress unprofessional' + ' Check communication experience bag early late reservation booking'
    breakfast_topic = 'Breakfast food egg bacon sausage toast potato waffle fruit omelette omelet cheese milk pastry coffee tea juice silverware plasticware cup plastic included selection taste fresh'
    facilities_topic = 'Facility elevator lift stair wheelchair pool jacuzzi gym vending spa sauna renovation bar restaurant lounge pet property' + ' WiFi internet connection' + ' Park lot car valet street'
    location_topic = 'Location surrounding far traffic highway walk street road neighborhood sketchy attraction center city town downtown nearby near walk transport subway park view safe dangerous drive'
    bathroom_topic = 'Bathroom shower tub bathtub drain curtain pressure sink water toiletry toilet mirror shampoo conditioner towel soap ply paper hair face wash vent ventilation fan window'
    room_amenities_topic = 'Room carpet curtain shade drape wardrobe outlet plug window tv balcony couch remote wall fridge refrigerator safe machine coffee tea kettle amenity microwave card door'
    bed_quality_topic = 'Bed thin sheet linen blanket cover comforter pillow hard soft firm mattress bug bedbug king double queen twin frame sleep pullout couch bunk comfortable flat'

    topics= [room_comfort_topic,staff_topic,breakfast_topic,facilities_topic,location_topic,bathroom_topic,room_amenities_topic,bed_quality_topic]
    n_topics = len(topics)

    # pre-process topic keyword lists
    topics_cleaned = list(map(clean_sentence,topics))

    # build complete dictionary
    tokenized_neg_reviews_and_topics = topics_cleaned + list(df_negative_sentences['review_sentence_cleaned'])
    neg_dictionary = corpora.Dictionary(tokenized_neg_reviews_and_topics)

    # create bag-of-words vectors
    corpus_neg_reviews = [neg_dictionary.doc2bow(text) for text in list(df_negative_sentences['review_sentence_cleaned'])]
    corpus_neg_topics = [neg_dictionary.doc2bow(text) for text in topics_cleaned]

    # build similarity matrix of word embeddings
    print('Building similarity matrix of word embeddings. Might take a few minutes...')
    termsim_index = WordEmbeddingSimilarityIndex(fasttext_model300)
    similarity_matrix = SparseTermSimilarityMatrix(termsim_index,neg_dictionary)
    print('done')

    # compute soft cosine similarity between sentences and topics
    print('Computing soft cosine similarity between sentences and topics. Might take a few minutes...')
    neg_data_topics = []
    for review_item in corpus_neg_reviews:
        review_item_topics = []
        for topic in corpus_neg_topics:
            review_item_topics.append(similarity_matrix.inner_product(review_item,topic,normalized=True))
        neg_data_topics.append(review_item_topics)
    print('done')    

    # extract topic with highest soft cosine similarity
    # I set a minimum threshold (0.10) that needs to be reached in order to assign a topic.
    # If above-threshold topics are within 0.01, I assign -1 (i.e. no main topic)  
    neg_data_closest_topic = []
    cossim_threshold = 0.10

    for review_item_topic_list in neg_data_topics:
        if max(review_item_topic_list)>cossim_threshold:
            review_item_array = np.array(review_item_topic_list)
            sorted_review_item_array = sorted(review_item_array,reverse=True)
            num_topics=1
            for item in sorted_review_item_array[1:]:
                if abs(item-sorted_review_item_array[0])<0.01:
                    num_topics+=1
            if num_topics==1:
                neg_data_closest_topic.append(np.argmax(review_item_topic_list))
            else:
                neg_data_closest_topic.append(-1)
        else:
            neg_data_closest_topic.append(-1)  

    # assign extracted topic   
    df_negative_sentences['review_topic'] = neg_data_closest_topic

    # construct pivot table of hotels (index), topics (columns), and topic counts (values)
    df_negative_sentences_by_topic = df_negative_sentences.groupby(['hotel_url','review_topic']).size().reset_index()
    df_negative_sentences_by_topic.rename({0:'review_topic_count'},axis=1,inplace=True)
    df_negative_sentences_by_topic_pt = df_negative_sentences_by_topic.pivot_table(values='review_topic_count',index='hotel_url',columns='review_topic').reset_index()
    df_negative_sentences_by_topic_pt.fillna(0,inplace=True)   

    # normalize each count by total number of negative sentences per hotel
    df_negative_sentences_count_by_hotel = df_negative_sentences.groupby('hotel_url').count().reset_index()[['hotel_url','review_topic']]
    df_negative_sentences_count_by_hotel.rename({'review_topic':'sentences_count'},axis=1,inplace=True)
    df_negative_sentences_by_topic_pt = df_negative_sentences_by_topic_pt.merge(df_negative_sentences_count_by_hotel,on='hotel_url')

    # create columns with normalized topic counts
    df_negative_sentences_by_topic_pt[[str(n)+'_pc' for n in range(-1,n_topics)]] = df_negative_sentences_by_topic_pt[[n for n in range(-1,n_topics)]].div(df_negative_sentences_by_topic_pt.sentences_count, axis=0)

    # write out pivot table to csv
    df_negative_sentences_by_topic_pt.to_csv('./output/df_negative_sentences_by_topic_count.csv',index=False)

    # validate topic extraction against manually-annotated dataset
    df_negative_sentences_annotated = pd.read_csv('./input/df_negative_sentences_annotated_topic.csv')

    # keep only relevant columns in annotated dataset
    df_negative_sentences_annotated = df_negative_sentences_annotated[['unique_id','review_date','review_topic_annotated']]

    # merge dataframes and keep only entries with manually-annotated topic
    df_negative_sentences_topic_validation = df_negative_sentences.merge(df_negative_sentences_annotated,on=['review_date','unique_id'])
    df_negative_sentences_topic_validation = df_negative_sentences_topic_validation[~pd.isnull(df_negative_sentences_topic_validation['review_topic_annotated'])]
    df_negative_sentences_topic_validation['review_topic_annotated'] = df_negative_sentences_topic_validation['review_topic_annotated'].apply(lambda x:int(x))

    # write out classification report on manually-annotated dataset to file
    valid_report_file = open("./output/topic_extraction_validation_report.txt", "w")
    valid_report_file.write(classification_report(df_negative_sentences_topic_validation['review_topic_annotated'],df_negative_sentences_topic_validation['review_topic']))
    valid_report_file.close()

    return

if __name__ == "__main__":
    main()
