import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# import auxiliary functions
from price_prediction_util import *

# define main function
def main():

    # import dataset of detailed info (stars, ratings, location) for each hotel + room price on 3 different days
    df_hotel_detailed_info_list = pd.read_csv('./input/sample0_nys_hotel_detailed_info_list.csv')
    df_hotel_detailed_info_list_2 = pd.read_csv('./input/sample1_nys_hotel_detailed_info_list.csv')
    df_hotel_detailed_info_list_3 = pd.read_csv('./input/sample2_nys_hotel_detailed_info_list.csv')

    # import dataset of number of negative mentions per topic for each hotel
    df_negative_sentences_by_topic_pt = pd.read_csv('./output/df_negative_sentences_by_topic_count.csv')

    # rename room price, room name, and room capacity columns
    df_hotel_detailed_info_list.rename(columns={'hotel_room_price': 'hotel_room_price'},inplace=True)
    df_hotel_detailed_info_list_2.rename(columns={'hotel_room_price': 'hotel_room_price_2','hotel_room_name':'hotel_room_name_2','hotel_room_capacity':'hotel_room_capacity_2'},inplace=True)
    df_hotel_detailed_info_list_3.rename(columns={'hotel_room_price': 'hotel_room_price_3','hotel_room_name':'hotel_room_name_3','hotel_room_capacity':'hotel_room_capacity_3'},inplace=True)

    # extract zip codes in 5-digit format
    df_hotel_detailed_info_list = df_hotel_detailed_info_list[~pd.isnull(df_hotel_detailed_info_list['hotel_address'])]
    df_hotel_detailed_info_list['hotel_address_zip_code_5dig'] = df_hotel_detailed_info_list['hotel_address'].apply(extract_zip_code_5dig)

    # extract room prices
    df_hotel_detailed_info_list['hotel_room_price'] = df_hotel_detailed_info_list['hotel_room_price'].apply(extract_price)
    df_hotel_detailed_info_list_2['hotel_room_price_2'] = df_hotel_detailed_info_list_2['hotel_room_price_2'].apply(extract_price)
    df_hotel_detailed_info_list_3['hotel_room_price_3'] = df_hotel_detailed_info_list_3['hotel_room_price_3'].apply(extract_price)

    # merge dataframes with room prices 2 and 3 onto main dataframe
    df_hotel_detailed_info_list = df_hotel_detailed_info_list.merge(df_hotel_detailed_info_list_2[['hotel_url','hotel_room_price_2','hotel_room_name_2','hotel_room_capacity_2']],on='hotel_url')
    df_hotel_detailed_info_list = df_hotel_detailed_info_list.merge(df_hotel_detailed_info_list_3[['hotel_url','hotel_room_price_3','hotel_room_name_3','hotel_room_capacity_3']],on='hotel_url')

    # extract room capacities
    df_hotel_detailed_info_list['hotel_room_capacity'] = df_hotel_detailed_info_list['hotel_room_capacity'].apply(extract_capacity)
    df_hotel_detailed_info_list['hotel_room_capacity_2'] = df_hotel_detailed_info_list['hotel_room_capacity_2'].apply(extract_capacity)
    df_hotel_detailed_info_list['hotel_room_capacity_3'] = df_hotel_detailed_info_list['hotel_room_capacity_3'].apply(extract_capacity)

    # determine room prices per person
    df_hotel_detailed_info_list['hotel_room_price_per_person'] = df_hotel_detailed_info_list['hotel_room_price']/df_hotel_detailed_info_list['hotel_room_capacity']
    df_hotel_detailed_info_list['hotel_room_price_per_person_2'] = df_hotel_detailed_info_list['hotel_room_price_2']/df_hotel_detailed_info_list['hotel_room_capacity_2']
    df_hotel_detailed_info_list['hotel_room_price_per_person_3'] = df_hotel_detailed_info_list['hotel_room_price_3']/df_hotel_detailed_info_list['hotel_room_capacity_3']

    # remove hotels without room prices
    df_hotel_detailed_info_list = df_hotel_detailed_info_list[~pd.isnull(df_hotel_detailed_info_list[['hotel_room_price_per_person','hotel_room_price_per_person_2','hotel_room_price_per_person_3']]).any(axis=1)]

    # determine average room price per person for each hotel
    df_hotel_detailed_info_list['hotel_room_price_per_person_avg'] = df_hotel_detailed_info_list[['hotel_room_price_per_person','hotel_room_price_per_person_2','hotel_room_price_per_person_3']].mean(axis=1)

    # extract aspect ratings
    df_hotel_detailed_info_list['hotel_cleanliness_rating'] = df_hotel_detailed_info_list['hotel_rating_breakdown'].apply(extract_cleanliness_rating)
    df_hotel_detailed_info_list['hotel_comfort_rating'] = df_hotel_detailed_info_list['hotel_rating_breakdown'].apply(extract_comfort_rating)
    df_hotel_detailed_info_list['hotel_location_rating'] = df_hotel_detailed_info_list['hotel_rating_breakdown'].apply(extract_location_rating)
    df_hotel_detailed_info_list['hotel_facilities_rating'] = df_hotel_detailed_info_list['hotel_rating_breakdown'].apply(extract_facilities_rating)
    df_hotel_detailed_info_list['hotel_staff_rating'] = df_hotel_detailed_info_list['hotel_rating_breakdown'].apply(extract_staff_rating)
    df_hotel_detailed_info_list['hotel_value_rating'] = df_hotel_detailed_info_list['hotel_rating_breakdown'].apply(extract_value_rating)
    df_hotel_detailed_info_list['hotel_wifi_rating'] = df_hotel_detailed_info_list['hotel_rating_breakdown'].apply(extract_wifi_rating)

    # median imputation for wifi rating
    # (since there are only a few missing values)
    hotel_wifi_median = df_hotel_detailed_info_list['hotel_wifi_rating'].median()
    df_hotel_detailed_info_list['hotel_wifi_rating'].fillna(hotel_wifi_median,inplace=True)

    # extract hotel stars
    df_hotel_detailed_info_list['hotel_stars'] = df_hotel_detailed_info_list['hotel_stars'].apply(extract_hotel_stars)

    # remove hotels without stars info
    # (since these correspond to non-hotel accommodations such as airbnbs or private apartments)
    df_hotel_detailed_info_list = df_hotel_detailed_info_list[~pd.isnull(df_hotel_detailed_info_list['hotel_stars'])]

    # determine average room price and its standard deviation by stars and location
    df_hotel_detailed_info_list['hotel_room_price_per_person_avg_grouped_mean'] = df_hotel_detailed_info_list.groupby(['hotel_stars','hotel_address_zip_code_5dig']).transform('mean')['hotel_room_price_per_person_avg']
    df_hotel_detailed_info_list['hotel_room_price_per_person_avg_grouped_std'] = df_hotel_detailed_info_list.groupby(['hotel_stars','hotel_address_zip_code_5dig']).transform('std')['hotel_room_price_per_person_avg']

    # remove isolated hotels (i.e. no std by stars and location)
    # (since it means that they have little competition and can charge whatever price they want)
    df_hotel_detailed_info_list = df_hotel_detailed_info_list[~pd.isnull(df_hotel_detailed_info_list['hotel_room_price_per_person_avg_grouped_std'])]

    # determine percent change in room price for each hotel compared to average by stars and location
    df_hotel_detailed_info_list['hotel_room_price_per_person_pc_change'] = df_hotel_detailed_info_list['hotel_room_price_per_person_avg']-df_hotel_detailed_info_list['hotel_room_price_per_person_avg_grouped_mean']
    df_hotel_detailed_info_list['hotel_room_price_per_person_pc_change'] = df_hotel_detailed_info_list['hotel_room_price_per_person_pc_change'].div(df_hotel_detailed_info_list['hotel_room_price_per_person_avg'], axis=0)

    # remove outliers
    # (defined as hotels charging room price more than 30% away from the average by stars and location)
    df_hotel_detailed_info_list = df_hotel_detailed_info_list[abs(df_hotel_detailed_info_list['hotel_room_price_per_person_pc_change'])<0.30]

    # merge dataframe with topic count dataframe
    neg_columns_to_rename = {'-1':'-1_neg','0':'0_neg','1':'1_neg','2':'2_neg','3':'3_neg','4':'4_neg','5':'5_neg','6':'6_neg','7':'7_neg',
                             '-1_pc':'-1_pc_neg','0_pc':'0_pc_neg','1_pc':'1_pc_neg','2_pc':'2_pc_neg','3_pc':'3_pc_neg','4_pc':'4_pc_neg','5_pc':'5_pc_neg','6_pc':'6_pc_neg','7_pc':'7_pc_neg',
                             'sentences_count':'sentences_count_neg'}
    df_negative_sentences_by_topic_pt.rename(columns=neg_columns_to_rename,inplace=True)
    df_hotel_detailed_info_list_merged = df_hotel_detailed_info_list.merge(df_negative_sentences_by_topic_pt,on='hotel_url')

    # select topic count features to include in feature set
    # (and make them explicit percentages)
    topic_features_list = ['0_pc_neg','1_pc_neg','2_pc_neg','3_pc_neg','4_pc_neg','5_pc_neg','6_pc_neg','7_pc_neg']
    df_hotel_detailed_info_list_merged[topic_features_list] = 100*df_hotel_detailed_info_list_merged[topic_features_list]

    # write out dataset with unique hotel names (hotel name + zip code) for web app
    df_hotels_unique_names = df_hotel_detailed_info_list_merged.copy()
    df_hotels_unique_names['hotel_unique_name'] = df_hotels_unique_names['hotel_name'] + ' (' + df_hotels_unique_names['hotel_address_zip_code_5dig'] + ')'
    df_hotels_unique_names.to_csv('./input_app/df_hotels.csv',index=False)

    # create complete feature set and target for modeling
    # using backward selection, the only statistically-significant features are:
    # hotel_stars, hotel_value_rating, hotel_comfort_rating (using t-tests)
    features_list1 = ['hotel_stars','hotel_value_rating','hotel_comfort_rating']
    # location (zip code) is a categorical feature and I decide to keep all levels   
    features_list2 = ['hotel_address_zip_code_5dig']
    # I keep all seven topic features to quantify their weight in price prediction
    features_complete_list = features_list1+features_list2+topic_features_list

    df_features = df_hotel_detailed_info_list_merged[features_complete_list]
    df_target = df_hotel_detailed_info_list_merged[['hotel_room_price_per_person_avg']]

    # one-hot encoding for categorical features
    categorical_features = ['hotel_address_zip_code_5dig']
    df_features_one_hot_encoded = df_features.copy()
    df_features_one_hot_encoded = pd.get_dummies(df_features_one_hot_encoded,columns=categorical_features,drop_first=True)

    # train linear regression model
    reg = LinearRegression()
    reg.fit(df_features_one_hot_encoded,df_target)

    # pickle model
    regression_model_filename = './input_app/price_regression_model.sav'
    pickle.dump(reg, open(regression_model_filename, 'wb'))

    # pickle additional parameters
    aux_info_1_filename = './input_app/categorical_features.sav'
    pickle.dump(categorical_features, open(aux_info_1_filename, 'wb'))
    aux_info_2_filename = './input_app/features_complete_list.sav'
    pickle.dump(features_complete_list, open(aux_info_2_filename, 'wb'))
    aux_info_3_filename = './input_app/features_complete_list_dummies.sav'
    pickle.dump(df_features_one_hot_encoded.columns.values, open(aux_info_3_filename, 'wb'))

    return

if __name__ == "__main__":
    main()
