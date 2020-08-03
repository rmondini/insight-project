#!/usr/bin/env python3.7

import pandas as pd
import numpy as np
import requests
import csv
import sys
from bs4 import BeautifulSoup

# define functions for scraping
def extract_review_body(review_query):
    if review_query != None:
        return review_query.find("span", itemprop="reviewBody").get_text()
    else:
        return None

def extract_review_staydate(review_query):
    if review_query != None:
        return review_query.get_text().strip('\n').strip('Stayed in ')
    else:
        return None
    
def remove_emojis(review_str):
    if review_str != None:
        return review_str.encode('ascii','ignore').decode('ascii')
    else:
        return review_str

def extract_hotel_stars(hotel_query):
    if hotel_query != None:
        if hotel_query.find(class_='invisible_spoken') != None:
            return hotel_query.find(class_='invisible_spoken').get_text().strip('\n')
        else:
            return None
    else:
        return None 
    
def extract_hotel_feature(hotel_query):
    if hotel_query != None:
        return hotel_query.get_text().strip('\n')
    else:
        return None    

def extract_hotel_category_list(hotel_query):
    if hotel_query != None:
        return hotel_query.find_all(class_='review_score_name')
    else:
        return None  

def extract_hotel_value_list(hotel_query):
    if hotel_query != None:
        return hotel_query.find_all(class_='review_score_value')
    else:
        return None
    
def build_rating_breakdown_list(cat_list,value_list):
    breakdown_list = []
    if cat_list != None:
        for idx in range(len(cat_list)):
            breakdown_list.append([cat_list[idx].get_text(),value_list[idx].get_text()])
    return breakdown_list

# define main function
def main():

    print('Running p0_booking_reviews_scraping.py...')

    # get user agent and dates for hotel room price scraping from input file
    with open(sys.argv[1]) as input_file:
        user_agent = input_file.readline().rstrip('\n')
        hotel_room_price_dates = [input_file.readline().rstrip('\n'),input_file.readline().rstrip('\n'),input_file.readline().rstrip('\n')]

    # define user agent for web scraping
    headers = {'User-Agent': user_agent}

    # get names and urls of hotels in NYS
    print('Getting list of hotels in New York State...')
    next_page=True
    ip=1
    hotel_results_offset=0
    hotel_info_list = []
    while next_page:    
        url= 'http://www.booking.com/reviews/region/new-york-state.html?offset='+str(hotel_results_offset)+'&'
        response = requests.get(url,headers=headers)
        hotel_info = BeautifulSoup(response.text, 'html.parser')
        hotel_info_page = hotel_info.find_all(class_=['rlp-main-hotel','li'])
        hotel_info_list.extend(hotel_info_page)
        ip+=1
        hotel_results_offset+=30
        if hotel_info_page==[]:
            next_page=False

    nys_hotel_info_list_csvname = './input/nys_hotel_info_list.csv'

    with open(nys_hotel_info_list_csvname, 'w', newline='', encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerow(['hotel_name','hotel_url'])
    
        for hotel_info_item in hotel_info_list:   
            hotel_name = hotel_info_item.find(class_ = 'rlp-main-hotel__hotel-name-link').get_text()
            hotel_url = hotel_info_item.find(class_ = 'rlp-main-hotel__hotel-name-link')['href'].split('.html?')[0]
            writer.writerow([hotel_name,hotel_url])        
    print('done')

    # get reviews of hotels in NYS
    print('Getting reviews for all the hotels in New York State')
    nys_hotel_info_list_csvname = './input/nys_hotel_info_list.csv'
    df_nys_hotel_info_list = pd.read_csv(nys_hotel_info_list_csvname)

    nys_hotel_reviews_list_csvname = './input/sample0_nys_hotel_reviews_list.csv'

    with open(nys_hotel_reviews_list_csvname, 'w', newline='', encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerow(['hotel_name','hotel_url','review_date','review_item_user_review_count','review_score_badge',
        	             'review_item_header_content','review_info_tag','review_staydate','review_pos','review_neg'])
    
        for idx,hotel_url_item in enumerate(df_nys_hotel_info_list['hotel_url']):

            next_page=True
            ip=1
            hotel_reviews_list = []
            while next_page:
                url= 'http://www.booking.com/reviews/us/hotel/'+hotel_url_item.split('/hotel/us/')[1]+'.html?page='+str(ip)+';r_lang=en'
                response = requests.get(url,headers=headers)
                hotel_reviews = BeautifulSoup(response.text, 'html.parser')
                hotel_reviews_page = hotel_reviews.find_all(class_=['review_item','li'])
                hotel_reviews_list.extend(hotel_reviews_page)
                ip+=1
                if hotel_reviews_page==[]:
                    next_page=False
            
            for hotel_reviews_item in hotel_reviews_list:   
                review_date = hotel_reviews_item.find(class_ = 'review_item_date').get_text().strip('\n').strip('Reviewed: ')
                review_item_user_review_count = hotel_reviews_item.find(class_ = 'review_item_user_review_count').get_text().strip('\n')
                review_score_badge = hotel_reviews_item.find(class_ = 'review-score-badge').get_text().strip('\"')
                review_item_header_content = hotel_reviews_item.find(class_ = 'review_item_header_content').get_text().strip('\n').replace('\u201c','').replace('\u201d','')
                review_info_tag = [tag.get_text().strip('\n')[2:] for tag in hotel_reviews_item.find_all(class_=['review_info_tag','li'])]
                review_staydate = extract_review_staydate(hotel_reviews_item.find(class_ = 'review_staydate'))
                review_pos = remove_emojis(extract_review_body(hotel_reviews_item.find(class_ = 'review_pos')))
                review_neg = remove_emojis(extract_review_body(hotel_reviews_item.find(class_ = 'review_neg')))
        
                writer.writerow([df_nys_hotel_info_list['hotel_name'][idx],hotel_url_item.split('/hotel/us/')[1],review_date,
                	             review_item_user_review_count,review_score_badge,review_item_header_content,review_info_tag,review_staydate,review_pos,review_neg])
    print('done')

    # get detailed info of hotels in NYS (stars, location, room price, etc)
    # (obtaining room price info on THREE different days and saving that info into three different files)
    print('Getting detailed hotel info and room prices...')
    for n_sample in range(3):

        nys_hotel_detailed_info_list_csvname = './input/sample' + str(n_sample) + '_nys_hotel_detailed_info_list.csv'
    
        with open(nys_hotel_detailed_info_list_csvname, 'w', newline='', encoding="UTF-8") as f:
            writer = csv.writer(f)
            writer.writerow(['hotel_name','hotel_url','hotel_stars','hotel_address','hotel_overall_rating','hotel_rating_breakdown','hotel_room_name','hotel_room_capacity','hotel_room_price'])
    
            for idx,hotel_url_item in enumerate(df_nys_hotel_info_list['hotel_url']):
        
                url= 'http://www.booking.com'+ hotel_url_item +'.html?' + hotel_room_price_dates[n_sample]
                response = requests.get(url,headers=headers)
                hotel_info = BeautifulSoup(response.text, 'html.parser')
        
                hotel_stars = extract_hotel_stars(hotel_info.find(class_='hp__hotel_ratings'))
                hotel_address = extract_hotel_feature(hotel_info.find(class_='hp_address_subtitle js-hp_address_subtitle jq_tooltip'))
                hotel_room_name = extract_hotel_feature(hotel_info.find(class_='hprt-roomtype-icon-link'))
                hotel_room_capacity = extract_hotel_feature(hotel_info.find(class_='c-occupancy-icons hprt-occupancy-occupancy-info'))
                hotel_room_price = extract_hotel_feature(hotel_info.find(class_='bui-price-display__value prco-text-nowrap-helper prco-font16-helper'))
        
                url= 'http://www.booking.com/reviews/us/hotel/'+hotel_url_item.split('/hotel/us/')[1]+'.html'
                response = requests.get(url,headers=headers)
                hotel_info = BeautifulSoup(response.text, 'html.parser')
        
                hotel_overall_rating = extract_hotel_feature(hotel_info.find(class_='review-score-badge'))
                hotel_rating_breakdown_category_list = extract_hotel_category_list(hotel_info.find(class_='review_score_breakdown_list'))
                hotel_rating_breakdown_value_list = extract_hotel_value_list(hotel_info.find(class_='review_score_breakdown_list'))
                hotel_rating_breakdown = build_rating_breakdown_list(hotel_rating_breakdown_category_list,hotel_rating_breakdown_value_list)
        
                writer.writerow([df_nys_hotel_info_list['hotel_name'][idx],hotel_url_item.split('/hotel/us/')[1],hotel_stars,hotel_address,
                	             hotel_overall_rating,hotel_rating_breakdown,hotel_room_name,hotel_room_capacity,hotel_room_price])
    print('done')

    return

if __name__ == "__main__":
    main()
