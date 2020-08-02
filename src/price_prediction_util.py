import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

# extract 5-digit zip code from hotel address
def extract_zip_code_5dig(hotel_address):
    address_tmp = hotel_address.split(',')[-2].strip(' ').split(' ')[-1][:5]
    if address_tmp == 'NY1':
        return hotel_address.split(',')[-2].strip(' ').split(' ')[-1][2:7]
    else:
        return address_tmp

# extract hotel room price    
def extract_price(hotel_room_price):
    room_price_tmp = str(hotel_room_price).strip('\r').strip('US$')
    if room_price_tmp == '':
        return None
    else:
        room_price_tmp = room_price_tmp.replace(',' , '')
        return float(room_price_tmp)

# extract hotel room capacity    
def extract_capacity(hotel_room_capacity):
    if pd.isnull(hotel_room_capacity):
        return None
    else:
        return int(hotel_room_capacity[-1])

# extract hotel stars       
def extract_hotel_stars(hotel_stars):
    if hotel_stars in ['1 stars', '1-star hotel']:
        return 1.0
    elif hotel_stars in ['1.5 stars', '1.5 star hotel']:
        return 1.5
    elif hotel_stars in ['2 stars', '2-star hotel']:
        return 2.0
    elif hotel_stars in ['2.5 stars', '2.5 star hotel']:
        return 2.5
    elif hotel_stars in ['3 stars', '3-star hotel']:
        return 3.0
    elif hotel_stars in ['3.5 stars', '3.5 star hotel']:
        return 3.5
    elif hotel_stars in ['4 stars', '4-star hotel']:
        return 4.0
    elif hotel_stars in ['4.5 stars', '4.5 star hotel']:
        return 4.5
    elif hotel_stars in ['5 stars', '5-star hotel']:
        return 5.0
    else:
        return None

# extract hotel cleanliness rating    
def extract_cleanliness_rating(hotel_rating_breakdown):
    item_tmp = hotel_rating_breakdown.strip('[]').split('], [')[0]
    if item_tmp.split(',')[0]=="'Cleanliness'":
        return float(item_tmp.split(',')[1].strip(' ').strip("''"))
    else:
        return None

# extract hotel comfort rating     
def extract_comfort_rating(hotel_rating_breakdown):
    item_tmp = hotel_rating_breakdown.strip('[]').split('], [')[1]
    if item_tmp.split(',')[0]=="'Comfort'":
        return float(item_tmp.split(',')[1].strip(' ').strip("''"))
    else:
        return None

# extract hotel location rating     
def extract_location_rating(hotel_rating_breakdown):
    item_tmp = hotel_rating_breakdown.strip('[]').split('], [')[2]
    if item_tmp.split(',')[0]=="'Location'":
        return float(item_tmp.split(',')[1].strip(' ').strip("''"))
    else:
        return None

# extract hotel facilities rating     
def extract_facilities_rating(hotel_rating_breakdown):
    item_tmp = hotel_rating_breakdown.strip('[]').split('], [')[3]
    if item_tmp.split(',')[0]=="'Facilities'":
        return float(item_tmp.split(',')[1].strip(' ').strip("''"))
    else:
        return None

# extract hotel staff rating     
def extract_staff_rating(hotel_rating_breakdown):
    item_tmp = hotel_rating_breakdown.strip('[]').split('], [')[4]
    if item_tmp.split(',')[0]=="'Staff'":
        return float(item_tmp.split(',')[1].strip(' ').strip("''"))
    else:
        return None

# extract hotel value for money rating     
def extract_value_rating(hotel_rating_breakdown):
    item_tmp = hotel_rating_breakdown.strip('[]').split('], [')[5]
    if item_tmp.split(',')[0]=="'Value for money'":
        return float(item_tmp.split(',')[1].strip(' ').strip("''"))
    else:
        return None

# extract hotel wifi quality rating     
def extract_wifi_rating(hotel_rating_breakdown):
    item_tmp = hotel_rating_breakdown.strip('[]').split('], [')
    if len(item_tmp)<7:
        return None
    elif item_tmp[6].split(',')[0]=="'Free WiFi'":
        return float(item_tmp[6].split(',')[1].strip(' ').strip("''"))
    else:
        return None