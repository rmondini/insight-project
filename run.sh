#!/bin/bash

python ./src/p0_booking_reviews_scraping.py
python ./src/p1_reviews_preprocessing.py
python ./src/p2_reviews_topic_extraction.py
python ./src/p3_price_prediction.py
