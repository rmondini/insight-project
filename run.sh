#!/bin/bash

python ./src/p0_booking_reviews_scraping.py ./input/run_p0_params.txt
python ./src/p1_reviews_preprocessing.py
python ./src/p2_reviews_topic_extraction.py
python ./src/p3_price_prediction.py
