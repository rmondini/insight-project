# Hotel Review Analyzer  

### PROBLEM STATEMENT

When searching for hotels, travelers weigh negative reviews more than all other factors (including price) and it is estimated that hotels can lose as much as 20% of bookings due to negative reviews. Hotel managers need to quickly identify the topics negatively discussed in reviews and address those issues. Reducing the number of negative reviews in turn allows managers to raise room prices.  

### SOLUTION

I built an interactive web app that helps hotel managers  
1. Quantify how much each specific topic is affecting room prices
2. Predict by how much room prices can be increased in the absence of negative reviews.  

The web app is hosted on AWS and can be found at [www.hotelreviewanalyzer.xyz](http://www.hotelreviewanalyzer.xyz:8050).

### SLIDE DECK

A slide deck about this project can be found [here](https://docs.google.com/presentation/d/1T-WfnKIgH7UeZImbGfyOjM70DOYIH3plCQ5ABvyBGtE/edit?usp=sharing).

### REPOSITORY STRUCTURE

The directory structure of this repository is the following:

    ├── README.md
    ├── run.sh
    ├── input
    │   └── ...
    ├── input_app
    │   └── ...
    ├── output
    │   └── ...
    ├── src
    │   └── p0_booking_reviews_scraping.py
    │   └── p1_reviews_preprocessing.py
    │   └── reviews_preprocessing_util.py
    │   └── p2_reviews_topic_extraction.py
    │   └── p3_price_prediction.py
    │   └── price_prediction_util.py
    ├── src_app
    │   └── local_app.py

The code is written in Python and organized as follows.  

In the `src` folder:
- `p0_booking_reviews_scraping.py` contains the code needed to scrape reviews and detailed information (name, location, stars, ratings, room price) for all the hotels in the state of New York from booking.com.
- `p1_reviews_preprocessing.py` performs the initial text cleaning and pre-processing on the review dataset.
- `p2_reviews_topic_extraction.py` performs topic extraction on the negative reviews using soft cosine similarity between review sentences and an identified set of topics.
- `p3_price_prediction.py` is where I built a linear regression model that predicts the room price for each hotel using engineered topic features and hotel attributes such as location, stars, and ratings.
- `reviews_preprocessing_util.py` and `price_prediction_util.py` contain the definition of modular functions needed by the other files.  

In the `src_app` folder:
- `local_app.py` builds a local version of the interactive web app using Plotly and Dash. 

### HOW TO RUN THE CODE

The code uses the following external libraries: `Beautiful Soup, NumPy, Pandas, NLTK, Gensim, Scikit-Learn, Plotly, Dash`, which therefore need to be locally installed before running the code. The code can then be run with the command
``` 
insight-project$ ./run.sh
```
after modifying the first line of `./input/run_p0_params.txt` with the appropriate web user agent (for web scraping). The user can also modify the three dates used to sample hotel room prices by changing lines 2-4 of `./input/run_p0_params.txt` as desired. Finally, since scraping all the reviews and hotel information from booking.com's website takes a few hours in total, in order to save the user time the hotel room price and review datasets generated by `p0_booking_reviews_scraping.py` have already been included in the `input` folder (so that the user can comment out the first line in `run.sh` and avoid running `p0_booking_reviews_scraping.py`).
  
A local version of the web app can be built with  
```
insight-project$ python ./src_app/local_app.py
``` 
