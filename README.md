# Hotel Review Analyzer  

### PROBLEM

When searching for hotels, travelers weigh negative reviews more than all other factors (including price) and it is estimated that hotels can lose as much as 20% of bookings due to negative reviews. Hotels managers need to quickly identify the topics negatively discussed in reviews and address those issues.  

### SOLUTION

I built an interactive web app that helps hotel managers  
1. Quantify how much each specific topic is affecting room prices
2. Predict by how much room prices can be increased in the absence of negative reviews.  

The web app can be found at [www.hotelreviewanalyzer.xyz](https://www.hotelreviewanalyzer.xyz).  

### CODE STRUCTURE

The code for this project is written in Python and organized in Jupyter notebooks. The structure is the following:  
- `booking-reviews-scraping.ipynb` contains the code needed to scrape reviews and detailed information (name, location, stars, ratings, room price) for all the hotels in the state of New York from booking.com.
- `reviews-preprocessing.ipynb` performs the initial text cleaning and pre-processing on the review dataset.
- `reviews-topic-extraction.ipynb` performs topic extraction on the negative reviews using soft cosine similarity between review sentences and an identified set of topics.
- `price-prediction.ipynb` is where I built a linear regression model that predicts the room price for each hotel based on the number of negative sentences per topic and hotel features such as location, stars, and ratings.
- the folder `input_app` contains files that are needed to build the interactive web app. More specifically, the `.sav` files contain the pickled regression model and other necessary parameters, while the `.csv` files contain the final hotel dataset used to train the regression model. All these files are generated in `price-prediction.ipynb`.
- `local_app.py` builds a local version of the interactive web app using Plotly and Dash (and the files contained in the `input_app` folder).



