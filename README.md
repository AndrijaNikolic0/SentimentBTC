# SentimentBTC
Based on cryptocurrency prices, and sentiment value(taken from Reddit, using vader from nltk library)from last 6 months, this program forecasts values from test dataset, using Random Forest model.
First, we take crypto prices from prices.py, and place them into final.csv.
After that, we make sentiment analysis on bitcoin posts on reddit,per every hour, and store that into another csv file, vader.csv.
Lastly, we unite final.csv and vader.csv into another csv file, along with average, and moving average columns, so we can predict future price trend.
We place all those information into sentimentIBtcFinalno.csv.
We use basic RandomForest model on previously mentioned csv file.
