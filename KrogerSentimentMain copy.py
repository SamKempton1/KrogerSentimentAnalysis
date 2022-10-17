import requests
import pandas as pd
import flair
import re
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser


BEARER_TOKEN = ""

search_url = "https://api.twitter.com/2/tweets/search/recent"

searchParams = {'query': 'kroger',
                'expansions': 'referenced_tweets.id',
                'tweet.fields': 'created_at',
                'max_results': '100'
                }


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "KStockAnalysis"
    return r


def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def get_data(tweet):
    data = {
        'id': tweet['id'],
        'created_at': tweet['created_at'],
        'text': tweet['text']
    }
    return data


def cleanData(tweet):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    kroger = re.compile(r"(?i)@Kroger(?=\b)")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    retweet = re.compile("RT")

    tweet = whitespace.sub(' ', tweet)
    tweet = web_address.sub('', tweet)
    tweet = kroger.sub('Kroger', tweet)
    tweet = user.sub('', tweet)
    tweet = retweet.sub('', tweet)
    return tweet


def sentimentAnalysis(text):
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    sentence = flair.data.Sentence(text)
    prediction = sentiment_model.predict(sentence)
    return prediction


dtformat = '%Y-%m-%dT%H:%M:%SZ'



def time_travel(now, mins):
    now = datetime.strptime(now, dtformat)  # get the current datetime, this is our starting point
    back_in_time = now - timedelta(minutes=mins)
    return back_in_time.strftime(dtformat)  # convert now datetime to format for API

# now = datetime.now()  # get the current datetime, this is our starting point
# last_week = datetime.now() - timedelta(days=1)  # datetime one week ago = the finish line
# now = now.strftime(dtformat)  # convert now datetime to format for API


def main():
    now = datetime.now()  # get the current datetime, this is our starting point
    last_week = datetime.now() - timedelta(days=6)  # datetime one week ago = the finish line
    now = now.strftime(dtformat)  # convert now datetime to format for API

    df = pd.DataFrame()
    created_at = []
    sentiments = []
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')

    while True:
        # IF NOT LAST WEEK
        while datetime.strptime(now, dtformat) > last_week:

            pre60 = time_travel(now, 60)


            searchParams['start_time'] = pre60
            searchParams['end_time'] = now


            json_response = connect_to_endpoint(search_url, searchParams)
            # print("Got New Endpoint.")

            now = pre60  # move the window 60 minutes earlier

            #New Hourly Count + Sentiment
            tweet_count = 0
            hourly_sentiment = 0

            # For tweet in hour
            for tweet in json_response["data"]:
                row = get_data(tweet)

                # # Date translation
                # ds = row['created_at']
                # date = parser.parse(ds)

                # Find and Clean Text
                tweetText = row['text']
                cleanText = cleanData(tweetText)

                # Ensure no empty string
                if cleanText.strip() == "":
                   print("Empty String!")
                else:
                    sentence = flair.data.Sentence(cleanText)
                    prediction = sentiment_model.predict(sentence)

                    if sentence.labels[0].value == 'POSITIVE':
                        hourly_sentiment += 1 * sentence.labels[0].score

                    elif sentence.labels[0].value == 'NEGATIVE':
                        hourly_sentiment += -1 * sentence.labels[0].score

                    tweet_count += 1


            # Find avg hour sentiment
            avgHourlySentiment = hourly_sentiment / tweet_count


            # Find the hour index and append to df
            ds = now
            date = parser.parse(ds)

            created_at.append(date)
            sentiments.append(avgHourlySentiment)
            now = pre60

        df['created_at'] = created_at
        df['sentiment'] = sentiments

        sns.lineplot(x=df['created_at'], y=df['sentiment'], markers='o')

        plt.ylim(-1, 1)
        plt.xlabel('Hour')
        plt.ylabel('Sentiment')
        plt.title("Kroger Sentiment")
        plt.style.use('dark_background')
        plt.show()


if __name__ == "__main__":
    main()


