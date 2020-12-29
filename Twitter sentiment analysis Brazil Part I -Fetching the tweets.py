# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:46:04 2019

@author: ALVAROALTAMIRANO
"""

import pandas as pd
import tweepy
import jsonpickle
# Consume:
#consumer key, consumer secret, access token, access secret.
ckey="YOUR TWITTER KEY HERE"
csecret="YOUR TWITTER KEY HERE"
atoken="YOUR TWITTER KEY HERE"
asecret="YOUR TWITTER KEY HERE"

# # Setup access API
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api
 
# #Dates for fetching
start_date = '2019-7-9'
# # Create API object
api = connect_to_twitter_OAuth()  
def get_save_tweets(INSS, api, query, max_tweets=200000, lang='pt'):

    tweetCount = 0

    #Open file and save tweets
    with open(INSS, 'w') as f:

        # Send the query
        for tweet in tweepy.Cursor(api.search,q=query,lang=lang, since=start_date).items(max_tweets):         
            #Convert to JSON format
            f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
            tweetCount += 1

        #Display how many tweets we have collected
        print("Downloaded {0} tweets".format(tweetCount))
query = '#Previdencia OR #Aposentadoria OR #Previdência OR #Aposentados OR Previdencia OR Previdência \
        OR Aposentados OR Aposentadoria OR previdencia OR previdência OR reformaprevidencia OR #INSS \
        OR #reformadaprevidência OR #reformaprevidencia OR #previdenciasocial -Filter:retweets'                        
      
# # Get those tweets
get_save_tweets('tweetsF.json', api, query)
# =============================================================================
def tweets_to_df(path):
    
    tweets = list(open('tweetsF.json', 'rt'))
    
    text = []
    weekday = []
    month = []
    day = []
    hour = []
    hashtag = []
    url = []
    favorite = []
    reply = []
    retweet = []
    follower = []
    location = []
    country_code = []
    coordinates = []

    for t in tweets:
        t = jsonpickle.decode(t)
        
        # Text
        text.append(t['text'])
        
        # Decompose date
        date = t['created_at']
        weekday.append(date.split(' ')[0])
        month.append(date.split(' ')[1])
        day.append(date.split(' ')[2])
        time = date.split(' ')[3].split(':')
        hour.append(time[0]) 
        # Has hashtag
        if len(t['entities']['hashtags']) == 0:
            hashtag.append(0)
        else:
            hashtag.append(1)
        # Has url
        if len(t['entities']['urls']) == 0:
            url.append(0)
        else:
            url.append(1)
        # Number of favs
        favorite.append(t['favorite_count'])
        # Is reply?
        if t['in_reply_to_status_id'] == None:
            reply.append(0)
        else:
            reply.append(1)       
        # Retweets count
        retweet.append(t['retweet_count'])
        # Followers number
        follower.append(t['user']['followers_count'])
        # Following number
        location.append(t['user']['location'])
        # Add country
        try:
            country_code.append(t['place']['country_code'])
        except Exception as e:
            #print(e)
            country_code.append(None)
        except tweepy.TweepError:  
            time.sleep(60)
        # Add screen name
        try:
            coordinates.append(t['place']['coordinates'])
        except Exception as e:
            #print(e)
            coordinates.append(None)
        except tweepy.TweepError:  
            time.sleep(60)
  
    d = {'text': text,
         'weekday': weekday,
         'month' : month,
         'day': day,
         'hour' : hour,
         'has_hashtag': hashtag,
         'has_url': url,
         'fav_count': favorite,
         'is_reply': reply,
         'retweet_count': retweet,
         'followers': follower,
         'location' : location,
         'country': country_code,
         'coordinates' : coordinates
        }
    
    return pd.DataFrame(data = d)

tweets_df = tweets_to_df('tweetsF.json')
tweets_df.columns
#Saving dataframe to csv file
tweets_df.to_csv(r"C:\\tweets_df.csv")
