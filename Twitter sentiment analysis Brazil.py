# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 23:26:36 2019

@author: ALVAROALT
# =============================================================================
# """
# import pandas as pd
# import tweepy
# import jsonpickle
# #pip install jsonpickle
# # Consume:
# #consumer key, consumer secret, access token, access secret.
# ckey="rCLMeeoamIQqJSKzNPJBgmtT5"
# csecret="ETMpITdQXRNGa95sDu9MfVykFAlkEojhuQGmo1VygNbcB7iupK"
# atoken="99835846-kKdOkYRkZJ9bameQou9LWl6HZ0mDdnQHy3GHAeQBu"
# asecret="43Nyv7BMRInwQILxVAJ19D6oAAoHzKuhTCeOR9X3Rk6p7"
# 
# # Setup access API
# def connect_to_twitter_OAuth():
#     auth = tweepy.OAuthHandler(ckey, csecret)
#     auth.set_access_token(atoken, asecret)
#     
#     api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
#     return api
#  
# #Datas do fetch
# start_date = '2019-7-9'
# # Create API object
# api = connect_to_twitter_OAuth()  
# def get_save_tweets(INSS, api, query, max_tweets=200000, lang='pt'):
# 
#     tweetCount = 0
# 
#     #Open file and save tweets
#     
#     with open(INSS, 'w') as f:
# 
#         # Send the query
#         for tweet in tweepy.Cursor(api.search,q=query,lang=lang, since=start_date).items(max_tweets):         
# 
#             #Convert to JSON format
#             f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
#             tweetCount += 1
# 
#         #Display how many tweets we have collected
#         print("Downloaded {0} tweets".format(tweetCount))
# query = '#Previdencia OR #Aposentadoria OR #Previdência OR #Aposentados OR Previdencia OR Previdência OR Aposentados OR Aposentadoria OR previdencia OR previdência OR reformaprevidencia OR #INSS OR #reformadaprevidência OR #reformaprevidencia OR #previdenciasocial -Filter:retweets'                        
#       
# # Get those tweets
# get_save_tweets('tweetsF.json', api, query)
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

#####################################################
#      Importando libraries de text analytics
#####################################################
import jsonpickle
import re
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk import tokenize 
import numpy as np   
import pandas as pd

tweets_df = tweets_to_df('tweetsF.json')
tweets_df.columns

#####################################################################
#                       Limpeza do texto
#####################################################################
#funcao de remocao de carateres nao desejados, lidando com urls, hashtags, etc.
def preprocess_tweet(tweet):
    tweet.lower() # a minusculas
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet) #URL a string "URL"
    tweet = re.sub('@[^\s]+','', tweet) #@username a "ACC_USUARIO"
    tweet = re.sub('[\s]+', ' ', tweet) #espacos em blanco multiplos a espacos em blanco individuais
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #hashtags: "#algo" a "algo"
    return tweet

stopwords = set(stopwords.words('portuguese')) #nao tem portuguese "built-in"

def feature_extraction(data, method = "tfidf"):
	#métodos: "tfidf" and "doc2vec"
	if method == "tfidf":
		from sklearn.feature_extraction.text import TfidfVectorizer
		tfv=TfidfVectorizer(sublinear_tf=True,max_df=0.90, min_df=2, max_features=2000, stop_words = stopwords) 
		features=tfv.fit_transform(data_treino)
	elif method == "doc2vec":
		None
	else:
		return "Incorrect inputs"
	return features

sentimento=['Positivo','Negativo','Neutro'] #string del classifier

#CORPORA USADA PARA TREINAR O MODELO
#tweets_df_TREINO = pd.read_csv('Tweets_Mg.csv')
tweets_df_TREINO = pd.read_csv('tweetsUSPbakedF.csv')
tweets_df_TREINO.count()

def train_classifier(features, label, classifier = "logistic_regression"):
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    if classifier == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        modelo = LogisticRegression(C=1.)
    elif classifier == "naive_bayes": 
        from sklearn.naive_bayes import MultinomialNB
        modelo = MultinomialNB()
    elif classifier == "random_forest": 
        from sklearn.ensemble import RandomForestClassifier
        modelo = RandomForestClassifier(n_estimators=200, random_state=0)
    else:
        print("Incorrect selection of classifier")
	#fit model to data
    modelo.fit(features, label)
    resultados = cross_val_predict(modelo, features, label, cv=30)
    print (metrics.classification_report(label,resultados,sentimento),'')
    print (pd.crosstab(label, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')

tweets_df_TREINO['Text'] =tweets_df_TREINO['Text'].apply(preprocess_tweet) #limpar dados de training
tweets_df_TREINO= tweets_df_TREINO.drop_duplicates(subset=['Text'], keep='first') #elimina tweets duplicados
tweets_df_TREINO['word_count_Limpos'] = tweets_df_TREINO['Text'].apply(lambda x: len(str(x).split(" ")))
tweets_df_TREINO = tweets_df_TREINO[(tweets_df_TREINO['word_count_Limpos'] >= 3)]   #drop short tweets (<2 palabras)
tweets_df_TREINO['Text']= tweets_df_TREINO['Text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>=2])) #drop short words (<2 caracteres)
tweets_df['Teste'] =tweets_df['text'].apply(preprocess_tweet) #limpar dados de teste
tweets_df= tweets_df.drop_duplicates(subset=['Teste'], keep='first') #elimina tweets duplicados
data_treino = np.array(tweets_df_TREINO.Text) 
label = np.array(tweets_df_TREINO.Classificacao)
features_treino = feature_extraction(data_treino, method = "tfidf")  #extraindo os features do dados de training
data_teste = np.array(tweets_df.Teste) 
train_classifier(features_treino, label, "naive_bayes") #treinando o modelo

#USANDO O MODELO TREINADO PARA CLASIFICAR OS TWEETS PREVIDENCIARIOS
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(sublinear_tf=True, stop_words = stopwords) 
features_treino=tfv.fit_transform(data_treino)  #extraindo os features do dados de treino
features_teste=tfv.transform(data_teste)  #extraindo os features do dados de teste
modelo = MultinomialNB()
modelo2 = RandomForestClassifier(n_estimators=200, random_state=0)
modelo.fit(features_treino,label)
resultados = cross_val_predict(modelo, features_treino, label, cv=30)
print (metrics.classification_report(label,resultados,sentimento),'')
print (pd.crosstab(label, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')

output=modelo.predict(features_teste)
tweets_df['Clasificados'] =pd.DataFrame(data=output)
Clasificados =pd.DataFrame(data=output)
import matplotlib
%matplotlib inline
Clasificados[0].value_counts().plot(kind='bar')

treinados =pd.DataFrame(data=tweets_df_TREINO['Classificacao'])
treinados['Classificacao'].value_counts().plot(kind='bar')

tweets_df['Clasificados'].describe()
tweets_df['Clasificados'].value_counts()

