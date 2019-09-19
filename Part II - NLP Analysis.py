# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:05:26 2019

@author: ALVAROALTAMIRANO
"""

import string
import re
import spacy
import pt_core_news_sm
import pandas as pd
from spacy.lang.pt import Portuguese
from spacy.lang.pt.examples import sentences 

# Create our list of punctuation marks
punctuations = string.punctuation
# Create our list of stopwords
nlp = pt_core_news_sm.load()
#Define limit for nlp max length and define the 'portuguese' parser
nlp.max_length = 25000000
parser = Portuguese()

def preprocess_tweet(tweet):
    tweet = tweet.lower() # a minusculas
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet) #URL a string "URL"
    tweet = re.sub('@[^\s]+','', tweet) #@username a "ACC_USUARIO"
    tweet = re.sub('[\s]+', ' ', tweet) #espacos em blanco multiplos a espacos em blanco individuais
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #hashtags: "#algo" a "algo"
    return tweet

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
stopwords = spacy.lang.pt.stop_words.STOP_WORDS #portuguese stopwords from spcacy news model

def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in stopwords]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens

tweets_df = pd.read_csv('tweets_df.csv')

tweets_df['text'] =tweets_df['text'].apply(preprocess_tweet) #limpar dados de training
tweets_df['word_count_Limpos'] = tweets_df['text'].apply(lambda x: len(str(x).split(" ")))
tweets_df = tweets_df[(tweets_df['word_count_Limpos'] >= 3)]   #drop short tweets (<2 palabras)
tweets_df['text']= tweets_df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>=3])) #drop short words (<2 caracteres)
#tweets_df['text'] =tweets_df['text'].apply(tokenizeText) #limpar dados de training

###POS, NER ANALYSIS, computationally intensive, aprox 1 gb of ram per 100 thousand words for spaCy's portuguese model. 
#If limited computing power, subsample like in the following example
textnew = ''.join(str(X) for X in tweets_df['text'][0:50])
len(textnew)
#doc = nlp(textnew)
#Disabling ner and parser options for memory to focus on POS only
doc = nlp(textnew, disable = ["ner", "parser"])

#Example loop to extract text's nlp attributes
# # =============================================================================
#  for token in doc:
#      print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#              token.shape_, token.is_alpha, token.is_stop)
# # =============================================================================
# =============================================================================
# mapping frequency counts of pos
pos_count = {}
for token in doc:
    # ignore stop words
    if token.is_stop:
        continue
    # pos should be one of these:
    # 'VERB', 'NOUN', 'ADJ' or 'ADV'
    if token.pos_ == 'VERB':
        if token.lemma_ in pos_count:
            pos_count[token.lemma_] += 1
        else:
            pos_count[token.lemma_] = 1

print("top 10 VERBs {}".format(sorted(pos_count.items(), key=lambda kv: kv[1], reverse=True)[:10]))

#most common verbs, nouns and adjectives with pandas
tweets_df['text'].str.count("reformar").sum()

#Text concordance with NLTK
from nltk.text import Text
from nltk import tokenize 

textnew = ''.join(str(X) for X in tweets_df['text']) #Transforming to string to use nltk's text concordance
text_tokens = tokenize.word_tokenize(textnew, language='portuguese')
textList = Text(text_tokens)
textList.concordance('idoso')

#WORD CLOUD with brazil's map as a mask
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
stopwords.add("pra")
stopwords.add("pro")
stopwords.add("vc")
stopwords.add("tá")
stopwords.add("pq")
stopwords.add("há")
stopwords.add("vou")

bandeira = np.array(Image.open("brasil.jpg"))
wc = WordCloud(background_color="white", mask=bandeira, max_words=400,stopwords=stopwords, contour_width=3, 
               contour_color='steelblue')

textolimpo = [w for w in  tweets_df['text'] if not w in stopwords]
wc =wc.generate(str(textolimpo))
plt.figure(figsize=(15,10))
wc.to_file(("bandeira.png"))
# Display the generated image:
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

#Distribution of locations, day and timeof tweets
pd.Series(tweets_df['location'].str.lower()).value_counts()[:50]
pd.Series(tweets_df['country'].str.lower()).value_counts()[:20]

pd.Series(tweets_df['day']).value_counts()[:10]
pd.Series(tweets_df['hour']).value_counts()[:20]
 
########################################################################################
#Dependencies trees ***RUN IN jupyter as spyder cannot yet embed htmls into the console
########################################################################################
import spacy
from spacy import displacy
from spacy.lang.pt import Portuguese
from spacy.lang.pt.examples import sentences 
import pt_core_news_sm
nlp = pt_core_news_sm.load()

doc = nlp(u"concordo plenamente com que disse deputado alessandro molon! sociedade parece estar sobre efeito de anestesia.")
displacy.render(doc, style="dep", jupyter= True)
    
##################################
#       Mapping the tweets
##################################
from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
from geopy.exc import GeopyError
import gmplot
from geopy.extra.rate_limiter import RateLimiter

tweets_df = pd.read_csv('tweets_df.csv')
tweets_df=tweets_df[0:50]
#locnew = ''.join(str(X) for X in tweets_df['location'][0:15])

geolocator = Nominatim(user_agent='unilyrics@gmail.com')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
# Go through all tweets and add locations to 'coordinates' dictionary
coordinates = {'latitude': [], 'longitude': []}  
for count, user_loc in enumerate(str(tweets_df.location)):  
    try:
        location = geolocator.geocode(user_loc)

        # If coordinates are found for location
        if location:
            coordinates['latitude'].append(location.latitude)
            coordinates['longitude'].append(location.longitude)

    # If too many connection requests
    except (GeopyError, AttributeError):
        pass
# Instantiate and center a GoogleMapPlotter object to show our map
gmap = gmplot.GoogleMapPlotter(30, 0, 3)
# Insert points on the map passing a list of latitudes and longitudes
gmap.heatmap(coordinates['latitude'], coordinates['longitude'], radius=20)
# Save the map to html file
#gmap.apikey = "AIzaSyB-OCKjWcW1kBNYOT-KDbCPYtqG4ubRVkY"
gmap.draw("INSS_tweets_heatmap.html") 

#Poucos tweets com geolocalizadores
my_tab = pd.crosstab(index=tweets_df["country"],  # Make a crosstab
                              columns="count")      # Name the count column