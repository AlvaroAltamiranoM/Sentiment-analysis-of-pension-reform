# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:10:49 2019

@author: ALVAROALTAMIRANO
"""
import string
import re
import spacy
import pt_core_news_sm
import pandas as pd
from spacy.lang.pt import Portuguese
from spacy.lang.pt.examples import sentences 
from sklearn.base import TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

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

class CleanTextTransformer(TransformerMixin):
   def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
   def fit(self, X, y=None, **fit_params):
        return self
   def get_params(self, deep=True):
        return {}
    # Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()
    
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

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))

##CHOOSE CORPORA TO TRAIN
tweets_df_TREINO = pd.read_csv('tweetsUSPbakedF.csv')
#tweets_df_TREINO = pd.read_csv('Tweets_Mg.csv')

#Optional, for binary classification only
#tweets_df_TREINO = tweets_df_TREINO[(tweets_df_TREINO['Classificacao']=="Negativo") | (tweets_df_TREINO['Classificacao']=="Positivo")]

X = tweets_df_TREINO['Text'] # the features we want to analyze
ylabels = tweets_df_TREINO['Classificacao']# the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size = 0.25, random_state = 40)

classifier = MultinomialNB()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleanText", CleanTextTransformer()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)

# test accuracy results
preds = pipe.predict(X_test)
print("accuracy:", metrics.accuracy_score(y_test, preds))
print(metrics.classification_report(y_test, preds, 
    target_names=tweets_df_TREINO['Classificacao'].unique()))

print("Top 10 features used to predict: ")

def printNMostInformative(vectorizer, classifier, N):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)

printNMostInformative(vectorizer, classifier, 30)

##Apply model to our tweets_df dataset
#USANDO O MODELO TREINADO PARA CLASIFICAR OS TWEETS PREVIDENCIARIOS
tweets_df = pd.read_csv('tweetsF.csv')
#Pre-process our tweets!
tweets_df['text'] =tweets_df['text'].apply(preprocess_tweet) #limpar dados de training
tweets_df['word_count_Limpos'] = tweets_df['text'].apply(lambda x: len(str(x).split(" ")))
tweets_df = tweets_df[(tweets_df['word_count_Limpos'] >= 3)]   #drop short tweets (<2 palabras)
tweets_df['text']= tweets_df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>=3])) #drop short words (<2 caracteres)
#tweets_df['text'] =tweets_df['text'].apply(tokenizeText) #limpar dados de training

Yclass = tweets_df['text'] # the features we want to analyze

preds = pipe.predict(Yclass)
print("accuracy:", metrics.accuracy_score(y_test, preds))
print(metrics.classification_report(y_test, preds, 
    target_names=tweets_df_TREINO['Classificacao'].unique()))


tweets_df['Clasificados'] =pd.DataFrame(data=preds)
Clasificados =pd.DataFrame(data=preds)
import matplotlib
%matplotlib inline
Clasificados[0].value_counts().plot(kind='bar')

treinados =pd.DataFrame(data=tweets_df_TREINO['Classificacao'])
treinados['Classificacao'].value_counts().plot(kind='bar')

tweets_df['Clasificados'].describe()
tweets_df['Clasificados'].value_counts()
