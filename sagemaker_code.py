#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np 
import pandas as pd 

# data processing/manipulation
pd.options.mode.chained_assignment = None
import re

# data visualization
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px

# stopwords, tokenizer, stemmer
import nltk
import torch
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist


# spell correction, lemmatization
from textblob import TextBlob
from textblob import Word

# sklearn
from sklearn.model_selection import train_test_split


# In[6]:


get_ipython().system('pip install torch')


# In[8]:


get_ipython().system('pip install transformers')


# In[10]:


get_ipython().system('pip install textblob')


# In[60]:


# Create a sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", framework="pt", model="cardiffnlp/twitter-roberta-base-sentiment", device=0 if torch.cuda.is_available() else -1)

def Sentiment_Analysis(df, batch_size=30):            #function to run sentiment analysis
    Tweet_list = df['filtered_tweets'].tolist()
    results = []
    for i in range(0, len(Tweet_list), batch_size):
        batch = Tweet_list[i:i+batch_size]
        batch_results = classifier(batch)
        results.extend(batch_results)
    labels = [result['label'].replace('LABEL_0', 'negative').replace('LABEL_1', 'neutral').replace('LABEL_2', 'positive') for result in results]
    scores = [result['score'] for result in results]
    df['label'] = labels
    df['score'] = scores
    return df[['tweet', 'filtered_tweets', 'label', 'score']]


# In[13]:


get_ipython().system('pip install boto3 pandas')


# In[46]:


import boto3
import pandas as pd

s3_client = boto3.client('s3')

path = 's3://dateset1000/donaldtrump_df.csv'

donaldtrump_df = pd.read_csv(path)


# In[52]:


donaldtrump_df.shape


# In[53]:


path = 's3://dateset1000/joebiden_df.csv'

joebiden_df = pd.read_csv(path)


# In[54]:


joebiden_df.describe()


# In[ ]:


trump_tweets = Sentiment_Analysis(donaldtrump_df)


# In[ ]:


biden_tweets = Sentiment_Analysis(joebiden_df)


# In[63]:


trump_tweets.to_csv('s3://teamoutput21/trump_tweets.csv')


# In[64]:


biden_tweets.to_csv('s3://teamoutput21/biden_tweets.csv')


# In[65]:


trump_tweets.shape


# In[66]:


biden_tweets


# In[ ]:




