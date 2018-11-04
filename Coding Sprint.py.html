
# coding: utf-8

# Notes from meeting with Prof. Westwood:
# 
# * think about spliting into a test and a training set (90-10?)
# * In order to assess how well the model is working, you need to use 90-10 to compare the prediction to the model
# 
# * today:
#     * Compute felish 
#     * valance
#     * topics
# * other things:
#     * you could pickle the text
#     * export text (he has code)

# # Initial Settings

# In[6]:


# import packages
import os
import json
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
import random
from gensim import corpora, models
import gensim
import pprint
import string

wordnet_lemmatizer = WordNetLemmatizer()


# In[7]:


##importing data

import os
import json
from pprint import pprint

if os.name == 'nt':
    root_dir = "..\\..\\data\\"
else:
    root_dir = "../../data/"
#"../../data/"
# "/data/"

with open(root_dir + 'fake_news.json') as data_file:    
    fake_news = json.load(data_file)


# In[8]:


#printing the data

print("Regular Print")
print(fake_news[0])
print("\n")
print ("Pretty Print:")
pprint(fake_news[0])


# Keys:
# * author
# * comments
# * country
# * domain_rank
# * language
# * likes
# * main_img_url
# * ord_in_thread
# * participants_count
# * published
# * replies_count
# * shares
# * site_url
# * spam_score
# * text
# * thread_title
# * title
# * type
# * uuid

# # Information about the Flesch-Kincaid test

# ### The Flesch Reading Ease formula

# 
# returns the Flesch Reading Ease Score. Following table is helpful to access the ease of readability in a document.
# 
# * 90-100 : Very Easy 
# * 80-89 : Easy 
# * 70-79 : Fairly Easy 
# * 60-69 : Standard 
# * 50-59 : Fairly Difficult 
# * 30-49 : Difficult 
# * 0-29 : Very Confusing

# In[53]:


##Flesch Kincaid:
import readability 
from nltk import word_tokenize

fleschkincaid_scores = []
minimal_word_count = []

for article in fake_news:
    
    text = word_tokenize(article['text'])
    #print(len(text))
    if len(text) < 3:
        minimal_word_count.append(article['title'])
        fleschkincaid_scores.append(np.NaN)
    else:    
        score = readability.getmeasures(text, lang=u'en', merge=False)
        fleschkincaid_scores.append(score)

print(minimal_word_count)

print(fleschkincaid_scores[0])
print ("\n")
print(fleschkincaid_scores[1])


# # Source Analysis

# In[52]:


## count number of unique sources

sources = {}

for article in fake_news:
    site = article['site_url']
    if site in sources.keys():
        sources[site] += 1
    else:
        sources[site] = 1

#print("The number of data entries in fake_news is " + str(len(fake_news)) + ".")
#print("The number of unique sources in fake_news is " + str(len(sources)) + ".")
#print ("Full dictionary of unique sources:"   )     
print (sources)



# # Headline Analysis

# In[46]:


## Valence and Arousal Score
with open(root_dir + 'anew.json') as data_file:    
    anew = json.load(data_file)
    
headline_arousal_score = []
headline_sentiment_score = []
count = 0

for headline in fake_news:
    
    #Tokenize words
    words = word_tokenize(headline["thread_title"])
    words = [word.lower() for word in words if word.isalpha()]
    
    arousal = 0
    sentiment = 0
    word_count = 0
    
    #Lemmatize words
    for word in words:
        word = wordnet_lemmatizer.lemmatize(word, pos='v')
    
    #Check if word is in the anew dictionary, calculate sentiment and arousal 
    for word in words:
        word_count = word_count + 1
        for anew_word in anew: 
            if word == anew_word["Word"]:
                arousal = arousal + anew_word["Arousal"]
                sentiment = sentiment + anew_word["Valence"]
   
    #Checking word count; if it's zero, setting the scores to 0 because we don't want to divide by 0!
    if word_count > 0:
        sentiment_score = sentiment / word_count 
        arousal_score = arousal / word_count
        headline_sentiment_score.append(sentiment_score)
        headline_arousal_score.append(arousal_score)
    else:
        headline_sentiment_score.append(0)
        headline_arousal_score.append(0)


# In[27]:


## Is the headline a question?

question = []
for headline in fake_news:
    #List of all headlines
    headlines = headline["thread_title"]
    
    #Looping through each character and in each headline and searching for ?
    count = 0
    for i in range(len(headlines)):
        if (headlines[i]=="?"):
            count += 1
    
    # 1 = there is a question; 0 = no question
    if count>0:
        question.append(1)
    else:
        question.append(0)

print (question)


# In[32]:


## Does the headline include numbers?

numbers = []

for headline in fake_news:
    #List of all headlines
    headlines = headline["thread_title"]
    
    count = 0
    #Looping through each character and in each headline and searching for digit
    for i in range(len(headlines)):
        num = unicode(headlines[i])
        if (num.isnumeric()):
            count += 1
    
    # 1 = there is a number; 0 = no number 
    if count>0:
        numbers.append(1)
    else:
        numbers.append(0)


# In[13]:


##Average number of likes/separating articles based on likes

numoflikes = 0
numofshares = 0

#Calculates average number of likes
for article in fake_news:
    numoflikes = numoflikes + float(article["likes"])
    avgnumoflikes = numoflikes/float(len(fake_news))

more_popular_articles = []
less_popular_articles = []
more_popular = []
less_popular = []

#Divides articles to more_popular, less_popular if more than, less than the average number of likes. 
for article in fake_news:
    if (float(article["likes"])>avgnumoflikes):
        more_popular.append(float(article["likes"]))
        more_popular_articles.append(article)
    else:
        less_popular.append(float(article["likes"]))
        less_popular_articles.append(article)
        
print avgnumoflikes
print more_popular
print len(more_popular)
print len(less_popular)
print len(fake_news)


# # Topic Modeling

# In[31]:


#1.2. Generate a DTM (do this twice). Once for all emails and once for a random subset of 1,000 emails.
#You might find random.sample() useful

from sklearn.feature_extraction import stop_words

def prepare_dtm(data):
    texts = []
    for text in data:
        # clean and tokenize document string (forcing unicode)

        text = text["text"].encode("ascii", "ignore").lower()
        # if using python 2
        text = text.translate(None, string.digits)
        text = text.translate(None, string.punctuation)
        # if using python 3
        # remove_digits = str.maketrans('', '', string.digits)
        # text= text.translate(remove_digits)
        # remove_punct = str.maketrans('', '', string.punctuation)
        # text= text.translate(remove_punct)
        tokens = nltk.word_tokenize(text)

        # remove stop words from tokens
        tokens = [i for i in tokens if not i in stop_words.ENGLISH_STOP_WORDS]

        # add tokens to list
        texts.append(tokens)
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_n_most_frequent(50)

    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary

corpus_full, dictionary_full = prepare_dtm(fake_news)
#corpus_subset, dictionary_subset = prepare_dtm(random.sample(fake_news,1000))


# In[32]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.LdaModel(corpus_full, num_topics=40, id2word=dictionary_full, passes=20)

import pprint
pprint.pprint(model.print_topics(num_topics=40, num_words=5))

loadings = model[corpus_full]
for document in loadings:
    print(document)

topic_count = [0] * 40

for document in loadings:
    for topic in document:
        topic_count[topic[0]] += 1

print(topic_count)

topic_frequency = [0] * 40
i=0
for topic in topic_count:
    topic_frequency[i] = float(topic) / float(len(fake_news)) *100
    i +=1

print(topic_frequency)


# In[36]:


import pandas as pd

topic_label = list(range(0, 40))

df = pd.DataFrame({"topics": topic_label, "frequency":topic_frequency})

df = df.loc[df['frequency'] > 20]
labels = ["Topic1", "Topic2","Topic3","Topic4","Topic5","Topic6","Topic7","Topic8","Topic9","Topic10","Topic11","Topic12","Topic13","Topic14","Topic15","Topic16","Topic17","Topic18","Topic19","Topic20","Topic21","Topic22","Topic23","Topic24", "Topic25"]
df['labels'] = labels
print df


# # Zero Inflated Model Python

# In[18]:


from __future__ import print_function
import numpy as np
from scipy import stats
from scipy.misc import factorial
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import pandas as pd     


# In[16]:


class ZeroInflatedPoisson(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        pi = params[0]
        lambda_ = params[1]

        return -np.log(zip_pmf(self.endog, pi=pi, lambda_=lambda_))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            lambda_start = self.endog.mean()
            excess_zeros = (self.endog == 0).mean() - stats.poisson.pmf(0, lambda_start)
            
            start_params = np.array([excess_zeros, lambda_start])
            
        return super(ZeroInflatedPoisson, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)


# In[ ]:


model = ZeroInflatedPoisson(x)
results = model.fit()


# In[69]:


title = []
uuid = []
likes = []
sources = []

for article in fake_news:
    title.append(article["title"])
    uuid.append(article["uuid"])
    likes.append(article["likes"])
    sources.append(article["site_url"])


# In[71]:


#Creating a dataframe
#special term in regression that allows for share of 0s to be a separate funciton of x's

df = pd.DataFrame({"Title": title, "ID": uuid, "Likes": likes, "Sources": sources,
                   "Question":question,"Num": numbers, "Arousal": headline_arousal_score, 
                   "Sentiment": headline_sentiment_score, "Flesch-Kincaid": fleschkincaid_scores})

print (df)


# In[72]:


print (len(headline_sentiment_score))
print (len(fleschkincaid_scores))
print (len(question))
print (len(numbers))
print (len(likes))

