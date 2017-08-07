
# coding: utf-8

# In[48]:

import re
import nltk
import numpy as np
import sklearn
import pandas as pd
from patsy import dmatrices
from scikitplot import plotters as skplt
import matplotlib.pyplot as plt
from pandas import Series
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.stem import RegexpStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.metrics import classification_report

#Train All
data = pd.read_json("./data/trainall.json")
# Define Word Stops
stopset = set(stopwords.words('english'))
morewords = ["'s", "swine", "bird", "h1n1", "'ve", "lol", "pig"]
stopset.update(morewords)
#Remove word from stopword list
itemsToRemove = ['can','am', 'are', 're', 'm','have','has','i', 'you', 'he', 'she', 'we', 'they']
stopset = [x for x in stopset if x not in itemsToRemove]

#Methods
# Remove URLs, RTs, and twitter handles
def clean_data(text):
    text = text.replace('[^\x00-\x7F]','')
    words = [text for text in text.split() if 'http' not in text and not text.startswith('@') and text != 'RT']
    return ' '.join(words)

# Text to Lower Case
def text_to_lower(text):
    return text.lower()

# Remove some characters
def remove_special_characters(text):
    bad_chars = '-#?(){}<>:;.!$%&/=+*^-`\'0123456789'
    rgx = re.compile('[%s]' % bad_chars)
    return rgx.sub('', text)

# Create a set of Stopwords
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stopset]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stopset:
            filtered_sentence.append(w)

    return ' '.join(filtered_sentence)

# Stemming words
def stem_words(text):
    words = word_tokenize(text)
    #Regex for Suffixes
    st = RegexpStemmer('ing$|s$|able$|ible$|ful$|less$|ive$|acy$|al$|ance$|ence$|dom$|er$|or$|ism$|ist$|ity$|ty$|ment$|ship$|sion$|tion$|ate$|en$|ify$|fy$|ize$|ise$', min=4)
    stemmed = []
    for word in words:
        stemmed.append(st.stem(word))
    return ' '.join(stemmed)


def clean_text(df):
    for i, row in df.iterrows():
      cleaned_text = row['text']
      cleaned_text= clean_data(cleaned_text)
      cleaned_text= text_to_lower(cleaned_text)
      cleaned_text= remove_special_characters(cleaned_text)
      cleaned_text= remove_stopwords(cleaned_text)
      cleaned_text= stem_words(cleaned_text)
      df.set_value(i,'text',cleaned_text)
    return df


#Vectorisation
predictors  = pd.read_csv("./predictors_improved.csv")
vocabulary = word_tokenize(' '.join(predictors))

cv = sklearn.feature_extraction.text.CountVectorizer(vocabulary=vocabulary)
list_text = data['text'].tolist()

array_text = cv.fit_transform(list_text).toarray()
# Create CSV file
#np.savetxt("./all_predictors_improved.csv", np.asarray(array_text.astype(int)), fmt='%i', delimiter=",")


# In[54]:


foo =  pd.read_csv("./data_vectorised/all_predictors_improved.csv")
foo['RESULT'] = Series(data['type'], index=foo.index)
foo['ID'] = Series(data['id'], index=foo.index)
foo.to_csv('./data_vectorised/reducedVectorised.csv',sep=',', index=False)



# # Logistic Regression

# In[55]:

data2 = pd.read_csv("./data_vectorised/reducedVectorised.csv")


# In[57]:

y, X = dmatrices("RESULT ~ flu + gett + im + shot + think + have + sick + feel + am + you + got + bett + worried + hope + today + vaccine + scared + week + has + back + home + might + worse + year + fev + she + already + try + they + bed + bug + symptom + dr + bit + care + weekend + hand + stomach + rest + old + hell + health + suck + us", data2, return_type = "dataframe")
# flatten y into a 1-D array
y = np.ravel(y)

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X,y)

# check the accuracy on the training set
model.score(X, y)


# In[ ]:
