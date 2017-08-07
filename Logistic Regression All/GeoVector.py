# coding: utf-8
import re
import csv
import nltk
import json
import numpy as np
import sklearn
import pandas as pd
from itertools import *
from sklearn.externals import joblib
from pandas import Series
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RegexpStemmer
from sklearn.feature_extraction.text import CountVectorizer
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def readjson(path):
    return pd.read_json(path, lines=True)

def loadjson2(path):
    # Reading the json as a dict
    with open(path) as json_data:
        data = json.load(json_data)
    # using the from_dict load function. Note that the 'orient' parameter
    #is not using the default value (or it will give the same error than you had)
    # We transpose the resulting df and set index column as its index to get this result
    jsonfile = pd.DataFrame.from_dict(data, orient='index').T.set_index('index')
    return jsonfile

def loadjson(path):
    # read the entire file into a python array
    with open(path, 'rb') as f:
        data = f.readlines()
    # remove the trailing "\n" from each line
    #with open('./random_sample.json', 'rb') as f:
    #    data = f.readlines()
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"
    #print data_json_str
    # now, load it into pandas
    sample = pd.read_json(data_json_str)
    return sample

#Classifier Load
#clf = joblib.load('./model/modelLogistic.pkl')

# In[33]:

#len(data09)
columns=['PREDICTION','PROBABILITY','ID','IDUSER','LOCATION','LAT','LON','TIMEZONE']
df = pd.DataFrame(columns=columns)
# In[34]:

# Define Word Stops
stopset = set(stopwords.words('english'))
morewords = ["'s", "swine", "bird", "h1n1", "'ve", "lol", "pig"]
stopset.update(morewords)
#Remove word from stopword list
itemsToRemove = ['can','am', 'are', 're', 'm','have','has','i', 'you', 'he', 'she', 'we', 'they']
stopset = [x for x in stopset if x not in itemsToRemove]

#Vectorisation
predictors  = pd.read_csv("./predictors.csv")
vocabulary = word_tokenize(' '.join(predictors))
count_vector = CountVectorizer(vocabulary=vocabulary)
#Predictors String
predictor_list=list(predictors)
predictors_str = ','.join(predictor_list)

#Methods
# Remove URLs, RTs, and twitter handles
def clean_data(text):
    #text= text.decode('utf-8')
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

def get_cleaned_text(text):
    try:
        cleaned_text= clean_data(text)
        cleaned_text= text_to_lower(cleaned_text)
        cleaned_text= remove_special_characters(cleaned_text)
        cleaned_text= remove_stopwords(cleaned_text)
        cleaned_text= stem_words(cleaned_text)
    except Exception:
        pass
    return text

def clean_text(df):
    for i, row in df.iterrows():
        cleaned_text = row['text']
        cleaned_text = get_cleaned_text(cleaned_text)
        df.set_value(i,'text',cleaned_text)
    return df

def get_vector(text):
    array_vector = count_vector.fit_transform([text]).toarray()[0]
    return array_vector

def classifier(X):
    return clf.predict(X)

def probability(X):
    return clf.predict_proba(X)

def text_classify(text):
    text= get_cleaned_text(text)
    X = [1] #Interceptor
    X2 = count_vector.fit_transform([text]).toarray()[0]
    X.extend(X2)
    return classifier(X)[0]

def text_prob(text):
    text= get_cleaned_text(text)
    X = [1] #Interceptor
    X2 = count_vector.fit_transform([text]).toarray()[0]
    X.extend(X2)
    return probability(X)[0][1]

#Vectorisation
predictors  = pd.read_csv("./predictors.csv")
vocabulary = word_tokenize(' '.join(predictors))
count_vector = CountVectorizer(vocabulary=vocabulary)

def get_vector(text):
    array_vector = count_vector.fit_transform([text]).toarray()[0]
    return array_vector

def classifier(X):
    X = np.array(X)
    X = X.reshape(1, -1)
    return clf.predict(X)

def probability(X):
    X = np.array(X)
    X = X.reshape(1, -1)
    return clf.predict_proba(X)

def text_classify(text):
    text= cleaned_text(text)
    X = [1] #Interceptor
    X2 = count_vector.fit_transform([text]).toarray()[0]
    X.extend(X2)
    return classifier(X)[0]

def text_prob(text):
    text= cleaned_text(text)
    X = [1] #Interceptor
    X2 = count_vector.fit_transform([text]).toarray()[0]
    X.extend(X2)
    return probability(X)[0][1]

def create_vector_file(df, path,name):
    data = clean_text(df)
    list_df = data['text'].tolist()
    array_df = count_vector.fit_transform(list_df).toarray()
    np.savetxt(path+'/'+name+'.csv', np.asarray(array_df), fmt='%i', delimiter=",", header=predictors_str)

def vector_prediction(index,vector):
    global df
    v= vector[0:116].values.tolist()
    X = [1] #Interceptor
    X.extend(v)
    prediction = classifier(X)[0]
    prob = probability(X)[0][1]
    df.set_value(index, columns,[prediction, prob, vector['ID'], vector['IDUSER'], vector['LOCATION'], vector['LAT'], vector['LON'], vector['TIMEZONE']],takeable=False)

def add_geodata_vector_file(path, output, df):
    foo =  pd.read_csv(path)
    foo['ID'] = Series(df['id_tweet'], index=foo.index)
    foo['IDUSER'] = Series(df['id_user'], index=foo.index)
    foo['DATE'] = Series(df['created_at'], index=foo.index)
    foo['LOCATION'] = Series(df['location'], index=foo.index)
    foo['LAT'] = Series(df['lat'], index=foo.index)
    foo['LON'] = Series(df['lon'], index=foo.index)
    foo['TIMEZONE'] = Series(df['created_at'], index=foo.index)
    foo.to_csv(output,sep=',', index=False, encoding='utf-8')

#TweetsAU
data = loadjson("./data/tweetDB-AU-05.json")
#create_vector_file(data,'./data_vectorised/data','geodata_tweetDB-AU-05')
add_geodata_vector_file('./data_vectorised/data/geodata_tweetDB-AU-05.csv','./data_vectorised/data/geodata_tweetDB-AU-05.csv', data)
