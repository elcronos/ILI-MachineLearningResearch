
# coding: utf-8

# In[ ]:

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import RegexpStemmer

train = pd.read_json("../../data/RelatedVsNotRelated.json")
train2 = pd.read_json("../../data/AwarenessVsInfection.json")
train3 = pd.read_json("../../data/SelfVsOthers.json")


# ## Related Vs Not Related:
#  0: Not related to influenza
#  1: Related to influenza

# In[ ]:

train_not_related = train.loc[train['type'] == 0]
train_related = train.loc[train['type'] == 1]


# ## Awareness Vs Infection

# 0: Influenza infection
# 1: Influenza awareness

# In[6]:

train_infection = train2.loc[train2['type'] == 0]
train_awareness = train2.loc[train2['type'] == 1]


# ## Self Vs Others

# 0: Others (the tweet describes someone else)
# 1: Self (the tweet describes the author)

# In[7]:

train_others = train3.loc[train3['type'] == 0]
train_self = train3.loc[train3['type'] == 1]


# ## Methods

# In[8]:

# Define Word Stops
stopset = set(stopwords.words('english'))
morewords = ['who','which','isn\'t','aren\'t', 'I\'m','\'m']
stopset.update(morewords)


# In[9]:

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
    bad_chars = '#?(){}<>:;.!$%&/=+*^-'

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
    #Regex for Suffixes
    st = RegexpStemmer('ing$|s$|e$|able$|ible$|ful$|less$|ive$|acy$|al$|ance$|ence$|dom$|er$|or$|ism$|ist$|ity$|ty$|ment$|ship$|sion$|tion$|ate$|en$|ify$|fy$|ize$|ise$', min=4)

    stemmed = []

    for word in words:
        stemmed.append(st.stem(word))

    return stemmed


def clean_text(df):
    for i, row in df.iterrows():
      cleaned_text = row['text']
      cleaned_text= clean_data(cleaned_text)
      cleaned_text= text_to_lower(cleaned_text)
      cleaned_text= remove_special_characters(cleaned_text)
      cleaned_text= remove_stopwords(cleaned_text)
      #cleaned_text= stem_words(cleaned_text)
      df.set_value(i,'text',cleaned_text)
    return df

def create_wordcloud(list_words, name_cloud):
    wordcloud = WordCloud(
                      stopwords= stopset,
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(list_words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('./wordclouds/'+name_cloud, dpi=300)
    plt.show()

def print_frequency(words, number):
    for word, frequency in fdist.most_common(number):
        print(u'{};{}'.format(word, frequency))



#Clean text on my Dataframe
train_related = clean_text(train_related)
train_not_related= clean_text(train_not_related)

train_infection = clean_text(train_infection)
train_awareness = clean_text(train_awareness)

train_others = clean_text(train_others)
train_self = clean_text(train_self)


# In[ ]:

# Create Wordcloud
list1 = ' '.join(train_related['text'])
list2 = ' '.join(train_not_related['text'])
list3 = ' '.join(train_infection['text'])
list4 = ' '.join(train_awareness['text'])
list5 = ' '.join(train_others['text'])
list6 = ' '.join(train_self['text'])

#create_wordcloud(list1, 'wordcloud_related')
#create_wordcloud(list2, 'wordcloud_not_related')

#create_wordcloud(list3, 'wordcloud_infection')
#create_wordcloud(list4, 'wordcloud_awareness')

#create_wordcloud(list5, 'wordcloud_others')
#create_wordcloud(list6, 'wordcloud_self')


# In[ ]:

#Calculate frequency distribution
fdist = nltk.FreqDist(list1)

# Output top 50 words
print_frquency(list1)


# In[ ]:
