import re
import csv
import nltk
import sklearn
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import RegexpStemmer

# Load Data
train = pd.read_json("../../data/RelatedVsNotRelated.json")
train2 = pd.read_json("../../data/AwarenessVsInfection.json")
train3 = pd.read_json("../../data/SelfVsOthers.json")

# ## Related Vs Not Related:
#  0: Not related to influenza
#  1: Related to influenza

train_not_related = train.loc[train['type'] == 0]
train_related = train.loc[train['type'] == 1]


# ## Awareness Vs Infection

# 0: Influenza infection
# 1: Influenza awareness

train_infection = train2.loc[train2['type'] == 0]
train_awareness = train2.loc[train2['type'] == 1]


# ## Self Vs Others

# 0: Others (the tweet describes someone else)
# 1: Self (the tweet describes the author)

train_others = train3.loc[train3['type'] == 0]
train_self = train3.loc[train3['type'] == 1]


# ## Methods

# Define Word Stops
stopset = set(stopwords.words('english'))
morewords = ['who','which', 'I\'m','\'m']
stopset.update(morewords)

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
    words = word_tokenize(text)
    #Regex for Suffixes
    st = RegexpStemmer('ing$|s$|e$|able$|ible$|ful$|less$|ive$|acy$|al$|ance$|ence$|dom$|er$|or$|ism$|ist$|ity$|ty$|ment$|ship$|sion$|tion$|ate$|en$|ify$|fy$|ize$|ise$', min=4)
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
    # Calculate frequency distribution
    fdist = nltk.FreqDist(words)
    for word, frequency in fdist.most_common(number):
        print('{}: {}'.format(word, frequency))

def create_cvs(text, name_file, number):

    with open('./most_common/'+name_file+'.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        data = [['Word', 'Frequency']]
        # Calculate frequency distribution
        fdist = nltk.FreqDist(text)
        for word, frequency in fdist.most_common(number):
            data.append([word, frequency])
        a.writerows(data)

#Clean text on my Dataframe
train_related = clean_text(train_related)
train_not_related= clean_text(train_not_related)

train_infection = clean_text(train_infection)
train_awareness = clean_text(train_awareness)

train_others = clean_text(train_others)
train_self = clean_text(train_self)

# Tokenizing DF
list1 = nltk.tokenize.word_tokenize(' '.join(train_related['text']))
list2 = nltk.tokenize.word_tokenize(' '.join(train_not_related['text']))
list3 = nltk.tokenize.word_tokenize(' '.join(train_infection['text']))
list4 = nltk.tokenize.word_tokenize(' '.join(train_awareness['text']))
list5 = nltk.tokenize.word_tokenize(' '.join(train_others['text']))
list6 = nltk.tokenize.word_tokenize(' '.join(train_self['text']))

create_cvs(list1,'related', 10)
create_cvs(list2,'notrelated', 10)
create_cvs(list3,'infection', 10)
create_cvs(list4,'awareness', 10)
create_cvs(list5,'others', 10)
create_cvs(list6,'self', 10)
