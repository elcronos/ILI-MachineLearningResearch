import re
import csv
import nltk
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

# Load Most Common Words
most_common  = pd.read_csv("../Experiment02/most_common/related.csv")
most_common2 = pd.read_csv("../Experiment02/most_common/infection.csv")
most_common3 = pd.read_csv("../Experiment02/most_common/self.csv")
