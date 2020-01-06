import pandas as pd
import numpy as np
import sys
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# import nltk
# from sklearn.feature_extraction.text import CountVectorizer
# import math
from sklearn import preprocessing
# nltk.download()
np.set_printoptions(threshold=sys.maxsize)
pd.options.display.width = 0
'''
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
count_vect = CountVectorizer(strip_accents='ascii')
'''

'''
def standardizeText(stop_words, ps):
    def _standardizeText(string):
        # tokenize
        # if math.isnan(string):
        if type(string) != str:
            return string
        tokens = word_tokenize(string)
        # remove stop words, punctuation and convert to lower case and stem
        std_tokens = [ps.stem(w.lower()) for w in tokens if w not in stop_words and w.isalpha()]
        return ' '.join(std_tokens)
        # return std_tokens

    return _standardizeText
'''
# read
df = pd.read_csv("data.csv",  sep='\t', error_bad_lines=False, dtype=object)
print(df.shape)
del df['Unnamed: 93']
del df['major']
print(df.shape)
print(df.columns)
# print(df.isna().any())
print(df.columns[df.isna().any()].tolist())

# for col in df.columns:
#    print(col)

print(df.isnull().sum())
# remove outliers
df['age'].values[df['age'].astype(int).values > 120] = 0
df['familysize'].values[df['familysize'].astype(int).values > 20] = 0
df = df.replace('NONE', np.nan)

'''
df['major'] = df['major'].fillna('')
df['major'] = df['major'].apply(standardizeText(stop_words, ps))
# df['major'] = list(count_vect.fit_transform(df['major']).toarray())
df['major'] = count_vect.fit_transform(df['major']).toarray()
# df = df.fillna(0)
'''
df = pd.get_dummies(
    df,
    prefix=[
        'urban', 'gender', 'engnat', 'hand', 'religion', 'orientation', 'race',
        'voted', 'married', 'uniqueNetworkLocation', 'country', 'source'
    ],
    columns=[
        'urban', 'gender', 'engnat', 'hand', 'religion', 'orientation', 'race',
        'voted', 'married', 'uniqueNetworkLocation', 'country', 'source'
    ],
    dummy_na=False
)

print(df.shape)
print(df.columns)
for col in df.columns:
   print(col)
print(df.columns[df.isna().any()].tolist())

# df = df.fillna(0)
# print(df.columns[df.isna().any()].tolist())

'''
x = df.values #returns a numpy array
# min_max_scaler = preprocessing.StandardScaler()
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x)
# inverse = scaler.inverse_transform(normalized)
'''

df.to_csv('ppData.csv', index=False)


# create X, Y
# split train, test, val

# Ni = number of input neurons.
# 94
# No = number of output neurons.
# 94
# Ns = number of samples in training data set.
# 100000
# α = an arbitrary scaling factor usually 2-10.
# 10
# Nh = Ns/(α∗(Ni+No))
# 53 = 100000/(10*(94+94))

# Radam
# Dense -> Leaky Relu -> Batch Norm -> Dropout
# pass the input of dropout layer to the Lambda layer

# MAIN
# 48 numerical 1-5
# METADATA
# 3 numerical big
# BIGFIVE
# 10 numerical 1-7
# LIE
# 16 numerical 0-1
# LABELS
# numerical 1-4
# onehot 3
# onehot 3
# onehot 2
# onehot 3
# numerical 1-100
# onehot 12
# onehot 5
# onehot 5
# onehot 2
# onehot 3
# numeric 1-20
# META
# onehot 2
# COUNTRY
# onehot huge (treat nan as different category)
# META
# onehot 3
# MAJOR
# normalize + tokenize + remove punctuation & stopwords + stemming
# + pick first word + treat nan as different category + onehot

# one neuron from each input stream
# Replace None with Nan
# inlocuiesc outlierii cu nimic
# TODO:
# vectorize
# normalize
# just drop major, it s out of scope at this point
# if anything, word2vec is better