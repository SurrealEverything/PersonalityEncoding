from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
# nltk.download()

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
def standardizeText(string, stop_words, ps):
    # tokenize
    tokens = word_tokenize(string)
    # remove stop words, punctuation and convert to lower case and stem
    std_tokens = [ps.stem(w.lower()) for w in tokens if w not in stop_words and w.isalpha()]
    return std_tokens

strs = [
    'Information Technology', 'logistics or economics', 'Nursing',
    'nursing; health physics; graphics', '', 'law']
print(strs)

for string in strs:
    std_tokens = standardizeText(string, stop_words, ps)
    print(std_tokens)