import string
import nltk
from spacy.lang.en import English
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

nltk.download('stopwords')

def tokenize_preprocess_corpus(reports):
    '''This function uses spacy tokenizer'''

    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    # this will be a nested list of tokens [['word1', 'word2',...], [...], [...], ...]
    tokenized_reports = []

    for report in reports:
        report = cleanText(report)
        tokens = tokenizer(report.strip())
        # 'token' is not a string type but rather spacy.token type
        # converting the spacy tokens into lists of words
        new_tokens = [str(token) for token in tokens]

        # removes the stopwords from the original tokens (still working-not completed)
        new_tokens = remove_stopwords(new_tokens)

        new_tokens=stemming(new_tokens)

        # appending unique word tokens only per document
        tokenized_reports.append(new_tokens)

    return tokenized_reports

def cleanText(text):
        text = re.sub('\|\|\|', ' ', text)
        text = text.lower()
        text = text.replace('x', '')
        return text

def stemming(tokens):

    stemmer = SnowballStemmer('english')
    stemmed_words=[]
    for token in tokens:
        stemmed_words.append(stemmer.stem(token))
    return stemmed_words

def remove_stopwords(tokens):
    '''This function will remove the stopwords'''

    # use this if you use stopword list directly from nltk

    stopWords = set(stopwords.words('english'))

    word_tokens = []

    for token in tokens:
        # removing stopwords and punctuations
        if token.lower() not in stopWords:
            word_tokens.append(token)
            # print(token)

    return word_tokens

#this was a test basis manual tokenization - not using for this classification
def manual_tokenization(reports):
    '''This function will tokenize the document and returns a list of tokenized doc'''

    # this will be a nested list of tokens [['word1', 'word2',...], [...], [...], ...]
    tokenized_reports = []

    for report in reports:
        # lowercase the string
        report = report.lower()

        # removing punctuation symbols
        report = report.translate(str.maketrans('', '', string.punctuation))

        # tokenize the text based on white space
        tokens = report.split()

        tokenized_reports.append(tokens)

    print(tokenized_reports[0])

    return tokenized_reports
