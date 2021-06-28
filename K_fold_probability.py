import Eval_Matrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np
from sklearn.model_selection import KFold


#import time



# a dummy function that just returns its input
def identity(x):
    return x

# decide on TF-IDF vectorization for feature
# based on the value of tfidf (True/False)
def tf_idf_func(tfidf):
    # TODO - change the values
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)

    else:
        vec = CountVectorizer(preprocessor = identity, tokenizer = identity)

    return vec

def training_input_base_model(Document, Class):


    kf = KFold(n_splits=10)

    probability1=[]

    for train_index, test_index in kf.split(Document):


        train_reports = np.array(Document)[train_index.astype(int)]
        train_labels = np.array(Class)[train_index.astype(int)]
        test_reports = np.array(Document)[test_index.astype(int)]
        test_labels = np.array(Class)[test_index.astype(int)]

        # Get the prediction probability!!

        prob=get_probability(trainDoc=train_reports, trainClass=train_labels,testDoc=test_reports, testClass=test_labels,tfIdf=True)
        probability1.append(prob)

    return probability1




def get_probability(trainDoc, trainClass,testDoc, testClass,tfIdf):


    total_doc = []
    for x in trainDoc:
        total_doc.append(x)
    for x in testDoc:
        total_doc.append(x)

    '''at first we have to fit vectorizer with whole available vocabulary in our corpus'''

    vectorizer = tf_idf_func(tfIdf)
    vectorizer.fit(total_doc)

    '''we will get a sparse matrix from tfidfvectorizer.But we need array version so we are just converting it to array'''

    trainDoc_tfidf_dense = vectorizer.transform(trainDoc)
    trainDoc_tfidf = trainDoc_tfidf_dense.toarray()

    '''same will be done for test data'''

    testDoc_tfidf_dense = vectorizer.transform(testDoc)
    testDoc_tfidf = testDoc_tfidf_dense.toarray()



    '''LinearSVC() works as one to rest classifier'''
    classifier = svm.LinearSVC()

    # classifier = make_pipeline(SelectKBest(f_classif, k=500), svm.SVC(kernel='linear', C=2.0))

    '''SVC kernel is linear'''
    #classifier=svm.SVC(kernel='linear', C=2.0)

    ''''Here trainDoc_tfidf are the tf-idf vectors from training set and trainClass is the class labels for those documents'''
    classifier.fit(trainDoc_tfidf, trainClass)

    # Use the classifier to predict the class for all the documents in the test set testDoc.Here we are providing tf_idf vector

    ''' Save those output class labels in testGuess. But not needed in this code '''
    #testGuess = classifier.predict(testDoc_tfidf)

    '''wrong try for probability'''
    # p = np.array(classifier.decision_function(trainDoc_tfidf))
    # prob = np.exp(p) / np.sum(np.exp(p), axis=1)
    # classes = classifier.predict(trainDoc_tfidf)
    # print("Sample={}, Prediction={},\n Votes={} \nP={}".format(idx, c, v, s) for idx, (v, s, c) in enumerate(zip(p, prob, classes)))

    # train_vectors_dbow_new = []
    #
    # for x, y in zip(train_labels, train_vectors_dbow):
    #     if x == 'entangled work space':
    #         train_vectors_dbow_new.append(np.append(y, keyword_vectors_dbow[0]))



    i=0
    probability=[]
    for x in testClass:
        print("Actual prediction:{}".format(x))
        '''reshape is needed'''
        g=testDoc_tfidf[i].reshape(1, -1)

        d = classifier.predict(g)

        '''will get decision scores of each class'''
        print("prediction by classifier:{}".format(d))
        e = classifier.decision_function(g)

        '''converting the decision score using softmax function.Softmax function, a wonderful activation function that turns
        numbers aka logits into probabilities that sum to one.Softmax function outputs a vector that represents the probability
        distributions of a list of potential outcomes.'''

        prob=np.exp(e)/np.sum(np.exp(e),axis=1)
        probability.append(prob)
        print("Decision function:{}".format(prob))
        i +=1
    return probability










