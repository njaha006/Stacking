import Eval_Matrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np


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

def SVM_Normal(trainDoc, trainClass, testDoc, testClass, tfIdf):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)

    vec = tf_idf_func(tfIdf)

    # classifier = Pipeline( [('vec', vec),
    #                         ('cls', svm.LinearSVC())])

    total_doc = []
    for x in trainDoc:
        total_doc.append(x)
    for x in testDoc:
        total_doc.append(x)

    vectorizer = tf_idf_func(tfIdf)
    vectorizer.fit(total_doc)

    trainDoc_tfidf_dense = vectorizer.transform(trainDoc)
    trainDoc_tfidf = trainDoc_tfidf_dense.toarray()

    testDoc_tfidf_dense = vectorizer.transform(testDoc)
    testDoc_tfidf = testDoc_tfidf_dense.toarray()



    #LinearSVC() works as one to rest classifier
    classifier = svm.LinearSVC()

    # classifier = make_pipeline(SelectKBest(f_classif, k=500), svm.SVC(kernel='linear', C=2.0))
    #SVC kernel is linear
    #classifier=svm.SVC(kernel='linear', C=2.0)

    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc_tfidf, trainClass)

    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    '''slight change here'''
    ##testGuess = classifier.predict(testDoc_tfidf)

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
        g=testDoc_tfidf[i].reshape(1, -1)
        d = classifier.predict(g)

        '''will get decision scores of each class'''
        print("prediction by classifier:{}".format(d))
        e=classifier.decision_function(g)

        '''converting the decision score using softmax function.Softmax function, a wonderful activation function that turns
        numbers aka logits into probabilities that sum to one.Softmax function outputs a vector that represents the probability
        distributions of a list of potential outcomes.'''

        prob=np.exp(e)/np.sum(np.exp(e),axis=1)
        probability.append(prob)
        print("Decision function:{}".format(prob))
        i +=1

    return probability





