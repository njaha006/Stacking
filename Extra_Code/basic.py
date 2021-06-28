from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from spacy.lang.bn import Bengali
import matplotlib.pyplot as plt

# reads the two files
def read_files():
    with open('train.txt', 'r', encoding='utf-8') as train:
        trainData = train.readlines()   # copy the content of the file in a list

    with open('test.txt', 'r', encoding='utf-8') as test:
        testData = test.readlines()

    # Read the Bangla Stop word list from file by: https://github.com/stopwords-iso/stopwords-bn
    with open('stopwords-bn.txt', 'r', encoding='utf-8') as test:
        stopwords_bn = test.readlines()
        # the above stopwords contains newline \n
        stop_bn = []

        for word in stopwords_bn:
            stop_bn.append(word.rstrip("\r\n"))

    return trainData, testData, stop_bn


# a dummy function that just returns its input
def identity(x):
    return x


# decide on TF-IDF vectorization for feature
# based on the value of tfidf (True/False)
def tf_idf_func(tfidf, stopwords_bn):
    # let's use the

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    # for our stop_word we also need to set the analyzer to a dummy function
    if tfidf:
        vec = TfidfVectorizer(analyzer=identity, stop_words = stopwords_bn,
                              preprocessor = identity, tokenizer = identity,  ngram_range=(1, 3))
    else:
        vec = CountVectorizer(analyzer=identity, stop_words = stopwords_bn,
                              preprocessor = identity, tokenizer = identity,  ngram_range=(1, 3))

    return vec


'''
** This function takes a Bangla Corpus of multiple lines of strings and returns a list of Tokens
$ params -> corpus =  a document with multiple lines of strings
$ returns -> tokens = a list of all the tokens in the corpus
'''
def tokenize_corpus(corpus):

    nlp = Bengali()  # use directly
    # nlp = spacy.blank('fi')  # blank instance - other way to use

    # creates a tokenizer instance specific to the language
    tokenizer = Bengali().Defaults.create_tokenizer(nlp)

    # Option 1: This is for a single line of string
    # tokens = tokenizer(u'বরাবর, মাননীয় প্রধানমন্ত্রী গণপ্রজাতন্ত্রী। বাংলাদেশ সরকার। মাননীয় প্রধানমন্ত্রী।')
    # for token in tokens:
    #     print(token)

    # Option 2: This is for a multiple line document e.g., list[]
    # for lines in tokenizer.pipe(corpus, batch_size=50):
    #     for token in lines:
    #         print(token)

    # Option 3: Use option 1 in a loop (I personally prefer this for simplicity)
    documents = []
    labels = []

    for line in corpus:
        word_tokens = []
        tokens = tokenizer(line.strip())    # line.stip() removes extra newline between characters

        for token in tokens:
            word_tokens.append(str(token))  # 'token' is not a sting type but rather spacy.token type
            # print(token)

        # 6-class problem: "sad", "happy", "disgust", "surprise", "fear", "angry"
        labels.append(word_tokens[0])  # tokens[0] is one of 6 topic types

        documents.append(word_tokens[1:])  # append the text - starts from 2nd tokens


    # just to check if everything is okay or not (comment it out)
    # print(word_tokens)
    # for word in word_tokens:
    #     print(word)

    return documents, labels



# SVM classification
def SVM_Linear(trainDoc, trainClass, testDoc, testClass, tfIdf, stopwords_bn):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a SVM classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.SVC(kernel='linear', C=1.0))] )

    # Fit/Train Multinomial Naive Bayes classifier according to trainDoc, trainClass
    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(testDoc)

    print("\n########### Default Linear SVM Classifier For ###########")

    # Call to function(s) to do the jobs ^_^
    calculate_measures(classifier, testClass, testGuess)


# SVM classifiers results for different values of C
# Returns different accuracy and f1-scores as list
def SVM_loop(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Decision Trees classifier

    C_val = []
    accu = []
    f1 = []

    print("\n##### Output of SVM classifier for different values of C (1-10) [TfidfVectorizer] #####")
    c = 1

    for k in range(1, 11):
        classifier = Pipeline( [('vec', vec),
                                ('cls', svm.SVC(kernel='linear', C=c))] )       # An interval of 1

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        C_val.append(c)
        accu.append(accuracy_score(testClass, testGuess))
        f1.append(f1_score(testClass, testGuess, average='macro'))

        c += 1      # C in interval of 1
        # print("K =", k, ": Accuracy = ", round(accuracy_score(testClass, testGuess), 3), "  F1-score (micro) = ", round(f1_score(testClass, testGuess, average='macro'), 3))

    print()
    for i in range(1, 11):
        print("C=",round(C_val[i-1],1),"   Accuracy=",accu[i-1],"     F1-score=",f1[i-1])

    return C_val, accu, f1


# for calculating different scores
def calculate_measures(classifier, testClass, testGuess):

    # Compare the accuracy of the output (Yguess) with the class labels of the original test set (Ytest)
    print("Accuracy = "+str(accuracy_score(testClass, testGuess)))
    print("F1-score(macro) = "+str(f1_score(testClass, testGuess, average='macro')))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
    print(classification_report(testClass, testGuess, labels=classifier.classes_, target_names=None, sample_weight=None, digits=3))

    # Showing the Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(testClass, testGuess, labels=classifier.classes_)
    print(classifier.classes_)
    print(cm)
    print()


# Draw plots based on different values of some parameter (val)
def draw_plots(val, accu, f1, value_name):
    plt.plot(val, accu, color='red', label='Accuracy')
    plt.plot(val, f1, color='yellow', label='F1-score')

    x_label = "Values of "+value_name
    plt.xlabel(x_label)
    plt.ylabel('Scores')
    plt.legend()

    plt.show()

# this is the main function but you can name it anyway you want
def main():

    # reads files
    trainSet, testSet, stop_bn = read_files()

    # divides the files into tokenized documents and class labels
    # this is for SVM
    trainDoc, trainClass = tokenize_corpus(trainSet)
    testDoc, testClass = tokenize_corpus(testSet)

    # Test the Linear SVM (True for Binary Class) with Tf-Idf Vectorizer
    print("\n\n Running the Model - Linear SVM:")
    SVM_Linear(trainDoc, trainClass, testDoc, testClass, True, stop_bn)

    # Try different values of C in linear SVM and To collect the data for curve
    C_val, accu, f1 = SVM_loop(trainDoc, trainClass, testDoc, testClass, stop_bn, True)
    draw_plots(C_val, accu, f1, "C")


# program starts from here
if __name__ == '__main__':
    main()
