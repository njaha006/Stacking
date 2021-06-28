import Read_Data
import Pre_Processing
import SVM_Classification
import K_fold_probability
import Cosine_similarity
from sklearn.ensemble import GradientBoostingClassifier
import Eval_Matrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
SEED = 222
np.random.seed(SEED)

def check_for_new_test(classifier):

    while 1:
        new_str = str(input('Give a sample input string to Test:'))

        if new_str == '\n':
            break

        tokenized_input = Pre_Processing.tokenize_preprocess_corpus([new_str])

        predicted_output = classifier.predict(tokenized_input)

        print('The Prediction Is: {}'.format(predicted_output))
if __name__ == '__main__':

    # reads and processes the csv file
    csv_dataframe = Read_Data.read_csv(file_name='Data/Plus_800.csv', separator=',')
    reports, labels = Read_Data.process_data(data=csv_dataframe)
    unique_list = sorted(list(set(labels)))

    for item in unique_list:
        print('{} = {}'.format(item, labels.count(item)))

    # tokenize and pre-process reports
    tokenized_reports = Pre_Processing.tokenize_preprocess_corpus(reports)

    # print(tokenized_reports[0])

    # Divide the reports and labels into Training (1600) and Test Documents (400)
    train_reports = tokenized_reports[:1600]
    train_labels = labels[:1600]
    test_reports = tokenized_reports[1600:]
    test_labels = labels[1600:]

    '''Getting predictions(decision scores which is converted to probability using softmax function) from baser learner one
     (SVM in this case) for training data'''

    '''Here we are using Kfold(n=10):
     1.at first use first fold as test data.train the base learner on rest 9 folds.get predictions for first fold
     2. secondly use second fold as test data. and 1 plus 3-10 no fold total 9 fold as train document.
     3. repeat whole procedure'''


    prediction_probabilities1 = K_fold_probability.training_input_base_model(train_reports,train_labels)

    prediction_probabilities=[]
    for x in prediction_probabilities1:
        for y in x:
            prediction_probabilities.append(y)

    # t = np.array(prediction_probabilities)
    # nsamples, nx, ny = t.shape
    # d2_prediction_probabilities = t.reshape((nsamples, nx * ny))

    '''Getting predictions(cosine-similarities which is converted to probability using softmax function) from baser learner two
     (manual cs learner in this case) for training data'''

    cs_trainDoc = Cosine_similarity.cos_sim(train_reports)

    '''combining predictions get from base learners which will be used for training meta learner'''

    i = 0
    trainDoc_probability = []
    for x in cs_trainDoc:
        trainDoc_probability.append(np.append(x, prediction_probabilities[i]))
        i += 1

    '''predictions get from base learner 1  for test data(original document)'''

    test_prediction_probability = SVM_Classification.SVM_Normal(trainDoc=train_reports, trainClass=train_labels, testDoc=test_reports, testClass=test_labels, tfIdf=True)

    # t1 = np.array(test_prediction_probability)
    #
    # nsamples1, nx1, ny1=t1.shape
    # d2_test_prediction_probability = t1.reshape((nsamples1, nx1 * ny1))

    '''predictions get from base learner 2 for test data(original document)'''

    cs_testDoc = Cosine_similarity.cos_sim(test_reports)

    '''combining them'''

    i = 0
    testDoc_probability = []
    for x in cs_testDoc:
        testDoc_probability.append(np.append(x, test_prediction_probability[i]))
        i += 1


    '''Meta learner definition'''

    '''Gradient Boosting'''

    meta_learner = GradientBoostingClassifier(
        n_estimators=1000,
        max_features=4,
        max_depth=3,
        subsample=0.5,
        learning_rate=0.005,
        random_state=SEED)

    '''XGBOOST'''

    # meta_learner = XGBClassifier(n_estimators=300,random_state=42,learning_rate=0.1, seed=2, colsample_bytree=0.8, subsample=1)

    '''Randomforest'''

    # meta_learner =RandomForestClassifier(n_estimators=73,random_state=0)

    '''Logistic Regression'''
    # scaler = StandardScaler()
    # # Fit on training set only.
    # scaler.fit(trainDoc_probability)
    # # Apply transform to both the training set and the test set.
    # train_img = scaler.transform(testDoc_probability)
    # test_img = scaler.transform(testDoc_probability)
    #
    # meta_learner = LogisticRegression(solver='lbfgs')

    '''meta learner will be trained with the probability values get from base learner'''

    meta_learner.fit(trainDoc_probability,train_labels)

    '''now meta learner will be used to predict original test document'''

    testGuess = meta_learner.predict(testDoc_probability)





    # print("length of prediction_probabilities:{}".format(len(prediction_probabilities)))
    #
    # print("length of cs_traindoc:{}".format(len(cs_trainDoc)))

    title="ensemble"

    Eval_Matrics.calculate_measures(meta_learner, test_labels, testGuess, title)



