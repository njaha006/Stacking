import Read_Data
import Pre_Processing
import SVM_Classification
from gensim.models import doc2vec
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import spatial


def cos_sim(totalDocument):


    keywords=[['platform', 'chip', 'zone, ’hole', 'aisl', 'roller', 'junction', 'exit', 'hall', 'catapul', 'board', 'water', 'desk', 'trench', 'shore', 'auditorium', 'grass', 'walkway', 'rotat', 'rain'],
              ['part', 'electr', 'digger', 'sign', 'breaker', 'termin', 'wire', 'panel', 'connector', 'blast', 'pole', 'sink', 'touch', 'socket', 'electrocut', 'shock', 'filter','screwdriv', 'transform', 'sign'],
              ['heat', 'temperatur', 'wrap', 'steam', 'burn', 'boiler', 'engulf', 'tube', 'oil', 'air', 'fire', 'chute','cooker', 'dryer', 'fryer', 'blew', 'pipelin', 'dislodg', 'molten', 'kettl'],
              ['firework', 'monoxid', 'odor', 'lightn', 'hydrofluor', 'fuel', 'diesel', 'pressur', 'chemic', 'chimney',
               'tanker', 'furnac', 'tetrazin', 'liquid', 'product', 'liquor', 'fluid', 'cylind', 'chlorin', 'carbon'],
              ['plywood', 'rail', 'beam', 'shield', 'deck', 'crate', 'shelv', 'arm', 'dock', 'tire', 'handl', 'pallet',
               'lever', 'scaffold', 'deck', 'case', 'screwdriv', 'etch','cart', 'lift'],
              ['truck', 'trailer', 'skidder', 'stuck', 'tractor', 'minivan', 'forklift', 'crusher', 'scraper', 'loader',
               'crane', 'excav', 'vehicl', 'fractur', 'bulldoz', 'mower', 'contain', 'picker', 'driver', 'move'],
              ['drill', 'blade', 'saw', 'rotat', 'press', 'shaft', 'conveyor', 'belt', 'lath', 'hose', 'jack', 'roller',
               'brake', 'gear', 'shear', 'grenad', 'fan', 'guardrail', 'cutter', 'knotch'],
              ['pave', 'park', 'concret', 'public', 'tool', 'inmat', 'stack', 'hallway', 'tame', 'spillway', 'sweep',
               'drain', 'assembl', 'thumb', 'shipyard', 'lead', 'stair', 'clock', 'tree', 'limb'],
              ['roof', 'ladder', 'top', 'build', 'billboard', 'floor', 'ankl', 'loos', 'sweeping', 'skylight', 'stair', 'shingl', 'shovel', 'sheet', 'impal', 'stage', 'scrap', 'paver', 'unsecur', 'glide'],
              ['drug', 'stroke', 'respons', 'die', 'dizzi', 'allerg', 'reaction', 'heart', 'aerial', 'respiratori',
               'suffer', 'unconsci', 'numb', 'attack', 'conscienc', 'carrier', 'crew', 'shack', 'unspecifi', 'pressur']]



    def label_sentences(corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the complaint narrative.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.TaggedDocument(v, [label]))
        return labeled

    # print(tokenized_reports[0])



    totalDocument = label_sentences(totalDocument, 'Total_Document')
    keywords = label_sentences(keywords,'Keyword')
    all_data = totalDocument + keywords
    # print('check:{}'.format(all_data[0]))
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])
    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha


    def get_vectors(model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors

    '''vector repesentation of totalDocument which is the input argument of this function'''

    totalDocument_dbow = get_vectors(model_dbow, len(totalDocument), 300, 'Total_Document')


    '''vector representation of keyword'''

    keyword_vectors_dbow= get_vectors(model_dbow,len(keywords),300,'Keyword')



    # print("check train vector shape:{}".format(train_vectors_dbow[2]))
    #print("check keyword vector:{}".format(len(keyword_vectors_dbow[0])))
    # print("check total report vector:{}".format(len(total_report_vector)))


    # j = 0
    # test_guess = []
    # for x in total_report_vector:
    #     test_guess.append('')

    cosine_similarity = []
    for x in totalDocument_dbow:
        a = []
        for y in keyword_vectors_dbow:
            e = 1 - spatial.distance.cosine(x, y)
            '''converting the cosine similarities to probabilities using softmax function'''
            r = np.exp(e)/np.sum(np.exp(e))
            a.append(r)
        cosine_similarity.append(a)


    # print("check cosine_similarity_one:{}".format(len(cosine_similarity)))
    # print("check test guess lenght:{}".format(len(test_guess)))

    # print("check test guess:{}".format(test_guess[0]))
    return cosine_similarity












