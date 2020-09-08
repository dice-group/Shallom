import os
import re
from collections import Counter, defaultdict
import itertools
import util as ut
import os.path
from numpy import linalg as LA
import numpy as np
import pandas as pd
import warnings
import sys
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

from util import performance_debugger


class Data:

    def __init__(self, data_dir=None):

        self.info = {'dataset': data_dir}
        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data_with_checking(data_dir, data_type="valid",
                                                       entities=self.get_entities(self.train_data))
        self.test_data = self.load_data_with_checking(data_dir, data_type="test",
                                                      entities=self.get_entities(self.train_data))

    @staticmethod
    def load_data_with_checking(data_dir, entities, data_type="train"):
        assert entities
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            triples = f.read().strip().split("\n")
            data = []
            for i in triples:
                s, p, o = tuple(i.split())
                if s in entities and o in entities:
                    data.append([s, p, o])
            return data

    @staticmethod
    def load_data(data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    @staticmethod
    def get_entities(data):
        entities = set()
        for i in data:
            s, p, o = i
            entities.add(s)
            entities.add(o)
        return sorted(list(entities))

    @staticmethod
    def get_entity_pairs_with_predicates(triples):
        sub_obj_pairs = dict()
        for s, p, o in triples:
            sub_obj_pairs.setdefault((s, o), set()).add(p)
        return sub_obj_pairs


class Parser:
    def __init__(self, logger=False, p_folder: str = 'not initialized', k=1):
        self.path = 'uninitialized'
        self.logger = logger
        self.p_folder = p_folder
        self.similarity_function = None
        self.similarity_measurer = None
        self.K = int(k)

    def set_similarity_function(self, f):
        self.similarity_function = f

    def set_similarity_measure(self, f):
        self.similarity_measurer = f

    def set_experiment_path(self, p):
        self.p_folder = p

    def set_k_entities(self, k):
        self.K = k

    @performance_debugger('Preprocessing')
    def pipeline_of_preprocessing(self, f_name, bound=''):
        return self.inverted_index(f_name, bound)

    @performance_debugger('Preprocessing')
    def pipeline_of_preprocessing_new(self, f_name, bound=''):
        return self.get_triples(f_name, bound)

    @performance_debugger('Constructing Inverted Index')
    def get_triples(self, path, bound):

        vocabulary = {}
        num_of_rdf = 0

        sub_obj_pairs = dict()
        sentences = ut.generator_of_reader(bound, ut.get_path_knowledge_graphs(path), ut.decompose_rdf)

        predicates = set()
        for s, p, o in sentences:
            num_of_rdf += 1

            # mapping from string to vocabulary
            vocabulary.setdefault(s, len(vocabulary))
            predicates.add(p)
            vocabulary.setdefault(o, len(vocabulary))

            sub_obj_pairs.setdefault((vocabulary[s], vocabulary[o]), set()).add(p)

        print('Number of RDF triples:', num_of_rdf)
        print('Number of vocabulary terms: ', len(vocabulary))
        print('Number of predicates: ', len(predicates))

        num_of_resources = len(vocabulary)
        ut.serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        num_of_entities = num_of_resources - len(predicates)

        return sub_obj_pairs, num_of_entities + 1

    @performance_debugger('Preprocessing')
    def pipeline_of_data(self, f_name, bound=''):
        sentences = ut.generator_of_reader(bound, ut.get_path_knowledge_graphs(f_name), ut.decompose_rdf)
        vocabulary = {}

        data = []
        num_of_rdf = 0

        for s, p, o in sentences:
            num_of_rdf += 1

            # mapping from string to vocabulary
            vocabulary.setdefault(s, len(vocabulary))
            vocabulary.setdefault(p, len(vocabulary))
            vocabulary.setdefault(o, len(vocabulary))

            data.append((vocabulary[s], vocabulary[p], vocabulary[o]))

        num_of_resources = len(vocabulary)
        ut.serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        return data, num_of_resources

    @performance_debugger('Constructing Inverted Index')
    def inverted_index(self, path, bound):

        inverted_index = {}
        vocabulary = {}

        num_of_rdf = 0
        type_info = defaultdict(set)

        sentences = ut.generator_of_reader(bound, ut.get_path_knowledge_graphs(path), ut.decompose_rdf)

        predicates = set()
        for s, p, o in sentences:
            num_of_rdf += 1

            # mapping from string to vocabulary
            vocabulary.setdefault(s, len(vocabulary))
            vocabulary.setdefault(p, len(vocabulary))
            predicates.add(vocabulary[p])
            vocabulary.setdefault(o, len(vocabulary))

            inverted_index.setdefault(vocabulary[s], []).extend([vocabulary[p], vocabulary[o]])

        print('Number of RDF triples:', num_of_rdf)
        print('Number of vocabulary terms: ', len(vocabulary))
        print('Number of predicates: ', len(predicates))

        num_of_resources = len(vocabulary)
        ut.serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        ut.serializer(object_=list(inverted_index.values()), path=self.p_folder, serialized_name='inverted_index')

        ut.serializer(object_=type_info, path=self.p_folder, serialized_name='type_info')
        del type_info

        return inverted_index, num_of_resources, predicates
