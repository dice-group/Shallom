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
from sklearn.preprocessing import MultiLabelBinarizer
from models import Shallom
import torch

warnings.filterwarnings('ignore')

from util import *


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


class Experiment:
    def __init__(self, d, s):
        self.dataset = d
        self.settings = s
        self.model_name = self.settings['model_name']
        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name=self.model_name, p=self.storage_path)
        self.entity_idx = None  # will be filled.

    def processed_data(self, dataset: Data):
        """

        :type dataset: object
        """
        y = []
        x = []
        entitiy_idx = dict()

        sub_obj_pairs = dataset.get_entity_pairs_with_predicates(dataset.train_data)
        for s_o_pair, predicates in sub_obj_pairs.items():
            s, o = s_o_pair
            entitiy_idx.setdefault(s, len(entitiy_idx))
            entitiy_idx.setdefault(o, len(entitiy_idx))
            x.append([entitiy_idx[s], entitiy_idx[o]])
            y.append(list(predicates))
        x = np.array(x)

        binarizer = MultiLabelBinarizer()
        y = binarizer.fit_transform(y)

        return x, y, entitiy_idx, binarizer

    def eval_relation_prediction(self, model: Shallom, binarizer, triples):
        self.logger.info('Relation Prediction Evaluation begins.')

        x_, y_ = [], []

        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        rank_per_relation = dict()

        for i in triples:  # test triples
            s, p, o = i
            x_.append((self.entity_idx[s], self.entity_idx[o]))
            y_.append(p)
        # generate predictions.
        tensor_pred = torch.from_numpy(model.predict(np.array(x_)))
        # faster sorting.
        _, ranked_predictions = tensor_pred.topk(k=len(binarizer.classes_))

        ranked_predictions = ranked_predictions.numpy()

        assert len(ranked_predictions) == len(y_)

        classes_ = binarizer.classes_.tolist()

        for i in range(len(y_)):
            true_relation = y_[i]
            ith_class = classes_.index(true_relation)

            rank = np.where(ranked_predictions[i] == ith_class)[0]

            rank_per_relation.setdefault(true_relation, []).append(rank + 1)

            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)

        hits = np.array(hits)
        ranks = np.array(ranks)
        self.logger.info('########## Relation Prediction Results ##########')

        self.logger.info('Mean Hits @5: {0}'.format(sum(hits[4]) / (float(len(y_)))))
        self.logger.info('Mean Hits @3: {0}'.format(sum(hits[2]) / (float(len(y_)))))
        self.logger.info('Mean @1: {0}'.format(sum(hits[0]) / (float(len(y_)))))
        self.logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        self.logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / ranks)))

        self.logger.info('########## Relation Prediction Analysis ##########')

        for pred, ranks in rank_per_relation.items():
            ranks = np.array(ranks)

            average_hit_at_1 = np.sum(ranks == 1) / len(ranks)
            average_hit_at_3 = np.sum(ranks <= 3) / len(ranks)
            average_hit_at_5 = np.sum(ranks <= 5) / len(ranks)

            self.logger.info('{0}:\t Hits@1:\t{1:.3f}'.format(pred, average_hit_at_1))
            self.logger.info('{0}:\t Hits@3:\t{1:.3f}'.format(pred, average_hit_at_3))
            self.logger.info('{0}:\t Hits@5:\t{1:.3f}'.format(pred, average_hit_at_5))
            self.logger.info('{0}:\t MRR:\t{1:.3f}\t number of occurrence {2}'.format(pred, np.mean(1. / ranks), len(ranks)))
            self.logger.info('################################')

    def train_and_eval(self):
        self.logger.info("Info pertaining to dataset:{0}".format(self.dataset.info))
        self.logger.info("Number of triples in training data:{0}".format(len(self.dataset.train_data)))
        self.logger.info("Number of triples in validation data:{0}".format(len(self.dataset.valid_data)))
        self.logger.info("Number of triples in testing data:{0}".format(len(self.dataset.test_data)))

        self.logger.info('Data is being reformating for multi-label classificaion.')
        X, y, self.entity_idx, binarizer = self.processed_data(self.dataset)
        model = Shallom(settings=self.settings, num_entities=len(self.entity_idx), num_relations=y.shape[1])
        self.logger.info('Shallom starts training.')
        model.fit(X, y)
        self.eval_relation_prediction(model=model, binarizer=binarizer, triples=self.dataset.test_data)
        self.logger.info('Keras model is being serialized.')
        model.model.save(self.storage_path)  # serialize keras Sequential.
        model.embeddings_save_csv(self.entity_idx,self.storage_path)
        self.logger.info('Embeddings are stored as pandas dataframe.')


