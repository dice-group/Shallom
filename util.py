import datetime
import os
import pickle
#from tensorflow.keras import backend as K
import numpy as np
import time
import bz2
import re
triple = 3


import datetime
import logging
import os
import time

def create_experiment_folder(folder_name='Experiments'):
    directory = os.getcwd() + '/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def performance_debugger(func_name):
    def function_name_decoratir(func):
        def debug(*args, **kwargs):
            long_string = ''
            starT = time.time()
            print('\n\n######', func_name, ' starts ######')
            r = func(*args, **kwargs)
            print(func_name, ' took ', time.time() - starT, ' seconds\n')
            long_string += str(func_name) + ' took:' + str(time.time() - starT) + ' seconds'

            return r

        return debug

    return function_name_decoratir


def pairwise_iteration(it):
    it = iter(it)
    while True:
        yield next(it), next(it)


def get_path_knowledge_graphs(path: str):
    """

    :param path: str represents path of a KB or path of folder containg KBs
    :return:
    """
    KGs = list()

    if os.path.isfile(path):
        KGs.append(path)
    else:
        for root, dir, files in os.walk(path):
            for file in files:
                print(file)
                if '.nq' in file or '.nt' in file or 'ttl' in file:
                    KGs.append(path + '/' + file)
    if len(KGs) == 0:
        print(path + ' is not a path for a file or a folder containing any .nq or .nt formatted files')
        exit(1)
    return KGs


def file_type(f_name):
    if f_name[-4:] == '.bz2':
        reader = bz2.open(f_name, "rt")
        return reader
    return open(f_name, "r")


def create_experiment_folder():
    directory = os.getcwd() + '/Experiments/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def serializer(*, object_: object, path: str, serialized_name: str):
    with open(path + '/' + serialized_name + ".p", "wb") as f:
        pickle.dump(object_, f)
    f.close()


def deserializer(*, path: str, serialized_name: str):
    with open(path + "/" + serialized_name + ".p", "rb") as f:
        obj_ = pickle.load(f)
    f.close()
    return obj_


def generator_of_reader(bound, knowledge_graphs, rdf_decomposer):
    for f_name in knowledge_graphs:
        reader = file_type(f_name)
        total_sentence = 0
        for sentence in reader:
            # Ignore Literals
            if '"' in sentence or "'" in sentence or '# started' in sentence:
                continue

            if total_sentence == bound: break
            total_sentence += 1

            try:
                s, p, o, flag = rdf_decomposer(sentence)

                # <..> <..> <..>
                if flag != triple:
                    print(sentence, '+', flag)
                    print('exitting')
                    exit(1)
                    continue

            except ValueError:
                print('value error')
                exit(1)

            yield s, p, o

        reader.close()


@performance_debugger('Training')
def learn(model, storage_path, x, y, batch_size=10000, epochs=1):
    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, use_multiprocessing=True)
    return model,history


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def decompose_rdf(sentence):

    flag = 0

    #components = re.findall('<(.+?)>', sentence)
    components = sentence.split()  # re.findall('<(.+?)>', sentence)
    
    if len(components) == 2:
        s, p = components
        remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
        literal = remaining_sentence[:-1]
        o = literal
        flag = 2

    elif len(components) == 4:
        del components[-1]
        s, p, o = components

        flag = 4

    elif len(components) == 3:
        s, p, o = components
        flag = 3


    elif len(components) > 4:

        s = components[0]
        p = components[1]
        remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
        literal = remaining_sentence[:remaining_sentence.index(' <http://')]
        o = literal

    else:
        ## This means that literal contained in RDF triple contains < > symbol
        raise ValueError()

    o = re.sub("\s+", "", o)
    s = re.sub("\s+", "", s)
    p = re.sub("\s+", "", p)
    
    if flag != 3:
        print(components)
        print(len(components))

        print('here')
        exit(1)

    return s, p, o, flag

def construct_subject_object_inverted_index(path, path_to_deseralize):
    s_o_inverted_index = dict()
    predicate_mapper = dict()

    vocab = dict()
    num_of_rdf = 0
    num_of_invaid_rdf_triples = 0
    # Construct s_o_inverted_index
    with open(path, 'r') as reader:
        for sentence in reader:
            num_of_rdf += 1

            try:
                # components = re.findall('<(.+?)>', sentence)
                # s, p, o = components[0], components[1], components[2]
                s, p, o = sentence.split()  # components[0], components[1], components[2]

            except:
                num_of_invaid_rdf_triples += 1
                continue

            # mapping from string to vocabulary
            vocab.setdefault(s, len(vocab))
            #            vocab.setdefault(p, len(vocab))
            vocab.setdefault(o, len(vocab))

            predicate_mapper.setdefault(p, len(predicate_mapper))

            s_o_inverted_index.setdefault((vocab[s], vocab[o]), []).append(predicate_mapper[p])

    print('Number of RDF triples processed:', num_of_rdf)
    print('Number of invalid RDF triples:', num_of_invaid_rdf_triples)

    print('Number of entities: ', len(vocab))
    print('Number of predicates: ', len(predicate_mapper))

    num_of_entities = len(vocab)
    serializer(object_=vocab, path=path_to_deseralize, serialized_name='vocabulary')
    del vocab

    return s_o_inverted_index, num_of_entities, predicate_mapper

def construct_dataset_for_relation_prediction(embeddings, kg_path, storage_path):
    s_o_inverted_index, num_of_resources, predicate_mapper = construct_subject_object_inverted_index(kg_path,
                                                                                                     storage_path)

    vocab = deserializer(path=storage_path, serialized_name='vocabulary')

    inverse_vocab = np.array(list(vocab.keys()))
    # inverse_predicate_mapper = np.array(list(predicate_mapper.keys()))

    num_of_predicates = len(predicate_mapper)  # correspond to number of labels as well
    del vocab, predicate_mapper

    X = []
    y_row = []  # construct representation.
    y_col = []
    for ith, t in enumerate(s_o_inverted_index.items()):
        i_s_o, i_predicates = t
        i_s, i_o = i_s_o

        for i_p in i_predicates:  # predicates have own indexes
            y_row.append(ith)
            y_col.append(i_p)


        emb_s=embeddings.loc[inverse_vocab[i_s]]
        emb_o=embeddings.loc[inverse_vocab[i_o]]

        x=emb_s.append(emb_o)
        X.append(x)

    del s_o_inverted_index

    X = np.array(X)

    y = csr_matrix((np.ones(len(y_row)), (np.array(y_row), np.array(y_col))),
                   shape=(len(X), num_of_predicates), dtype=np.uint16)

    return X, y

def eval_h_at_N_sparse(path, model, inverse_output_mapper, logger, vocab=None, embeddings=None):
    hit_at_1 = []
    hit_at_3 = []
    hit_at_5 = []
    hit_at_10 = []

    logger.info('Evaluation starts on {0}'.format(path))

    num_of_entitiy_not_seen_training = 0

    with open(path, 'r') as reader:

        for index, sentence in enumerate(reader):
            # components = re.findall('<(.+?)>', sentence)
            # s, p, o = components[0], components[1], components[2]
            s, p, o = sentence.split()
            if vocab:
                try:
                    i_s, i_o = vocab[s], vocab[o]
                    input_ = np.array([i_s, i_o]).reshape(1, 2)
                except:
                    num_of_entitiy_not_seen_training += 1
                    continue
            else:
                try:
                    input_ = embeddings.loc[s].append(embeddings.loc[o]).to_numpy()
                except:
                    num_of_entitiy_not_seen_training += 1
                    continue

                input_ = input_.reshape(1, len(input_))

            predictions = model.predict(input_)[0]

            idx_of_tops_ = predictions.argsort()[::-1]  # Total time complexity: |G^Testing| |Relations|

            h1 = inverse_output_mapper[idx_of_tops_[0]]
            h3 = inverse_output_mapper[idx_of_tops_[:2]]
            h5 = inverse_output_mapper[idx_of_tops_[:4]]
            h10 = inverse_output_mapper[idx_of_tops_[:9]]

            hit_at_1.append(1 if p in h1 else 0)
            hit_at_3.append(1 if p in h3 else 0)
            hit_at_5.append(1 if p in h5 else 0)
            hit_at_10.append(1 if p in h10 else 0)

            if index % 500 == 0:
                if index > 0:
                    logger.info('###')
                    logger.info('{0}th test triple: HIT@1 {1}'.format(index, stats.describe(hit_at_1)))
                    logger.info('{0}th test triple: HIT@3 {1}'.format(index, stats.describe(hit_at_3)))
                    logger.info('{0}th test triple: HIT@5 {1}'.format(index, stats.describe(hit_at_5)))
                    logger.info('{0}th test triple: HIT@10 {1}'.format(index, stats.describe(hit_at_10)))
                    logger.info('###')
        logger.info('Number of triples in testing:{0}'.format(index + 1))

    return stats.describe(hit_at_1), stats.describe(hit_at_3), stats.describe(hit_at_5), stats.describe(hit_at_10)