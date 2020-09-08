import datetime
import os
import pickle
#from tensorflow.keras import backend as K
import numpy as np
import time
import bz2
import re
triple = 3


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
