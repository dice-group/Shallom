import json
from util import *
from helper_classes import *
from collections import defaultdict
from models import Shallom
import argparse
import traceback
import sys
from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="KGs/FB15k-237", nargs="?",
                        help="Which dataset to use with KGs/XXX: FB15k,WN18,WN18RR, FB15k-237.")
    parser.add_argument("--embedding_dim", type=int, default=50, nargs="?",
                        help="Number of dimensions in embedding space.")
    parser.add_argument("--num_of_epochs", type=int, default=100, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=1000, nargs="?",
                        help="Batch size.")
    parser.add_argument("--input_dropout", type=float, default=0.5, nargs="?",
                        help="Dropout rate for concatenated embeddings.")
    parser.add_argument("--hidden_dropout", type=float, default=0.5, nargs="?",
                        help="Dropout rate for composite embeddings.")
    parser.add_argument("--hidden_width_rate", type=int, default=3, nargs="?",
                        help="How many times wider should be the hidden layer than embeddings.")
    parser.add_argument("--L2reg", type=float, default=.1, nargs="?",help="L2.")

    args = parser.parse_args()
    setting = {
        'model_name': 'Shallom',
        'embedding_dim': args.embedding_dim,
        'epochs': args.num_of_epochs,
        'batch_size': args.batch_size,
        'input_dropout': args.input_dropout,
        'hidden_dropout': args.hidden_dropout,
        'hidden_width_rate': args.hidden_width_rate,
        'reg':args.L2reg}

    seed = 1
    np.random.seed(seed)

    kg_name = args.dataset[args.dataset.rfind('/') + 1:]
    print('Dataset being processed.')
    dataset = Data(data_dir="%s/" % args.dataset)
    experiment = Experiment(dataset, setting)
    experiment.train_and_eval()
