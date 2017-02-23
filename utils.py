import numpy as np

import csv
from os import path


LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]


def load_csv(fname):
    with open(fname) as csvfile:
        reader = csv.DictReader(csvfile)
        yield from reader


def fnc_1_data(datadir='./data/'):
    bodies = load_csv(path.join(datadir, 'train_bodies.csv'))
    bodiesLookup = {b['Body ID']: b['articleBody'] for b in bodies}
    articles = list(load_csv(path.join(datadir, 'train_stances.csv')))
    for article in articles:
        body_id = article['Body ID']
        article['articleBody'] = bodiesLookup[body_id]
    return articles


def score_submission(gold_labels, test_labels, labels):
    score = 0.0
    related_labels = [i for i, l in enumerate(labels) if l in RELATED]
    unrelated = labels.index('unrelated')
    for i, (gold, test) in enumerate(zip(gold_labels, test_labels)):
        if gold == test:
            score += 0.25
            if gold != unrelated:
                score += 0.50
        if gold in related_labels and test in related_labels:
            score += 0.25
    return score


def score_submission_loss(labels):
    import keras.backend as K
    related_labels = np.asarray([i for i, l in enumerate(labels)
                                 if l in RELATED])
    unrelated = labels.index('unrelated')
    def objective(y_true_cat, y_pred_cat):
        y_true = K.argmax(y_true_cat, axis=1)
        y_pred = K.argmax(y_pred_cat, axis=1)
        correct = K.max(y_true_cat * y_pred_cat, axis=1)
        is_unrelated = K.not_equal(y_true, unrelated)
        is_related = (K.any([K.equal(y_true, r) for r in related_labels]) and
                      K.any([K.equal(y_pred, r) for r in related_labels]))
        score = correct * (0.25 + is_unrelated*0.5) + is_related*0.25
        return 1.0 / (score + K.epsilon())
    return objective

