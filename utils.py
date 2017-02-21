import csv
from os import path


def load_csv(fname):
    with open(fname) as csvfile:
        reader = csv.DictReader(csvfile)
        yield from reader


def fnc_1_data(datadir='./data/'):
    bodies = load_csv(path.join(datadir, 'train_bodies.csv'))
    bodiesLookup = {b['Body ID']:b['articleBody'] for b in bodies}
    articles = list(load_csv(path.join(datadir, 'train_stances.csv')))
    for article in articles:
        body_id = article['Body ID']
        article['articleBody'] = bodiesLookup[body_id]
    return articles
