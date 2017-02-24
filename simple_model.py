from utils import fnc_1_data

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from pymicha.keras.plots import plot_confusion_matrix
from pymicha.utils import timer
import pylab as py

from utils import score_submission

import random


def get_headline(articles):
    return [a['Headline'] for a in articles]


def get_body(articles):
    return [a['articleBody'] for a in articles]


@timer
def run_pipeline(name, classifier, ngram_range):
    print("Running pipeline:", name)
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('headline', Pipeline([
                ('get_headline', FunctionTransformer(get_headline,
                                                     validate=False)),
                ('tfidf_headline', TfidfVectorizer(stop_words='english',
                                                   ngram_range=ngram_range,
                                                   max_features=1024))
            ])),
            ('body', Pipeline([
                ('geat_body', FunctionTransformer(get_body,
                                                  validate=False)),
                ('tdidf_body', TfidfVectorizer(stop_words='english',
                                               ngram_range=ngram_range,
                                               max_features=1024))
            ]))
        ])),
        ('classifier', classifier),
    ])
    pipeline.fit(articles[N:], targets[N:])
    result = pipeline.predict_proba(articles[:N])

    y_valid = targets[:N]
    y_valid_pred = result.argmax(axis=1)

    cm = metrics.confusion_matrix(y_valid, y_valid_pred)
    score = score_submission(y_valid, y_valid_pred, labels)
    score /= len(y_valid)
    max_score = score_submission(y_valid, y_valid, labels)
    max_score /= len(y_valid)
    print("Score: ", score, max_score)

    accuracy = (y_valid == y_valid_pred).sum() / len(y_valid)
    print("Accuracy: ", accuracy)

    py.clf()
    plot_confusion_matrix(
        cm,
        labels,
        title=("Submission Score: {:0.4f} / {:0.4f}: {:0.2f}%"
               .format(score, max_score, 100 * score / max_score)),
        normalize=True
    )
    py.savefig("images/simple_model_confusion_{}.png".format(name))


if __name__ == "__main__":
    articles = fnc_1_data('./data/')
    random.shuffle(articles)
    N = int(len(articles) * 0.1)

    labels = list({a['Stance'] for a in articles})
    targets = [labels.index(a['Stance']) for a in articles]

    run_pipeline("logistic_regression", LogisticRegression(C=0.95, class_weight='balanced'), (1, 2))
    run_pipeline("adaboost", AdaBoostClassifier(n_estimators=256), (1, 2))
