from skipthoughts import skipthoughts as st
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
import numpy as np
import nltk

import utils

from functools import partial
import itertools as IT


SKIPTHOUGHTS_DATA = '/home/micha/ssd/work/tldr/skipthoughts/data/'
memory = Memory(cachedir='./cache/', verbose=1)
st.encode = memory.cache(st.encode, ignore=['model'], verbose=0)


@memory.cache
def semantic_similarity_feature(articles):
    num_article_sent = lambda a: len(a['article_vectors'])  # NOQA
    articles.sort(key=num_article_sent)
    article_batches = []
    target_batches = []
    for num_sent, articles in IT.groupby(tqdm(articles, 'sem sim'),
                                         num_article_sent):
        articles = list(articles)
        num_articles = len(articles)
        sample_vec = articles[0]['headline_vector']
        vector_length = sample_vec.shape[0]
        vector_group = np.zeros((num_articles, num_sent, 2*vector_length),
                                dtype=sample_vec.dtype)
        target_group = []
        for a, article in enumerate(tqdm(articles, 'sem sim batch')):
            headline_vec = article['headline_vector']
            target_group.append(article['Stance'])
            for s, sent_vector in enumerate(article['article_vectors']):
                vec = np.hstack((np.abs(headline_vec - sent_vector),
                                 headline_vec * sent_vector))
                vector_group[a, s, :] = vec
        article_batches.append(vector_group)
        target_batches.append(target_group)
    return article_batches, target_batches


@memory.cache
def headline_vector_merge_mean(articles):
    """
    Some article titles have more than one sentence in the title. To have
    better comparison with the semantic similarity task for skipthoughts, we
    need this to be one vector. This extractor simply takes the mean vector of
    all headline vectors.
    """
    for article in tqdm(articles, 'merging headline vectors'):
        article['headline_vector'] = article['headline_vectors'].mean(axis=0)
    return articles


@memory.cache
def skipthoughts_articles(articles, max_title_sentences=None,
                          max_article_sentences=None):
    """
    Filter articles so that we have at max `max_title_sentences` sentences in
    the title and `max_article_sentences` sentences in the body of the article.

    Then, we add in the skipthought vectors for all sentences in the titles and
    bodies of the articles into the `headline_vectors` and `article_vectors`
    key.
    """
    article_vectors = []
    st_model = st.load_model(data_path=SKIPTHOUGHTS_DATA)
    for article in tqdm(articles, 'skipthoughts encoding articles'):
        title_sentences = nltk.sent_tokenize(article['Headline'])
        if max_title_sentences is not None and  \
                len(title_sentences) > max_title_sentences:
            continue
        article_sentences = nltk.sent_tokenize(article['articleBody'])
        if max_article_sentences is not None and \
                len(article_sentences) > max_article_sentences:
            continue
        vectors = st.encode(st_model, title_sentences + article_sentences,
                            verbose=False, batch_size=128).astype('float16')
        N = len(title_sentences)
        article['headline_vectors'] = vectors[:N]
        article['article_vectors'] = vectors[N:]
        article_vectors.append(article)
    return article_vectors


if __name__ == "__main__":
    data = utils.fnc_1_data('./data/')
    vectorize_args = dict(max_article_sentences=32)

    FT = partial(FunctionTransformer, validate=False)
    preprocess = Pipeline([
        ('vectorize', FT(skipthoughts_articles, kw_args=vectorize_args)),
        ('merge_headline', FT(headline_vector_merge_mean)),
        ('sem_sim_features', FT(semantic_similarity_feature)),
    ])
    features = preprocess.transform(data)
