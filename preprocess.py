from skipthoughts import skipthoughts as st
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
import numpy as np
import h5py
import nltk

import utils

import pickle
from functools import partial
import itertools as IT


SKIPTHOUGHTS_DATA = '/home/micha/ssd/work/tldr/skipthoughts/data/'
memory = Memory(cachedir='./cache/', verbose=1)
st.encode = memory.cache(st.encode, ignore=['model'], verbose=0)


@memory.cache
def semantic_similarity_feature(articles):
    num_article_sent = lambda a: len(a['article_vectors'])  # NOQA
    articles.sort(key=num_article_sent)
    target_lookup = list({article['Stance'] for article in articles})
    dataset = h5py.File("./data/dataset.hdf5", "w")
    dataset.attrs['stance_lookup'] = np.asarray(target_lookup).astype('|S9')
    group_names = []
    for num_sent, articles in IT.groupby(tqdm(articles, 'sem sim'),
                                         num_article_sent):
        articles = list(articles)
        num_articles = len(articles)
        sample_vec = articles[0]['headline_vector']
        vector_length = sample_vec.shape[0]
        vector_group = dataset.create_dataset(
            "features/num_sent_{}".format(num_sent),
            (num_articles, num_sent, 2*vector_length),
            dtype=sample_vec.dtype
        )
        lookup_group = dataset.create_dataset(
            "article_lookup/num_sent_{}".format(num_sent),
            (num_articles,),
            dtype='uint8'
        )
        target_group = dataset.create_dataset(
            "targets/num_sent_{}".format(num_sent),
            (num_articles, len(target_lookup)),
            dtype='uint8',
        )
        group_names.append('num_sent_{}'.format(num_sent))
        for a, article in enumerate(tqdm(articles, 'sem sim batch')):
            headline_vec = article['headline_vector']
            target_idx = target_lookup.index(article['Stance'])
            target_group[a, target_idx] = 1
            lookup_group[a] = int(article['Body ID'])
            article['article_batch_location'] = (num_sent, a)
            for s, sent_vector in enumerate(article['article_vectors']):
                vec = np.hstack((np.abs(headline_vec - sent_vector),
                                 headline_vec * sent_vector))
                vector_group[a, s, :] = vec
            # save memory
            article.pop('headline_vector')
            article.pop('article_vectors')
    dataset.attrs['group_names'] = np.asarray(group_names).astype('|S{}'.format(max(map(len, group_names))))
    dataset.close()
    return article


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
        article.pop('headline_vectors')  # save memory
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
    articles = preprocess.transform(data)
    with open("data/articles.pkl", "wb+") as fd:
        pickle.dump(articles, fd)
