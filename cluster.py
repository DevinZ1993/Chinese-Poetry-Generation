#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from rank_words import get_stopwords
from data_utils import kw_train_path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score


def get_cluster_labels(texts, tokenizer, n_clusters):
    print "Clustering %d texts into %d groups ..." %(len(texts), n_clusters)
    vectorizer = CountVectorizer(tokenizer = tokenizer,
            stop_words = get_stopwords())
    transformer = TfidfTransformer()
    km = KMeans(n_clusters = n_clusters)

    tfidf = transformer.fit_transform(vectorizer.fit_transform(texts))
    km.fit(tfidf)
    return km.labels_.tolist()


def _eval_cluster(texts, tokenizer, n_clusters):
    vectorizer = CountVectorizer(tokenizer = tokenizer,
            stop_words = get_stopwords())
    transformer = TfidfTransformer()
    km = KMeans(n_clusters = n_clusters)
    tfidf = transformer.fit_transform(vectorizer.fit_transform(texts))
    km.fit(tfidf)
    return silhouette_score(tfidf,
            km.labels_.tolist(),
            sample_size = 1000)


if __name__ == '__main__':
    texts = []
    with codecs.open(kw_train_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            texts.append(line.strip())
            line = fin.readline()
    for n in range(2, 30):
        score = _eval_cluster(texts,
                tokenizer = lambda x: x.split('\t'),
                n_clusters = n)
        print "n_clusters = %d, score = %f" %(n, score)

