# _*_coding:utf-8_*_

import numpy as np
import pickle
from scipy.special import gammaln


def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)


def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx


class LDA(object):

    def __init__(self, corpus, word_id, num_topics, alpha=0.1, beta=0.1, max_iter=100):
        self.corpus = corpus
        self.word_id = word_id
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.num_doc, self.num_words = corpus.shape
        self.topics = {}                                       # 每个文档每个单词对应的主题
        self.nd = np.zeros((self.num_doc, self.num_topics))    # 每个文档中被指定为主题j的次数
        self.nw_sum = np.zeros(self.num_topics)                # 第j个主题下的单词数量
        self.nw = np.zeros((self.num_topics, self.num_words))  # 每一个文档中每个主题包含的单词数量
        self.nd_sum = np.zeros(self.num_doc)                   # 每一篇文档中单词的总数

        '''
        initial parameters
        '''
        for m in xrange(self.num_doc):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(self.corpus[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.num_topics)
                self.nd[m,z] += 1
                self.nd_sum[m] += 1
                self.nw[z,w] += 1
                self.nw_sum[z] += 1
                self.z[m][w] = z
                self.topics[(m, i)] = z

    def condition_proba(self, w_i, topic_k, d_i):
        """
        calculate conditional probability of Gibbs Sampling:  Non-vector methods
        """
        word_at_topic_proba = (self.nw[topic_k][w_i] + self.beta) / \
                              (np.sum(self.nw[topic_k, :]) + self.num_words * self.beta)
        topic_at_doc_proba = (self.nd[d_i][topic_k] + self.alpha) / \
                             (np.sum(self.nd[d_i, :]) + self.num_topics * self.alpha)

        return word_at_topic_proba * topic_at_doc_proba

    def conditional_distribution(self, m, w):
        """
        calculate conditional probability of Gibbs Sampling:  vector methods
        """
        vocab_size = self.num_words
        left = (self.nw[:, w] + self.beta) / \
               (self.nw_sum + self.beta * vocab_size)
        right = (self.nd[m,:] + self.alpha) / \
                (self.nd_sum[m] + self.alpha * self.num_topics)

        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def condition_proba_predict(self, w, nw, nw_sum, nd, nd_sum):
        """
        calculate conditional probability of Gibbs Sampling(Prediction):  vector methods
        """
        word_at_topic_proba = (self.nw[:, w] + nw[:, w] + self.beta) / \
                              (self.nw_sum + nw_sum + self.num_words * self.beta)
        topic_at_doc_proba = (nd + self.alpha) / (nd_sum + self.num_topics * self.alpha)

        proba = word_at_topic_proba * topic_at_doc_proba
        proba /= np.sum(proba)

        return proba

    def random_choice(self, proba):
        """
        Random choice of multinomial Distribution
        """
        u = np.random.random()
        for i in range(1, self.num_topics):
            proba[i] += proba[i-1]
        index = 0
        for index in range(self.num_topics):
            if proba[index] > u:
                break
        return index

    def log_likelihood(self):
        """
        Compute the likelihood that the model generated the data.
        calculated by joint probability distribution P(W|a,b)
        """
        vocab_size = self.num_words
        n_docs = self.num_doc
        lik = 0

        for z in xrange(self.num_topics):
            lik += log_multi_beta(self.nw[z, :]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nd[m, :]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.num_topics)

        return lik

    def parameter_theta(self):
        """
        calculate theta parameters which indicate topic-probability in each training doc: doc-->topic
        """
        theta = np.zeros((self.num_doc, self.num_topics))
        tmp = self.nd + self.alpha
        for d in range(self.num_doc):
            theta[d, :] = (tmp[d, :]) / np.sum(tmp[d, :])

        return theta

    def parameter_phi(self):
        """
        calculate phi parameters which indicate word-probability in each topic: topic-->words
        """
        phi = np.zeros((self.num_topics, self.num_words))
        tmp = self.nw + self.beta
        for z in range(self.num_topics):
            phi[z, :] = (tmp[z, :]) / np.sum(tmp[z, :])

        return phi

    def train(self):
        """
        training LDA model
        """
        for iteration in range(self.max_iter):
            print 'iteration', iteration
            for d in range(self.num_doc):
                for i, w in enumerate(word_indices(self.corpus[d, :])):
                    topic_present = self.topics[(d, i)]
                    w_i = self.word_id[w]
                    self.nw[topic_present][w_i] -= 1
                    self.nw_sum[topic_present] -= 1
                    self.nd[d][topic_present] -= 1
                    self.nd_sum[d] -= 1

                    proba = np.zeros(self.num_topics)
                    for k in range(self.num_topics):
                        proba[k] = self.condition_proba(w_i, k, d)

                    proba /= np.sum(proba)

                    topic_new = self.random_choice(proba)

                    self.nw[topic_new][w_i] += 1
                    self.nw_sum[topic_new] += 1
                    self.nd[d][topic_new] += 1
                    self.nd_sum[d] += 1
                    self.topics[(d, i)] = topic_new

            print self.log_likelihood()

    def predict(self, matrix, max_iter=100):
        """
        make prediction for new document: each prediction with one document
        """
        new_nw = np.zeros((self.num_topics, self.num_words))
        new_nwsum = np.zeros(self.num_topics)
        new_nd = np.zeros(self.num_topics)
        new_ndsum = 0
        topics = {}

        for i, w in enumerate(word_indices(matrix)):
            # choose an arbitrary topic as first topic for word i
            z = np.random.randint(self.num_topics)
            new_nd[z] += 1
            new_ndsum += 1
            new_nw[z, w] += 1
            new_nwsum[z] += 1
            topics[i] = z

        for iteration in range(max_iter):
            for i, w in enumerate(word_indices(matrix)):
                topic_present = topics[i]
                new_nw[topic_present][w] -= 1
                new_nwsum[topic_present] -= 1
                new_nd[topic_present] -= 1

                proba = self.condition_proba_predict(w, new_nw, new_nwsum, new_nd, new_ndsum)
                topic_new = self.random_choice(proba)

                new_nw[topic_new][w] += 1
                new_nwsum[topic_new] += 1
                new_nd[topic_new] += 1
                topics[i] = topic_new


if __name__ == '__main__':
    corpus = pickle.load(open(u'F:/算法学习主题/LDA/document/term_doc_matrix.txt', 'r'))
    word_id = {}
    for i in range(corpus.shape[1]):
        word_id[i] = i
    num_topic = 20
    alpha = 0.1
    beta = 0.1
    max_iter = 10
    lda = LDA(corpus, word_id, num_topic, alpha, beta, max_iter)
    lda.train()
    theta = lda.parameter_phi()
    lda.predict(corpus[1, :])
    print lda.nd[1]

    # import pandas as pd
    # pd.DataFrame(theta).to_csv(u'F:/算法学习主题/LDA/document/theta.csv', index=False)