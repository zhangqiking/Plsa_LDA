# _*_coding:utf-8_*_

import numpy as np
import pickle
import math
import random
from scipy.special import gamma, psi
from lda_alpha import *
from operator import itemgetter


def word_indices(vec):
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx


class EM(object):
    def __init__(self, corpus, num_topic):
        self.corpus = corpus
        self.num_doc = corpus.shape[0]
        self.num_words = corpus.shape[1]
        self.num_topic = num_topic
        self.var_gamma = np.zeros((self.num_doc, self.num_topic))
        self.phi = np.zeros((self.num_words, self.num_topic))
        self.alpha = 5

        self.class_word = np.zeros((self.num_topic, self.num_words))
        self.class_total = np.zeros(num_topic)
        self.log_prob_w = np.zeros((self.num_topic, self.num_words))
        self.alpha_suffstats = 0.

    def corpus_init(self):
        """
        initialize corpus statics
        """
        num_init = 1
        # seen用于保存已经选取过的doc号，该方式必须保证doc数量大于topic数量
        # seen = np.zeros((self.num_topic, num_init))

        '''
        for k in range(self.num_topic):
            for i in range(num_init):
                """
                while selected:
                    d = np.random.randint(0, self.num_doc)
                    if d not in seen[:k, i]:
                        break
                seen[k][i] = d
                """
                # 随机选取一篇doc
                d = np.random.randint(0, self.num_doc)
                for j, w in enumerate(word_indices(self.corpus[d, :])):
                    self.class_word[k][w] += 1

            # 平滑smoothed，为每个词的topic-word的统计量都初始化为1
            for n in range(self.num_words):
                self.class_word[k][n] += 1
                self.class_total[k] += self.class_word[k][n]
        '''
        self.init_model()
        print self.class_total
        print self.class_word
        self.update_beta()
        print self.log_prob_w

    def init_model(self):
        for z in range(0, self.num_topic):
            for w in range(0, self.num_words):
                self.class_word[z, w] += 1.0 / self.num_words + random.random()
                self.class_total[z] += self.class_word[z, w]

    def zero_initialize(self):
        """
        zero initialize of sufficient statistics in each EM iteration
        :return:
        """
        for k in range(self.num_topic):
            self.class_total[k] = 0
            for w in range(self.num_words):
                self.class_word[k, w] = 0
        self.alpha_suffstats = 0

    def update_beta(self):
        for k in range(self.num_topic):
            for w in range(self.num_words):
                if self.class_word[k][w] > 0:
                    self.log_prob_w[k][w] = math.log(self.class_word[k][w]) - math.log(self.class_total[k])
                else:
                    print "sdfv"
                    self.log_prob_w[k][w] = -100

    def doc_e_step(self, d, phi):
        """
        :param d:
        :param phi:
        :return:
        """
        # posterior inference
        likelihood = self.lda_inference(d, phi)

        # update sufficient statistics
        gamma_sum = 0.
        for k in range(self.num_topic):
            gamma_sum += self.var_gamma[d][k]
            self.alpha_suffstats += psi(self.var_gamma[d][k])
        self.alpha_suffstats -= self.num_topic * psi(gamma_sum)

        for i, w in enumerate(word_indices(self.corpus[d, :])):
            for k in range(self.num_topic):
                self.class_word[k][w] += phi[w][k]
                self.class_total[k] += phi[w][k]

        return likelihood

    def doc_m_step(self, estimate_alpha = True):
        """
        update beta and alpha parameters
        :return: None
        """
        self.update_beta()
        if estimate_alpha:
            self.alpha = opt_alpha(self.alpha_suffstats, self.num_doc, self.num_topic)
        else:
            pass
        print "new alpha = ", self.alpha

    def lda_inference(self, d, phi):
        """
        lda inference of each doc
        """
        old_phi = np.zeros(self.num_topic)
        digamma_gam = np.zeros(self.num_topic)
        likelihood = 0.
        old_likelihood = 0.

        # computer posterior dirichlet
        for k in range(self.num_topic):
            self.var_gamma[d][k] = self.alpha + np.sum(self.corpus[d, :]) * 1.0 / self.num_topic
            digamma_gam[k] = psi(self.var_gamma[d][k])
            for i, w in enumerate(word_indices(self.corpus[d, :])):
                phi[w][k] = 1.0 / self.num_topic

        for iteration in range(30):
            for n, w in enumerate(word_indices(self.corpus[d, :])):
                phisum = 0
                for k in range(self.num_topic):
                    old_phi[k] = phi[w][k]
                    phi[w][k] = digamma_gam[k] + self.log_prob_w[k][w]
                    if k > 0:
                        phisum = math.log(math.exp(phisum) + math.exp(phi[w, k]))
                    else:
                        phisum = phi[w][k]

                for k in range(self.num_topic):
                    phi[w][k] = math.exp(phi[w][k] - phisum)
                    self.var_gamma[d][k] = self.var_gamma[d][k] + self.corpus[d][w] * (phi[w][k] - old_phi[k])
                    digamma_gam[k] = psi(self.var_gamma[d][k])

            likelihood = self.compute_likelihood(phi, d)
            # converged = (old_likelihood - likelihood) / old_likelihood
            old_likelihood = likelihood

        return likelihood

    def compute_likelihood(self, phi, d):
        """
        ELOB lower bound refer to likelihood function
        :param phi:
        :param d:
        :return:
        """
        var_gamma_sum = 0.
        dig = np.zeros(self.num_topic)
        for k in range(self.num_topic):
            dig[k] = psi(self.var_gamma[d, k])
            var_gamma_sum += self.var_gamma[d, k]
        dig_sum = psi(var_gamma_sum)
        likelihood = log_gamma(self.num_topic * self.alpha) - \
                     self.num_topic * log_gamma(self.alpha) - \
                     log_gamma(var_gamma_sum)

        for k in range(self.num_topic):
            likelihood += (self.alpha - 1) * (dig[k] - dig_sum) + \
                          log_gamma(self.var_gamma[d][k]) - \
                          (self.var_gamma[d][k] - 1) * (dig[k] - dig_sum)

            for i, w in enumerate(word_indices(self.corpus[d, :])):
                if phi[w][k] > 0:
                    likelihood += phi[w][k] * (dig[k] - dig_sum) + \
                                  phi[w][k] * self.log_prob_w[k][w] - \
                                  phi[w][k] * math.log(phi[w][k])

        return likelihood

    def run_em(self, max_iter=4):
        """
        EM algorithm iteration with E-step and M-step
        :return: None
        """
        phi = np.zeros((self.num_words, self.num_topic))
        self.corpus_init()
        for iteration in range(max_iter):
            print("**** EM iteration: %d ****" % (iteration))
            likelihood = 0.
            self.zero_initialize()
            for d in range(self.num_doc):
                if (d % 1) == 0:
                    print("**** document: %d****" % (d))
                # E-step
                likelihood += self.doc_e_step(d, phi)
            print("likelihood: %f" % (likelihood))

            # M-step
            self.doc_m_step()

    def document_topic_distribution(self, filepath, topk):
        f = open(filepath, "w")
        for d in range(self.num_doc):
            topic_index_prob = []
            for z in range(self.num_topic):
                topic_index_prob.append([z, self.var_gamma[d][z]])
            topic_index_prob = sorted(topic_index_prob, key=itemgetter(1), reverse=True)
            f.write("Document #" + str(d) + ":\n")
            for i in range(topk):
                index = topic_index_prob[i][0]
                f.write("topic" + str(index) + " ")
            f.write("\n")

        f.close()

    def topic_word_distribution(self, topk, filepath, vocabulary):
        """
        Print topic-word distribution to file and list @topk most probable words for each topic
        """
        print "Writing topic-word distribution to file: " + filepath
        f = open(filepath, "w")
        for z in range(self.num_topic):
            word_prob = self.log_prob_w[z]
            word_index_prob = []
            for i in range(self.num_words):
                word_index_prob.append([i, word_prob[i]])
            word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True)  # sort by word count
            f.write("Topic #" + str(z) + ":\n")
            for i in range(topk):
                index = word_index_prob[i][0]
                f.write(vocabulary[index] + " ")
            f.write("\n")

        f.close()


if __name__ == '__main__':
    corpus = pickle.load(open(u'F:/算法学习主题/LDA/document/term_doc_matrix.txt', 'r'))
    vocab = pickle.load(open(u'F:/算法学习主题/LDA/document/vocabulary', 'r'))
    lda = EM(corpus, 20)
    lda.run_em()
    print lda.var_gamma
    print lda.log_prob_w
    lda.document_topic_distribution("./document/result/topic-word_LDA.txt", 10)
    lda.topic_word_distribution(10, "./document/result/document-topic_LDA.txt",vocab)
    # lda.corpus_init()
    # print lda.num_doc
    # print lda.class_word
    # print lda.class_total
    # phi = np.zeros((lda.num_words, lda.num_topic))
    # lda.lda_inference(0, phi)





