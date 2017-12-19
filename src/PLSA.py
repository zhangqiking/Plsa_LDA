# _*_coding:utf-8_*_

import numpy as np
from operator import itemgetter

def uniform(row, column):
    tmp1 = np.random.random(size=(row, column))
    for i in range(row):
        tmp1[i] = tmp1[i]/sum(tmp1[i])
    return tmp1


class PLSA(object):
    def __init__(self, corpus, num_topic, num_doc, num_words, num_per_doc):
        self.corpus = corpus
        self.num_doc = num_doc
        self.num_words = num_words
        self.num_topic = num_topic
        self.z_dw = {key: uniform(1,self.num_topic) for key in self.corpus.keys()}
        self.w_z = {key: uniform(1, self.num_words) for key in range(self.num_topic)}
        self.z_d = {key: uniform(1, self.num_topic) for key in range(self.num_doc)}
        self.p_d = np.array([1.0/self.num_doc for i in range(self.num_doc)])
        self.words = list(set([key[1] for key in self.z_dw.keys()]))
        self.doc = [i for i in range(self.num_doc)]
        self.num_per_doc = num_per_doc
        self.p_dw = {}

    def e_step(self):
        # print self.w_z
        # print self.z_d
        for key in self.z_dw.keys():
            denominator = 0
            for z in range(self.num_topic):
                d_i = key[0]
                w_j = key[1]
                denominator += self.w_z[z][0, w_j] * self.z_d[d_i][0, z]
            for z_k in range(self.num_topic):
                d_i = key[0]
                w_j = key[1]
                numerator = self.w_z[z_k][0, w_j] * self.z_d[d_i][0, z_k]

                self.z_dw[key][0, z_k] = numerator / denominator

        # print self.z_dw

    def m_step(self):
        for z in range(self.num_topic):
            denominator = 0
            for i in range(self.num_doc):
                for m in range(self.num_words):
                    d_i = self.doc[i]
                    w_m = self.words[m]
                    denominator += self.corpus[(d_i, w_m)] * self.z_dw[(d_i, w_m)][0, z]
            for j in range(self.num_words):
                numerator = 0
                w_j = self.words[j]
                for i in range(self.num_doc):
                    d_i = self.doc[i]
                    numerator += self.corpus[(d_i, w_j)] * self.z_dw[(d_i, w_j)][0, z]
                self.w_z[z][0, j] = numerator / denominator
        print np.sum(self.w_z[0][0])

        for i in range(self.num_doc):
            n_di = self.num_per_doc[i]
            d_i = i
            for z in range(self.num_topic):
                z_k = z
                numerator = 0
                for j in range(self.num_words):
                    w_j = self.words[j]
                    numerator += self.corpus[(d_i, w_j)] * self.z_dw[(d_i, w_j)][0, z_k]
                self.z_d[d_i][0, z_k] = numerator / n_di

    def calculate_p_dw(self):
        for key in self.z_dw.keys():
            p_wz = 0
            d = key[0]
            w = key[1]
            for z in range(self.num_topic):
                p_wz += self.w_z[z][0, w] * self.z_d[d][0, z]
            self.p_dw[key] = p_wz

    def likehood_log(self):
        likehood = 0.
        t = 0.
        for i in range(self.num_doc):
            d_i = i
            tmp1 = 0.
            for j in range(self.num_words):
                w_j = j
                tmp2 = float(self.corpus[(d_i, w_j)]) / self.num_per_doc[d_i]
                tmp3 = 0.
                for k in range(self.num_topic):
                    z_k = k
                    tmp3 += self.z_d[d_i][0, z_k] * self.w_z[z_k][0, w_j]
                tmp1 += tmp2 * np.log(tmp3)
                # print tmp1
            likehood += self.num_per_doc[d_i]*(np.log(self.p_d[d_i]) + tmp1)
        # print t
        return likehood

    def likehood_log2(self):
        l_likehood = 0.
        for i in range(self.num_doc):
            d_i = i
            for j in range(self.num_words):
                w_j = j
                l_likehood += self.corpus[(d_i, w_j)] * np.log(self.p_dw[(d_i, w_j)])
        return l_likehood

    def train(self, thershold=0.1, max_iter=60):
        iterator = 0
        likehood = float('inf')
        while iterator < max_iter:
            self.e_step()
            self.m_step()
            self.calculate_p_dw()
            new_likehood = self.likehood_log2()
            print new_likehood
            if new_likehood < likehood:
                likehood = new_likehood
            # print likehood
            iterator += 1

    def document_topic_distribution(self, filepath, topk):
        f = open(filepath, "w")
        for d in range(self.num_doc):
            topic_index_prob = []
            for z in range(self.num_topic):
                topic_index_prob.append([z, self.z_d[d][0, z]])
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
            word_prob = self.w_z[z]
            word_index_prob = []
            for i in range(self.num_words):
                word_index_prob.append([i, word_prob[0, i]])
            word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True)  # sort by word count
            f.write("Topic #" + str(z) + ":\n")
            for i in range(topk):
                index = word_index_prob[i][0]
                f.write(vocabulary[index] + " ")
            f.write("\n")

        f.close()


if __name__ == '__main__':
    cor = np.random.randint(0, 10, size=(50, 100))
    cor_dic = {}
    num_doc, num_words = cor.shape
    for i in range(num_doc):
        for j in range(num_words):
            cor_dic[(i, j)] = cor[i][j]
    num_pre_doc = np.sum(cor, axis=1)

    plsa = PLSA(cor_dic, 2, num_doc, num_words, num_pre_doc)
    plsa.train()

    # print plsa.z_dw[(0, 1)][1]
    # print plsa.z_d
    # print plsa.w_z
    print plsa.words
    print plsa.doc




