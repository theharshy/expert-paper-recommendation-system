##
# System to assign a set of papers to Experts based on their own publications.
#

# Packages used for computation
from __future__ import print_function

import nltk
import re
import io
import operator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans

from collections import defaultdict


# Class Definition for a Paper
class Paper:
    def __init__(self, contents, author, index, expert):
        self.author = author
        self.index = index
        self.expert = expert
        self.contents = contents

    def assignindex(self, index):
        self.index = index


# Class Definition for an Expert with Paper Objects as attributes
class Expert:
    def __init__(self, paper, name):
        self.name = name
        self.papers = [paper]


# Class Description for the ExpertSystem
class ExpertSystem:
    def __init__(self, n_clusters, experts_per_paper):
        # Global Variables used
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stemmer = SnowballStemmer("english")
        self.rankings = defaultdict(dict)
        self.n_clusters = n_clusters
        self.experts_per_paper = experts_per_paper

    # Function to Process Text prior to TFIDF calculation
    def tokenize_and_stem(self, text):
        # first tokenize by sentence, then by word
        # Punctuation are caught as their own token
        tokens = [word for sent in nltk.sent_tokenize(text)
                  for word in nltk.word_tokenize(sent)
                  if word not in self.stopwords]
        filtered_tokens = []
        # filter out any tokens not containing letters
        # (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        # storing the stems of each term
        stems = [self.stemmer.stem(t) for t in filtered_tokens]

        return stems

    # Function to compute the TF_IDF matrix from the contents of all Papers
    def get_tfidf(self, contents):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.98, max_features=20000,
                                           min_df=0.05, stop_words='english',
                                           lowercase=True, use_idf=True,
                                           tokenizer=self.tokenize_and_stem,
                                           ngram_range=(1, 3))

        # fit the vectorizer to contents of the paper
        tfidf_matrix = tfidf_vectorizer.fit_transform(contents)

        print(tfidf_matrix.shape)
        return tfidf_matrix

    # Cluster the TF-IDF matrix using KMeans
    def get_kmeans(self, tfidf_matrix):
        km = KMeans(n_clusters=self.n_clusters)
        km.fit(tfidf_matrix)

        clusters = km.labels_.tolist()
        return clusters

    def get_cosine_sim(self, tfidf_matrix):
        similarity = cosine_similarity(tfidf_matrix)

        print(similarity.shape)
        return similarity

    # Extracting the most similar paper written by an Expert withnin a Cluster
    def get_expert_score(self, sim, expert, items):
        a = [sim[items[0], paper.index] for paper in expert.papers]

        if(a == []):
            return 0
        else:
            return max(a)

    # Determining the most appropriate Expert for the unassigned papers
    # Using a Semi Supervised Learning Algorithm
    def determine_paper_experts(self, lables, sim, Experts):
        i = 0

        indexed_lables = []
        for l in lables:
            indexed_lables.append((i, l))
            i = i + 1

        for cluster in range(0, self.n_clusters):
            for expert in Experts:
                self.rankings[cluster][expert.name] = 0

        for cluster in range(0, self.n_clusters):
            for expert in Experts:
                for items in list(filter((lambda x: x[1] == cluster),
                                  indexed_lables)):
                        self.rankings[cluster][expert.name] = \
                            self.rankings[cluster][expert.name] + \
                            (self.get_expert_score(sim, expert, items))
        return self.rankings

    # Finds Experts with respect to highest scores within similarity matrix
    def algorithm1(self, sim, Experts, research_papers, all_paper_objects):
        all_recommendation = []
        for paper in research_papers:
            indices = sim[paper.index].argsort()
            rev_indices = indices[::-1]
            recommend = []
            for index in rev_indices[1:]:
                if(all_paper_objects[index].expert is True
                   and ((all_paper_objects[index].author) not in
                        [name for (name, score) in recommend])):
                    item = (all_paper_objects[index].author,
                            sim[paper.index][index])
                    recommend.append(item)
                if(len(recommend) == self.experts_per_paper):
                    all_recommendation.append(recommend)
                    break

        return all_recommendation

    # Allocates Experts based on more prominent Authors within a observed field
    def algorithm2(self, lables, sim, Experts, research_papers):
        scores = self.determine_paper_experts(lables, sim, Experts)
        all_recommendation = []
        recommend = []
        for i in range(0, self.n_clusters):
            sorted_scores = sorted(scores[i].items(),
                                   key=operator.itemgetter(1), reverse=True)
            recommend.append(sorted_scores[0:(self.experts_per_paper - 1)])

        for paper in research_papers:
            paper_cluster = lables[paper.index]
            all_recommendation.append(recommend[paper_cluster])

        return all_recommendation
