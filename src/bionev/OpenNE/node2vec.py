# -*- coding: utf-8 -*-

from gensim.models import Word2Vec

from bionev.OpenNE import walker
import bionev.OpenNE.graph as og

import joblib

class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):

        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0
        self.path_length = path_length
        self.num_paths = num_paths
        self.graph = graph
        self.vectors = {}
        if dw:
            self.walker = walker.BasicWalker(graph, workers=kwargs["workers"])
        else:
            self.walker = walker.Walker(
                graph, p=p, q=q, update=False, vectors=self.vectors, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()

        sentences = self.walker.simulate_walks(
            num_walks=self.num_paths, walk_length=self.path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
        print("Learning representation...")
        self.word2vec = Word2Vec(**kwargs)
        for word in self.graph.G.nodes():
            self.vectors[word] = self.word2vec.wv[word]

    def update_node2vec(self, graph):
        self.graph = og.Graph()
        self.graph.read_edgelist(graph, weighted=False)
        self.walker.update = True
        self.walker.G = self.graph.G
        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(
            num_walks=self.num_paths, walk_length=self.path_length)
        self.word2vec.build_vocab(sentences=sentences, update=True)
        for word in self.graph.G.nodes():
            if word in self.vectors.keys():
                continue
            self.vectors[word] = self.word2vec.wv[word]

    def get_embeddings(self):
        return self.vectors

    def save_model(self, path):
        joblib.dump(self, path)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
