# -*- coding: utf-8 -*-
import json

from gensim.models import Word2Vec

from bionev.OpenNE import walker
import bionev.OpenNE.graph as og

from ast import literal_eval
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
        self.vectors = {}
        if dw:
            self.walker = walker.Walker(
                graph, p=p, q=q, update=False, workers=kwargs["workers"])
            sentences = self.walker.simulate_walks(
                num_walks=self.num_paths, walk_length=self.path_length, vectors=None)
        else:
            self.walker = walker.Walker(
                graph, p=p, q=q, update=False, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
            sentences = self.walker.simulate_walks(
                num_walks=self.num_paths, walk_length=self.path_length, vectors=self.vectors)

        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
        print("Learning representation...")
        self.word2vec = Word2Vec(**kwargs)
        for word in graph.G.nodes():
            self.vectors[word] = self.word2vec.wv[word]

    def update_model(self, graph, alias_edges_path=None):
        self.walker.update = True
        self.walker.G = graph.G
        print("Preprocess transition probs...")
        if alias_edges_path is not None:
            with open(alias_edges_path, 'r') as f:
                obj = json.load(f)
                self.walker.alias_edges = {literal_eval(k): literal_eval(v) for k, v in obj.items()}
        self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(
            num_walks=self.num_paths, walk_length=self.path_length, vectors=self.vectors)
        self.word2vec.build_vocab(sentences=sentences, update=True)
        for word in graph.G.nodes():
            if word in self.vectors.keys():
                continue
            self.vectors[word] = self.word2vec.wv[word]

    def get_embeddings(self):
        return self.vectors

    def save_model(self, model_path, alias_edges_path=None):
        if alias_edges_path is not None:
            with open(alias_edges_path, 'w') as f:
                json.dump({str(k): v.tolist() for k, v in self.walker.alias_edges.items()}, f)
            self.walker.alias_edges.clear()
        joblib.dump(self, model_path)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
