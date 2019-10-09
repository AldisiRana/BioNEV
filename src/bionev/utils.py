# -*- coding: utf-8 -*-

import copy
import itertools
import random

import networkx as nx
import numpy as np

import bionev.OpenNE.graph as og
import bionev.struc2vec.graph as sg


def read_for_OpenNE(filename, weighted=False):
    graph = og.Graph()
    print("Loading training graph for learning embedding...")
    graph.read_edgelist(filename=filename, weighted=weighted)
    print("Graph Loaded...")
    return graph


def read_for_struc2vec(filename):
    print("Loading training graph for learning embedding...")
    graph = sg.load_edgelist(filename, undirected=True)
    print("Graph Loaded...")
    return graph


def read_for_gae(filename, weighted=False):
    print("Loading training graph for learning embedding...")
    edgelist = np.loadtxt(filename, dtype='float')
    if weighted:
        edgelist = [(int(edgelist[idx, 0]), int(edgelist[idx, 1])) for idx in range(edgelist.shape[0]) if
                    edgelist[idx, 2] > 0]
    else:
        edgelist = [(int(edgelist[idx, 0]), int(edgelist[idx, 1])) for idx in range(edgelist.shape[0])]
    min_idx = min([x[0] for x in edgelist] + [x[1] for x in edgelist])
    max_idx = max([x[0] for x in edgelist] + [x[1] for x in edgelist])
    adj = nx.adjacency_matrix(nx.from_edgelist(edgelist), nodelist=list(range(min_idx, max_idx + 1)))
    print(adj)
    print("Graph Loaded...")
    print(adj.shape)
    return adj


def read_for_SVD(filename, weighted=False):
    if weighted:
        graph = nx.read_weighted_edgelist(filename)
    else:
        graph = nx.read_edgelist(filename)
    return graph


def train_test_graph(input_edgelist, training_edgelist, testing_edgelist, weighted=False):
    if weighted:
        graph = nx.read_weighted_edgelist(input_edgelist)
        g_train = nx.read_weighted_edgelist(training_edgelist)
        g_test = nx.read_weighted_edgelist(testing_edgelist)
    else:
        graph = nx.read_edgelist(input_edgelist)
        g_train = nx.read_edgelist(training_edgelist)
        g_test = nx.read_edgelist(testing_edgelist)
    testing_pos_edges = g_test.edges
    node_num1, edge_num1 = len(g_train.nodes), len(g_train.edges)
    print('Training Graph: nodes:', node_num1, 'edges:', edge_num1)
    return graph, g_train, testing_pos_edges, training_edgelist


def split_train_test_graph(*, input_edgelist, testing_ratio=0.2, weighted=False):
    if weighted:
        graph = nx.read_weighted_edgelist(input_edgelist)
    else:
        graph = nx.read_edgelist(input_edgelist)
    node_num1, edge_num1 = len(graph.nodes), len(graph.edges)
    print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
    testing_edges_num = int(len(graph.edges) * testing_ratio)
    testing_pos_edges = random.sample(graph.edges, testing_edges_num)
    g_train = copy.deepcopy(graph)
    for edge in testing_pos_edges:
        node_u, node_v = edge
        if g_train.degree(node_u) > 1 and g_train.degree(node_v) > 1:
            g_train.remove_edge(node_u, node_v)

    train_graph_filename = 'graph_train.edgelist'
    if weighted:
        nx.write_edgelist(g_train, train_graph_filename, data=['weight'])
    else:
        nx.write_edgelist(g_train, train_graph_filename, data=False)

    node_num1, edge_num1 = len(g_train.nodes), len(g_train.edges)
    print('Training Graph: nodes:', node_num1, 'edges:', edge_num1)

    return graph, g_train, testing_pos_edges, train_graph_filename


def generate_neg_edges(graph: nx.Graph, m: int):
    """Get m samples from the edges in the graph that don't exist."""
    negative_edges = [
        (source, target)
        for source, target in itertools.combinations(graph, 2)
        if not graph.has_edge(source, target)
    ]

    return random.sample(negative_edges, m)


def load_embedding(embedding_file_name, node_list=None):
    with open(embedding_file_name) as f:
        node_num, _ = f.readline().split()
        embedding_look_up = {}
        if node_list:
            for line in f:
                vec = line.strip().split()
                node_id = vec[0]
                if node_id in node_list:
                    emb = [float(x) for x in vec[1:]]
                    embedding_look_up[node_id] = list(emb)

            assert len(node_list) == len(embedding_look_up)
        else:
            for line in f:
                vec = line.strip().split()
                node_id = vec[0]
                emb = [float(x) for x in vec[1:]]
                embedding_look_up[node_id] = list(emb)

            assert int(node_num) == len(embedding_look_up)
        f.close()
        return embedding_look_up


def read_node_labels(filename):
    fin = open(filename, 'r')
    node_list = []
    labels = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        node_list.append(vec[0])
        labels.append(vec[1:])
    fin.close()
    print('Nodes with labels: %s' % len(node_list))
    return node_list, labels


def split_train_test_classify(embedding_look_up, x, y, testing_ratio: float = 0.2):
    training_ratio = 1 - testing_ratio
    training_size = int(training_ratio * len(x))
    shuffle_indices = np.random.permutation(np.arange(len(x)))
    x_train = [embedding_look_up[x[shuffle_indices[i]]] for i in range(training_size)]
    y_train = [y[shuffle_indices[i]] for i in range(training_size)]
    x_test = [embedding_look_up[x[shuffle_indices[i]]] for i in range(training_size, len(x))]
    y_test = [y[shuffle_indices[i]] for i in range(training_size, len(x))]

    x_train = np.array(x_train).ravel()
    y_train = np.array(y_train).ravel()
    x_test = np.array(x_test).ravel()
    y_test = np.array(y_test).ravel()

    return x_train, y_train, x_test, y_test


def get_y_pred(y_test, y_pred_prob):
    y_pred = np.zeros(y_pred_prob.shape)
    sort_index = np.flip(np.argsort(y_pred_prob, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = np.sum(y_test[i])
        for j in range(num):
            y_pred[i][sort_index[i][j]] = 1
    return y_pred


def get_xy_sets(embeddings, graph_edges, neg_edges):
    x = []
    y = []
    for edge in graph_edges:
        node_u_emb = np.array(embeddings[edge[0]])
        node_v_emb = np.array(embeddings[edge[1]])
        feature_vector = node_u_emb * node_v_emb
        x.append(feature_vector.tolist())
        y.append(1)
    for edge in neg_edges:
        node_u_emb = np.array(embeddings[edge[0]])
        node_v_emb = np.array(embeddings[edge[1]])
        feature_vector = node_u_emb * node_v_emb
        x.append(feature_vector.tolist())
        y.append(0)

    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)
    x = np.array(x)
    y = np.array(y)
    return x, y
