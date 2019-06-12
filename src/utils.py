import OpenNE.graph as og
import struc2vec.graph as sg
import random
import networkx as nx
import itertools
import numpy as np
import copy


def read_for_OpenNE(filename, weighted=False):
    G = og.Graph()
    print("Loading training graph for learning embedding...")
    G.read_edgelist(filename=filename, weighted=weighted)
    print("Graph Loaded...")
    return G


def read_for_struc2vec(filename):
    print("Loading training graph for learning embedding...")
    G = sg.load_edgelist(filename, undirected=True)
    print("Graph Loaded...")
    return G


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
        G = nx.read_weighted_edgelist(filename)
    else:
        G = nx.read_edgelist(filename)
    return G

def train_test_graph(input_edgelist, training_edgelist, testing_edgelist, weighted=False):
    if (weighted):
        G = nx.read_weighted_edgelist(input_edgelist)
        G_train = nx.read_weighted_edgelist(training_edgelist)
        G_test = nx.read_weighted_edgelist(testing_edgelist)
    else:
        G = nx.read_edgelist(input_edgelist)
        G_train = nx.read_edgelist(training_edgelist)
        G_test = nx.read_edgelist(testing_edgelist)
    testing_pos_edges = G_test.edges
    node_num1, edge_num1 = len(G_train.nodes), len(G_train.edges)
    print('Training Graph: nodes:', node_num1, 'edges:', edge_num1)

    return G, G_train, testing_pos_edges, training_edgelist

def split_train_test_graph(input_edgelist, testing_ratio=0.2, weighted=False, seed=None):
    if (weighted):
        G = nx.read_weighted_edgelist(input_edgelist)
    else:
        G = nx.read_edgelist(input_edgelist)
    node_num1, edge_num1 = len(G.nodes), len(G.edges)
    print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
    testing_edges_num = int(len(G.edges) * testing_ratio)
    if seed is not None:
        random.seed(seed)
    testing_pos_edges = random.sample(G.edges, testing_edges_num)
    G_train = copy.deepcopy(G)
    for edge in testing_pos_edges:
        node_u, node_v = edge
        if (G_train.degree(node_u) > 1 and G_train.degree(node_v) > 1):
            G_train.remove_edge(node_u, node_v)

    # G_train.remove_nodes_from(nx.isolates(G_train))
    # node_num2, edge_num2 = len(G_train.nodes), len(G_train.edges)
    # assert node_num1 == node_num2

    #train_graph_filename = 'training_edgelist.edgelist'
    #print('number of training nodes: %d' % len(G_train.nodes()))
    train_graph_filename = 'graph_train.edgelist'
    if weighted:
        nx.write_edgelist(G_train, train_graph_filename, data=['weight'])
    else:
        nx.write_edgelist(G_train, train_graph_filename, data=False)

    # with open(dataset_name + '_test_pos.edgelist', 'w') as wf:
    #     for edge in testing_pos_edges:
    #         node_u, node_v = edge
    #         wf.write('%s %s\n' % (node_u, node_v))
    #     wf.close()

    node_num1, edge_num1 = len(G_train.nodes), len(G_train.edges)
    print('Training Graph: nodes:', node_num1, 'edges:', edge_num1)

    return G, G_train, testing_pos_edges, train_graph_filename

# def edges_generator(L, iter):
#     for comb in itertools.combinations(L, iter):
#         yield comb

def generate_neg_edges(graph: nx.Graph, m: int, seed=None):
    """Get m samples from the edges in the graph that don't exist."""
    if seed is not None:
        random.seed(seed)

    negative_edges = [
        (source, target)
        for source, target in itertools.combinations(graph, 2)
        if not graph.has_edge(source, target)
    ]

    return random.sample(negative_edges, m)

def load_embedding(embedding_file_name, node_list=None):
    with open(embedding_file_name) as f:
        node_num, _ = f.readline().split()
        print('Nodes with embedding: %s'%node_num)
        embedding_look_up = {}
        if node_list:
            for line in f:
                vec = line.strip().split()
                node_id = vec[0]
                if (node_id in node_list):
                    emb = [float(x) for x in vec[1:]]
                    emb = emb / np.linalg.norm(emb)
                    emb[np.isnan(emb)] = 0
                    embedding_look_up[node_id] = list(emb)
            assert len(node_list) == len(embedding_look_up)
        else:
            for line in f:
                vec = line.strip().split()
                node_id = vec[0]
                embeddings = vec[1:]
                emb = [float(x) for x in embeddings]
                emb = emb / np.linalg.norm(emb)
                emb[np.isnan(emb)] = 0
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
    print('Nodes with labels: %s'%len(node_list))
    return node_list, labels


def split_train_test_classify(embedding_look_up, X, Y, testing_ratio=0.2, seed=0):
    state = np.random.get_state()
    training_ratio = 1 - testing_ratio
    training_size = int(training_ratio * len(X))
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X_train = [embedding_look_up[X[shuffle_indices[i]]] for i in range(training_size)]
    Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
    X_test = [embedding_look_up[X[shuffle_indices[i]]] for i in range(training_size, len(X))]
    Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    np.random.set_state(state)
    return X_train, Y_train, X_test, Y_test


def get_y_pred(y_test, y_pred_prob):
    y_pred = np.zeros(y_pred_prob.shape)
    sort_index = np.flip(np.argsort(y_pred_prob, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = np.sum(y_test[i])
        for j in range(num):
            y_pred[i][sort_index[i][j]] = 1
    return y_pred

def get_xy_sets(embedding_look_up, graph_edges, neg_edges):
    x = []
    y = []
    for edge in graph_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        x.append(feature_vector)
        y.append(1)
    for edge in neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        x.append(feature_vector)
        y.append(0)

    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)
    x = np.array(x)
    y = np.array(y)
    return x, y
