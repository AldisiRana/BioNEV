# -*- coding: utf-8 -*-

import ast
import logging
import os

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from bionev.GAE.train_model import gae_model
from bionev.OpenNE import gf, grarep, hope, lap, line, node2vec, sdne
from bionev.SVD.model import SVD_embedding
from bionev.struc2vec import struc2vec
from bionev.utils import *


def embedding_training(
    *,
    method,
    train_graph_filename,
    OPT1=True,
    OPT2=True,
    OPT3=True,
    until_layer=6,
    workers=4,
    number_walks,
    walk_length,
    dimensions,
    window_size,
    learning_rate=0.01,
    epochs,
    hidden=32,
    weight_decay=5e-4,
    dropout=0,
    gae_model_selection='gcn_ae',
    kstep,
    weighted=False,
    p,
    q,
    order,
    encoder_list='[1000,128]',
    alpha,
    beta,
    nu1=1e-5,
    nu2=1e-4,
    batch_size=200,
):
    if method == 'struct2vec':
        model = train_embed_struct2vec(
            train_graph_filename=train_graph_filename,
            OPT1=OPT1,
            OPT2=OPT2,
            OPT3=OPT3,
            until_layer=until_layer,
            workers=workers,
            number_walks=number_walks,
            walk_length=walk_length,
            dimensions=dimensions,
            window_size=window_size,
        )
    elif method == 'GAE':
        model = train_embed_gae(
            learning_rate=learning_rate,
            epochs=epochs,
            hidden=hidden,
            dimensions=dimensions,
            weight_decay=weight_decay,
            dropout=dropout,
            gae_model_selection=gae_model_selection,
            train_graph_filename=train_graph_filename,
        )
    elif method == 'SVD':
        model = train_embed_svd(
            weighted=weighted,
            train_graph_filename=train_graph_filename,
            dimensions=dimensions)
    elif method == 'Laplacian':
        model = train_embed_laplacian(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            weighted=weighted,
        )
    elif method == 'GF':
        model = train_embed_gf(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weighted=weighted
        )
    elif method == 'HOPE':
        model = train_embed_hope(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            weighted=weighted
        )
    elif method == 'GraRep':
        model = train_embed_grarep(
            train_graph_filename=train_graph_filename,
            kstep=kstep,
            dimensions=dimensions,
            weighted=weighted
        )
    elif method == 'DeepWalk':
        model = train_embed_deepwalk(
            train_graph_filename=train_graph_filename,
            walk_length=walk_length,
            number_walks=number_walks,
            dimensions=dimensions,
            workers=workers,
            window_size=window_size,
            weighted=weighted
        )
    elif method == 'node2vec':
        model = train_embed_node2vec(
            train_graph_filename=train_graph_filename,
            walk_length=walk_length,
            number_walks=number_walks,
            dimensions=dimensions,
            workers=workers,
            window_size=window_size,
            weighted=weighted,
            p=p,
            q=q
        )
    elif method == 'LINE':
        model = train_embed_line(
            train_graph_filename=train_graph_filename,
            epochs=epochs,
            dimensions=dimensions,
            order=order,
            weighted=weighted
        )
    elif method == 'SDNE':
        model = train_embed_sdne(
            train_graph_filename=train_graph_filename,
            encoder_list=encoder_list,
            alpha=alpha,
            beta=beta,
            nu1=nu1,
            nu2=nu2,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weighted=weighted
        )
    return model


def train_embed_struct2vec(
    *,
    train_graph_filename,
    OPT1=True,
    OPT2=True,
    OPT3=True,
    until_layer=6,
    workers=4,
    number_walks=32,
    walk_length=64,
    dimensions=100,
    window_size=10,
):
    G_ = read_for_struc2vec(train_graph_filename)
    logging.basicConfig(filename='./src/struc2vec/struc2vec.log', filemode='w', level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    if (OPT3):
        until_layer = until_layer
    else:
        until_layer = None

    G = struc2vec.Graph(G_, workers, untilLayer=until_layer)

    if (OPT1):
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if (OPT2):
        G.create_vectors()
        G.calc_distances(compactDegree=OPT1)
    else:
        G.calc_distances_all_vertices(compactDegree=OPT1)

    print('create distances network..')
    G.create_distances_network()
    print('begin random walk...')
    G.preprocess_parameters_random_walk()

    G.simulate_walks(number_walks, walk_length)
    print('walk finished..\nLearning embeddings...')
    walks = LineSentence('random_walks.txt')
    model = Word2Vec(
        walks,
        size=dimensions,
        window=window_size,
        min_count=0,
        hs=1,
        sg=1,
        workers=workers,
    )
    os.remove("random_walks.txt")
    return model


def train_embed_gae(
    *,
    learning_rate=0.01,
    epochs=5,
    hidden=32,
    dimensions=100,
    weight_decay=5e-4,
    dropout=0,
    gae_model_selection='gcn_ae',
    train_graph_filename
):
    G_ = read_for_gae(train_graph_filename)
    # initialize necessary parameters
    model = gae_model(learning_rate, epochs, hidden, dimensions, weight_decay, dropout, gae_model_selection)
    # input the graph data
    model.train(G_)
    # save embeddings
    return model


def train_embed_svd(
    *,
    weighted=False,
    train_graph_filename,
    dimensions=100
):
    G_ = read_for_SVD(train_graph_filename, weighted=weighted)
    model = SVD_embedding(G_, size=dimensions)
    return model


def train_embed_laplacian(
    *,
    train_graph_filename,
    dimensions=100,
    weighted=False
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = lap.LaplacianEigenmaps(G_, rep_size=dimensions)
    return model


def train_embed_gf(
    *,
    train_graph_filename,
    dimensions=100,
    epochs=5,
    learning_rate=0.01,
    weight_decay=5e-4,
    weighted=False
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = gf.GraphFactorization(
        G_,
        rep_size=dimensions,
        epoch=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay)
    return model


def train_embed_hope(
    *,
    train_graph_filename,
    dimensions=100,
    weighted=False
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = hope.HOPE(graph=G_, d=dimensions)
    return model


def train_embed_grarep(
    *,
    train_graph_filename,
    kstep=4,
    dimensions=100,
    weighted=False
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = grarep.GraRep(graph=G_, Kstep=kstep, dim=dimensions)
    return model


def train_embed_deepwalk(
    *,
    train_graph_filename,
    walk_length=64,
    number_walks=32,
    dimensions=100,
    workers=4,
    window_size=10,
    weighted=False
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = node2vec.Node2vec(
        graph=G_,
        path_length=walk_length,
        num_paths=number_walks,
        dim=dimensions,
        workers=workers,
        window=window_size,
        dw=True)
    return model


def train_embed_node2vec(
    *,
    train_graph_filename,
    walk_length=64,
    number_walks=32,
    dimensions=100,
    workers=4,
    p=1.0,
    q=1.0,
    window_size=10,
    weighted=False
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = node2vec.Node2vec(
        graph=G_,
        path_length=walk_length,
        num_paths=number_walks,
        dim=dimensions,
        workers=workers,
        p=p,
        q=q,
        window=window_size,
    )
    return model


def train_embed_line(
    *,
    train_graph_filename,
    epochs=5,
    dimensions=100,
    order=2,
    weighted=False
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = line.LINE(
        G_,
        epoch=epochs,
        rep_size=dimensions,
        order=order)
    return model


def train_embed_sdne(
    *,
    train_graph_filename,
    encoder_list='[1000,128]',
    alpha=0.3,
    beta=0,
    nu1=1e-5,
    nu2=1e-4,
    batch_size=200,
    epochs=5,
    learning_rate=0.01,
    weighted=False
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    encoder_layer_list = ast.literal_eval(encoder_list)
    model = sdne.SDNE(
        G_,
        encoder_layer_list=encoder_layer_list,
        alpha=alpha,
        beta=beta,
        nu1=nu1,
        nu2=nu2,
        batch_size=batch_size,
        epoch=epochs,
        learning_rate=learning_rate)
    return model
