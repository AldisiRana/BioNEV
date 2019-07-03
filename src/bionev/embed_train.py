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
        OPT1,
        OPT2,
        OPT3,
        until_layer,
        workers,
        number_walks,
        walk_length,
        dimensions,
        window_size,
        seed,
        learning_rate,
        epochs,
        hidden,
        weight_decay,
        dropout,
        gae_model_selection,
        kstep,
        weighted,
        p,
        q,
        order,
        encoder_list,
        alpha,
        beta,
        nu1,
        nu2,
        batch_size,
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
            seed=seed
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
            train_graph_filename=train_graph_filename
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
            weighted=weighted
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
        OPT1,
        OPT2,
        OPT3,
        until_layer,
        workers,
        number_walks,
        walk_length,
        dimensions,
        window_size,
        seed
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
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, hs=1, sg=1,
                     workers=workers, seed=seed)
    os.remove("random_walks.txt")
    return model

def train_embed_gae(
        *,
        learning_rate,
        epochs,
        hidden,
        dimensions,
        weight_decay,
        dropout,
        gae_model_selection,
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
        weighted,
        train_graph_filename,
        dimensions
):
    G_ = read_for_SVD(train_graph_filename, weighted=weighted)
    model = SVD_embedding(G_, size=dimensions)
    return model

def train_embed_laplacian(
        *,
        train_graph_filename,
        dimensions,
        weighted
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = lap.LaplacianEigenmaps(G_, rep_size=dimensions)
    return model

def train_embed_gf(
        *,
        train_graph_filename,
        dimensions,
        epochs,
        learning_rate,
        weight_decay,
        weighted
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
        dimensions,
        weighted
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = hope.HOPE(graph=G_, d=dimensions)
    return model

def train_embed_grarep(
        *,
        train_graph_filename,
        kstep,
        dimensions,
        weighted
):
    G_ = read_for_OpenNE(train_graph_filename, weighted=weighted)
    model = grarep.GraRep(graph=G_, Kstep=kstep, dim=dimensions)
    return model

def train_embed_deepwalk(
        *,
        train_graph_filename,
        walk_length,
        number_walks,
        dimensions,
        workers,
        window_size,
        weighted
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
        walk_length,
        number_walks,
        dimensions,
        workers,
        p,
        q,
        window_size,
        weighted
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
        window=window_size)
    return model

def train_embed_line(
        *,
        train_graph_filename,
        epochs,
        dimensions,
        order,
        weighted
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
        encoder_list,
        alpha,
        beta,
        nu1,
        nu2,
        batch_size,
        epochs,
        learning_rate,
        weighted
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
