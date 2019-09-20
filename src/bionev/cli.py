# -*- coding: utf-8 -*-

import datetime
import getpass
import json
import os
import random
import time

import click
import networkx as nx
from bionev.embed_train import embedding_training
from bionev.pipeline import do_link_prediction, do_node_classification, create_prediction_model
from bionev.utils import split_train_test_graph, train_test_graph, read_node_labels


@click.command()
@click.option('--input', required=True, help='Input graph file. Only accepted edgelist format.')
@click.option('--output', help='Output graph embedding file', default=None)
@click.option('--task', type=click.Choice(['none', 'link-prediction', 'node-classification']), default=None,
              help='Choose to evaluate the embedding quality based on a specific prediction task. '
                   'None represents no evaluation, and only run for training embedding.')
@click.option('--testingratio', default=0.2, type=float, help='Testing set ratio for prediction tasks.'
                                                              'In link prediction, it splits all the known edges; '
                                                              'in node classification, it splits all the labeled nodes.')
@click.option('--number-walks', default=32, type=int, help='Number of random walks to start at each node. '
                                                           'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
@click.option('--walk-length', default=64, type=int, help='Length of the random walk started at each node. '
                                                    'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
@click.option('--workers', default=8, type=int, help='Number of parallel processes. '
                                                'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
@click.option('--dimensions', default=100, type=int, help='the dimensions of embedding for each node.')
@click.option('--window-size', default=10, type=int,
              help='Window size of word2vec model. '
                   'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
@click.option('--epochs', default=5, type=int, help='The training epochs of LINE, SDNE and GAE')
@click.option('--p', default=1.0, type=float, help='p is a hyper-parameter for node2vec, '
                                                   'and it controls how fast the walk explores.')
@click.option('--q', default=1.0, type=float, help='q is a hyper-parameter for node2vec, '
                                        'and it controls how fast the walk leaves the neighborhood of starting node.')
@click.option('--method', required=True, type=click.Choice(['Laplacian', 'GF', 'SVD', 'HOPE', 'GraRep', 'DeepWalk',
            'node2vec', 'struc2vec', 'LINE', 'SDNE', 'GAE']),
            help='The embedding learning method')
@click.option('--label-file', default='', help='The label file for node classification')
@click.option('--negative-ratio', default=5, type=int, help='the negative ratio of LINE')
@click.option('--weighted', type=bool, default=False, help='Treat graph as weighted')
@click.option('--directed', type=bool, default=False, help='Treat graph as directed')
@click.option('--order', default=2, type=int,
    help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
@click.option('--weight-decay', type=float, default=5e-4,
              help='coefficient for L2 regularization for Graph Factorization.')
@click.option('--kstep', default=4, type=int, help='Use k-step transition probability matrix for GraRep.')
@click.option('--lr', default=0.01, type=float, help='learning rate')
@click.option('--alpha', default=0.3, type=float, help='alpha is a hyperparameter in SDNE')
@click.option('--beta', default=0, type=float, help='beta is a hyperparameter in SDNE')
@click.option('--nu1', default=1e-5, type=float, help='nu1 is a hyperparameter in SDNE')
@click.option('--nu2', default=1e-4, type=float, help='nu2 is a hyperparameter in SDNE')
@click.option('--bs', default=200, type=int, help='batch size of SDNE')
@click.option('--encoder-list', default='[1000, 128]', type=str,
              help='a list of numbers of the neuron at each encoder layer, the last number is the '
                   'dimension of the output node representation')
@click.option('--OPT1', default=True, type=bool, help='optimization 1 for struc2vec')
@click.option('--OPT2', default=True, type=bool, help='optimization 2 for struc2vec')
@click.option('--OPT3', default=True, type=bool, help='optimization 3 for struc2vec')
@click.option('--until-layer', type=int, default=6,
              help='Calculation until the layer. A hyper-parameter for struc2vec.')
@click.option('--dropout', default=0, type=float, help='Dropout rate (1 - keep probability).')
@click.option('--hidden', default=32, type=int, help='Number of units in hidden layer.')
@click.option('--gae_model_selection', default='gcn_ae', type=str,
              help='gae model selection: gcn_ae or gcn_vae')
@click.option('--eval-result-file', help='save evaluation performance')
@click.option('--seed', default=random.randint(1, 10000000), type=int, help='seed value')
@click.option('--training-edgelist', default=None, help='input training edgelist')
@click.option('--testing-edgelist', default=None, help='input testing edgelist')
@click.option('--model-path', default=None, help='save classifier model. Input filepath and name')
def main(
        input,
        output,
        task,
        testingratio,
        number_walks,
        walk_length,
        workers,
        dimensions,
        window_size,
        epochs,
        p,
        q,
        method,
        label_file,
        negative_ratio,
        weighted,
        directed,
        order,
        weight_decay,
        kstep,
        lr,
        alpha,
        beta,
        nu1,
        nu2,
        bs,
        encoder_list,
        opt1,
        opt2,
        opt3,
        until_layer,
        dropout,
        hidden,
        gae_model_selection,
        eval_result_file,
        seed,
        training_edgelist,
        testing_edgelist,
        model_path,
):
    print('#' * 70)
    print('Embedding Method: %s, Evaluation Task: %s' % (method, task))
    print('#' * 70)
    if task == 'link-prediction':
        if None not in (training_edgelist, testing_edgelist):
            G, G_train, testing_pos_edges, train_graph_filename = train_test_graph(input,
                                                                                   training_edgelist,
                                                                                   testing_edgelist,
                                                                                   weighted=weighted)
        else:
            G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(input,
                                                                                         weighted=weighted,
                                                                                         seed=seed,
                                                                                         testing_ratio=testingratio)
        time1 = time.time()
        model = embedding_training(
            method=method,
            train_graph_filename=train_graph_filename,
            OPT1=opt1,
            OPT2=opt2,
            OPT3=opt3,
            until_layer=until_layer,
            workers=workers,
            number_walks=number_walks,
            walk_length=walk_length,
            dimensions=dimensions,
            window_size=window_size,
            seed=seed,
            learning_rate=lr,
            epochs=epochs,
            hidden=hidden,
            weight_decay=weight_decay,
            dropout=dropout,
            gae_model_selection=gae_model_selection,
            kstep=kstep,
            weighted=weighted,
            p=p,
            q=q,
            order=order,
            encoder_list=encoder_list,
            alpha=alpha,
            beta=beta,
            nu1=nu1,
            nu2=nu2,
            batch_size=bs)
        embed_train_time = time.time() - time1
        print('Embedding Learning Time: %.2f s' % embed_train_time)
        if output is not None:
            model.save_embeddings(output)
        time1 = time.time()
        print('Begin evaluation...')
        if method == 'LINE':
            embeddings = model.get_embeddings_train()
        else:
            embeddings = model.get_embeddings()
        result = do_link_prediction(
            embeddings=embeddings,
            original_graph=G,
            train_graph=G_train,
            test_pos_edges=testing_pos_edges,
            seed=seed,
            save_model=model_path
        )
        eval_time = time.time() - time1
        print('Prediction Task Time: %.2f s' % eval_time)
        if None in (training_edgelist, testing_edgelist):
            os.remove(train_graph_filename)

    elif task == 'node-classification':
        if not label_file:
            raise ValueError("No input label file. Exit.")
        node_list, labels = read_node_labels(label_file)
        train_graph_filename = input
        time1 = time.time()
        model = embedding_training(
            method=method,
            train_graph_filename=train_graph_filename,
            OPT1=opt1,
            OPT2=opt2,
            OPT3=opt3,
            until_layer=until_layer,
            workers=workers,
            number_walks=number_walks,
            walk_length=walk_length,
            dimensions=dimensions,
            window_size=window_size,
            seed=seed,
            learning_rate=lr,
            epochs=epochs,
            hidden=hidden,
            weight_decay=weight_decay,
            dropout=dropout,
            gae_model_selection=gae_model_selection,
            kstep=kstep,
            weighted=weighted,
            p=p,
            q=q,
            order=order,
            encoder_list=encoder_list,
            alpha=alpha,
            beta=beta,
            nu1=nu1,
            nu2=nu2,
            batch_size=bs)
        embed_train_time = time.time() - time1
        print('Embedding Learning Time: %.2f s' % embed_train_time)
        if output is not None:
            model.save_embeddings(output)
        time1 = time.time()
        print('Begin evaluation...')
        if method == 'LINE':
            embeddings = model.get_embeddings_train()
        else:
            embeddings = model.get_embeddings()
        result = do_node_classification(
            embeddings=embeddings,
            node_list=node_list,
            labels=labels,
            testing_ratio=testingratio,
            seed=seed,
            save_model=model_path
        )
        eval_time = time.time() - time1
        print('Prediction Task Time: %.2f s' % eval_time)
    else:
        train_graph_filename = input
        time1 = time.time()
        model = embedding_training(
            method=method,
            train_graph_filename=train_graph_filename,
            OPT1=opt1,
            OPT2=opt2,
            OPT3=opt3,
            until_layer=until_layer,
            workers=workers,
            number_walks=number_walks,
            walk_length=walk_length,
            dimensions=dimensions,
            window_size=window_size,
            seed=seed,
            learning_rate=lr,
            epochs=epochs,
            hidden=hidden,
            weight_decay=weight_decay,
            dropout=dropout,
            gae_model_selection=gae_model_selection,
            kstep=kstep,
            weighted=weighted,
            p=p,
            q=q,
            order=order,
            encoder_list=encoder_list,
            alpha=alpha,
            beta=beta,
            nu1=nu1,
            nu2=nu2,
            batch_size=bs
        )
        if output is not None:
            model.save_embeddings(output)
        original_graph = nx.read_edgelist(input)
        if method == 'LINE':
            embeddings = model.get_embeddings_train()
        else:
            embeddings = model.get_embeddings()
        create_prediction_model(
            embeddings=embeddings,
            original_graph=original_graph,
            seed=seed,
            save_model=model_path
        )
        embed_train_time = time.time() - time1
        print('Embedding Learning Time: %.2f s' % embed_train_time)

    if eval_result_file and result:
        _results = dict(
            input=input,
            task=task,
            method=method,
            dimension=dimensions,
            user=getpass.getuser(),
            date=datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'),
            seed=seed,
        )

        if task == 'link-prediction':
            auc_roc, auc_pr, accuracy, f1, mcc = result
            _results['results'] = dict(
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                accuracy=accuracy,
                f1=f1,
                mcc=mcc,
            )
        else:
            accuracy, f1_micro, f1_macro = result
            _results['results'] = dict(
                accuracy=accuracy,
                f1_micro=f1_micro,
                f1_macro=f1_macro,
            )

        with open(eval_result_file, 'a+') as wf:
            print(json.dumps(_results, sort_keys=True), file=wf)


if __name__ == "__main__":
    main()
