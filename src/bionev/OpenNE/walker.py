# -*- coding: utf-8 -*-

import random

import numpy as np


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)


class BasicWalker:
    def __init__(self, G, workers):
        self.G = G.G
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        walks = []
        nodes = list(self.G.nodes())
        print('Begin random walks...')
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            # print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=node))
            # pool.close()
            # pool.join()
        # print(len(walks))
        print('Walk finished...')
        return walks


class Walker:
    def __init__(self, G, p, q, update, workers):
        self.G = G.G
        self.p = p
        self.q = q
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict
        self.update = update
        self.alias_nodes = {}
        self.alias_edges = {}

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    next = cur_nbrs[alias_draw(self.alias_edges[pos][0],
                                               self.alias_edges[pos][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length, vectors):
        '''
        Repeatedly simulate random walks from each node.
        '''
        walks = []
        nodes = list(self.G.nodes())
        print('Begin random walk...')
        for walk_iter in range(num_walks):
            # print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                if self.update and node in vectors.keys():
                    continue
                walks.append(self.node2vec_walk(
                    walk_length=walk_length, start_node=node))
        print('Walk finished...')
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''

        unnormalized_probs = []
        for dst_nbr in self.G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'] / self.p)
            elif self.G.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        for node in self.G.nodes():
            if self.update and node in self.alias_nodes.keys():
                continue
            unnormalized_probs = [self.G[node][nbr]['weight']
                                  for nbr in self.G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            if norm_const == 0.0 :
                normalized_probs = unnormalized_probs
            else:
                normalized_probs = [
                    float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = alias_setup(normalized_probs)

        triads = {}

        for edge in self.G.edges():
            if self.update and edge in self.alias_edges.keys():
                continue
            self.alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
