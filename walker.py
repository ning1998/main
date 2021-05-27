from __future__ import print_function
import random
import numpy as np
import multiprocessing


class Walker:
    def __init__(self, G):
        self.G = G.G
        self.node_size = G.node_size
        print('node2vec walker...')
    def metapath2vec_walk(self, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        node_size = self.node_size

        meta_path=self.meta_path
        walk = [start_node]

        for t in range(1,len(meta_path)):
            Vt=meta_path[t]
            cur = walk[-1]
            cur_nbrs = [nbr for nbr in G.neighbors(cur) if Vt in G.nodes[nbr]['label']]
            if len(cur_nbrs) > 0:
                walk.append(cur_nbrs[alias_draw(alias_nodes[cur][t][0], alias_nodes[cur][t][1])])
                #alias_nodes[cur][t]是第t-1步处于cur，第t步从cur移向其各个邻居的概率，alias_draw是从这些邻居中按概率采样得到一个邻居
            else:
                break

        return walk  
            

    def simulate_walks(self, num_walks):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        V0=self.meta_path[0]
        nodes = [node for node in G.nodes() if V0 in G.nodes[node]['label']]
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.metapath2vec_walk(start_node=node))

        return walks


    def preprocess_transition_probs(self,meta_path):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        print('preprocess_transition_prob')
        G = self.G
        self.meta_path=meta_path
        alias_nodes = {}
        for node in G.nodes():
            alias_nodes[node]={}
            for t in range(1,len(meta_path)):
                Vt=meta_path[t]
                unnormalized_probs = [G[node][nbr]['weight'] for nbr in G.neighbors(node) if Vt in G.nodes[nbr]['label']]
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
                alias_nodes[node][t] = alias_setup(normalized_probs) 
        self.alias_nodes = alias_nodes
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
        q[kk] = K*prob
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
    http://shomy.top/2017/05/09/alias-method-sampling/
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
