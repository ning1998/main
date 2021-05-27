"""Graph utilities."""

# from time import time
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp
#图社区划分、最短路径、中心度：https://blog.csdn.net/weixin_39911007/article/details/111622217
#https://python-louvain.readthedocs.io/en/latest/
#__author__ = "Zhang Zhengyan"
#__email__ = "zhangzhengyan14@mails.tsinghua.edu.cn"


class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk,其中每一行第一个字母是source node, 后面是target node
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.Graph())#DiGraph是有向图，Graph是无向图
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self, filename, weighted=False, directed=False):
        if directed:
            self.G = nx.DiGraph()#nx.DiGraph()是有向图
            def read_unweighted(l):
                src, dst, this_label = l.split()
                #print(type(this_label))
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0
                self.G[src][dst]['label'] = this_label
                self.G[dst][src]['label'] = this_label

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        else:
            self.G = nx.Graph()#nx.DiGraph()是有向图
            def read_unweighted(l):
                src, dst, this_label = l.split()
                #print(type(this_label))
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0
                #self.G[src][dst]['label'] = this_label
    
            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            if vec[0] in self.G.nodes():
                self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()
