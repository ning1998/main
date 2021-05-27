from __future__ import print_function
import time
from gensim.models import Word2Vec
from walker import *
from Graph import *

class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):
        #**kwargs用法：https://blog.csdn.net/woody891/article/details/52089549
        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        if dw:
            self.walker = BasicWalker(graph, workers=kwargs["workers"])
        else:
            self.walker = Walker(
                graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        #kwargs["window"] = 4
        kwargs["iter"] = 20
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

my_graph=Graph()
my_graph.read_edgelist('./data/edgelist.txt', weighted=False, directed=False)
my_graph.read_node_label('./data/entity2label.txt')
print(len(list(my_graph.G.nodes())))
#print(list(my_graph.G.nodes())[0:10])
print(list(my_graph.G.neighbors('63')))

p=0.8
q=1.2
num_walks=10
walk_length=4
my_node2vec=Node2vec(my_graph, path_length=walk_length, num_paths=num_walks, dim=100, p=p, q=q, dw=False)
my_node2vec.save_embeddings('./data/embedding.txt')









