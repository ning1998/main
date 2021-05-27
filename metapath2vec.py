from __future__ import print_function
import time
from gensim.models import Word2Vec
from walker import *
from Graph import *
import random
class metapath2vec(object):

    def __init__(self, graph, num_paths=10,meta_paths, dim, **kwargs):
        print(kwargs)
        kwargs["workers"] = kwargs.get("workers", 1)
        self.graph = graph
        self.walker = Walker(self.graph)
        sentences=[]
        for meta_path in meta_paths:
            self.walker.preprocess_transition_probs(meta_path)
            sentences += self.walker.simulate_walks(num_walks=num_paths)
        print(len(sentences))#264060,324040
        print(sentences[0:10])
        random.shuffle(sentences)
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
        #for word in word2vec.wv:#有的症状结点未被遍历
        #    self.vectors[word] = word2vec.wv[word]
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
'''
print(len(list(my_graph.G.nodes())))
#print(list(my_graph.G.nodes())[0:10])
print(list(my_graph.G.neighbors('63')))
print(list(my_graph.G.nodes['3']['label']))
print(my_graph.G['4050']['514']['label'])
print(my_graph.G['4050']['514']['weight'])
'''
'''
walker = Walker(my_graph)
print("Preprocess transition probs...")
meta_paths=[["disease","symptom","disease"],["disease","disease","symptom","disease"],["disease","symptom","disease","disease"]]
sentences=[]
for meta_path in meta_paths:
    walker.preprocess_transition_probs(meta_path)
    sentences += walker.simulate_walks(num_walks=10)
print(len(sentences))#264060
print(sentences[0:10])
'''
meta_paths=[["disease","symptom","disease"],
            ["disease","disease","symptom","disease"],
            ["symptom","disease","disease","symptom"],
            ["disease","symptom","disease","disease"]]
num_walks=10
my_metapath2vec=metapath2vec(my_graph, num_paths=num_walks, meta_paths=meta_paths,dim=100)
my_metapath2vec.save_embeddings('./data/embedding.txt')









