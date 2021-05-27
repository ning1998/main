import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from Graph import *
import json
import codecs
#https://blog.csdn.net/weixin_39911007/article/details/111622217
'''
# load the karate club graph
G = nx.karate_club_graph()
print(G.nodes(),len(G.edges()))
# compute the best partition
partition = community_louvain.best_partition(G)
print(partition)
with codecs.open('./partition.json', 'w','utf-8') as f:
    json.dump([partition], f,ensure_ascii=False)
print("num of cpmmunity: ",max(partition.values()) + 1)
# draw the graph
pos = nx.spring_layout(G)
print("pos ",pos)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
print("cmap ",cmap)

nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
'''
my_graph=Graph()
my_graph.read_edgelist('./data/edgelist.txt', weighted=False, directed=False)
#my_graph.read_node_label('./data/entity2label.txt')
print("my_graph.G.nodes(): ",len(my_graph.G.nodes()))
print("my_graph.G.nodedgeses(): ",len(my_graph.G.edges()))
G=my_graph.G
partition = community_louvain.best_partition(G)
with codecs.open('./partition.json', 'w','utf-8') as f:
    json.dump([partition], f,ensure_ascii=False)
print("num of cpmmunity: ",max(partition.values()) + 1)

pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=20,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()


