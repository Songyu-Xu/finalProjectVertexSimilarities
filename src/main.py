'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='../graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='../emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	print("Read graph......")
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	# walks = list(map(str, walk) for walk in walks)
	# print(walks)
	# print(len(walks))
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return model

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)

	G.preprocess_transition_probs()
	# print(G.alias_nodes)
	# plt.figure(figsize=(15, 14))
	# pos = nx.spring_layout(G, seed=5)
	# nx.draw(G, pos, with_labels=True)
	# plt.show()

	walks = G.simulate_walks(args.num_walks, args.walk_length)
	model = learn_embeddings(walks)
	#
	# # # 降维
	# rawNodeVec = []
	# node2ind = {}
	# for i, w in enumerate(model.wv.index_to_key):
	# 	rawNodeVec.append(model.wv[w]) #词向量
	# 	node2ind[w] = i #{词语：序号}
	# rawNodeVec = np.array(rawNodeVec)
	# X_reduced = PCA(n_components=2).fit_transform(rawNodeVec)
	# print("shape before", rawNodeVec.shape)
	# print("shape after", X_reduced.shape)
	# # # 绘图
	# fig = plt.figure(figsize= (15, 10))
	# ax = fig.gca()
	# ax.set_facecolor('white')
	# ax.plot(X_reduced[:, 0], X_reduced[:, 1], '^', markersize = 7, alpha = 1, color = "red")
	# plt.savefig('../figure/test.jpg')
	# plt.show()

class ARGS:
	def __init__(self):
		input = "../graph/karate.edgelist"
		output = "../emb/karate.emb"
		dimensions = 16
		walk_length = 5
		num_walks = 10
		window_size = 3
		iter = 1
		workers = 4
		p = 1
		q = 0.5
		weighted = False
		directed = False

if __name__ == "__main__":

	# args = parse_args()
	args = ARGS()
	args.input = "../graph/karate.edgelist"
	args.output = "../emb/karate.emb"
	args.dimensions = 16
	args.walk_length = 5
	args.num_walks = 10
	args.window_size = 3
	args.iter = 1
	args.workers = 4
	args.p = 1
	args.q = 0.5
	args.weighted = False
	args.directed = False
	main(args)
